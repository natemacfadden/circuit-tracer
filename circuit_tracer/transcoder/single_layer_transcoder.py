import json
import os
import struct
import warnings
import weakref
from collections import deque
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from safetensors.torch import save_file
from torch import nn

from circuit_tracer.transcoder.activation_functions import JumpReLU, TopK
from circuit_tracer.utils import get_default_device


class SingleLayerTranscoder(nn.Module):
    """
    A per-layer transcoder (PLT) that replaces MLP computation with interpretable features.

    Per-layer transcoders decompose the output of a single MLP layer into sparsely active
    features that often correspond to interpretable concepts. Unlike cross-layer transcoders,
    each PLT operates independently on its assigned layer, which can result in longer paths
    through attribution graphs when features amplify across multiple layers.

    Attributes:
        d_model: Dimension of the transformer's residual stream
        d_transcoder: Number of learned features (typically >> d_model for superposition)
        layer_idx: Which transformer layer this transcoder replaces
        W_enc: Encoder weights mapping residual stream to feature space
        W_dec: Decoder weights mapping features back to residual stream
        b_enc: Encoder bias terms
        b_dec: Decoder bias terms (reconstruction baseline)
        W_skip: Optional skip connection weights (https://arxiv.org/abs/2501.18823)
        activation_function: Sparsity-inducing nonlinearity (e.g., ReLU, JumpReLU)
    """

    def __init__(
        self,
        d_model: int,
        d_transcoder: int,
        activation_function,
        layer_idx: int,
        skip_connection: bool = False,
        transcoder_path: str | None = None,
        lazy_encoder: bool = False,
        lazy_decoder: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()

        if device is None:
            device = get_default_device()

        self.d_model = d_model
        self.d_transcoder = d_transcoder
        self.layer_idx = layer_idx
        self.transcoder_path = transcoder_path
        self.lazy_encoder = lazy_encoder
        self.lazy_decoder = lazy_decoder

        if lazy_encoder or lazy_decoder:
            assert self.transcoder_path is not None, "Transcoder path must be set for lazy loading"

        if not lazy_encoder:
            self.W_enc = nn.Parameter(
                torch.zeros(d_transcoder, d_model, device=device, dtype=dtype)
            )

        if not lazy_decoder:
            self.W_dec = nn.Parameter(
                torch.zeros(d_transcoder, d_model, device=device, dtype=dtype)
            )

        self.b_enc = nn.Parameter(torch.zeros(d_transcoder, device=device, dtype=dtype))
        self.b_dec = nn.Parameter(torch.zeros(d_model, device=device, dtype=dtype))

        if skip_connection:
            self.W_skip = nn.Parameter(torch.zeros(d_model, d_model, device=device, dtype=dtype))
        else:
            self.W_skip = None

        self.activation_function = activation_function

    @property
    def device(self):
        """Get the device of the module's parameters."""
        return next(self.parameters()).device

    @property
    def dtype(self):
        """Get the dtype of the module's parameters."""
        return self.b_enc.dtype

    def __getattr__(self, name):
        """Dynamically load weights when accessed if lazy loading is enabled."""

        if name == "W_enc" and self.lazy_encoder and self.transcoder_path is not None:
            with safe_open(self.transcoder_path, framework="pt", device=str(self.device)) as f:
                return f.get_tensor("W_enc").to(self.dtype)
        elif name == "W_dec" and self.lazy_decoder and self.transcoder_path is not None:
            with safe_open(self.transcoder_path, framework="pt", device=str(self.device)) as f:
                return f.get_tensor("W_dec").to(self.dtype)

        return super().__getattr__(name)

    def _get_decoder_vectors(self, feat_ids=None):
        to_read = feat_ids if feat_ids is not None else np.s_[:]
        if not self.lazy_decoder:
            return self.W_dec[to_read].to(self.dtype)

        if isinstance(to_read, torch.Tensor):
            to_read = to_read.cpu()
        # safetensors can't index-slice straight into MPS memory: read on the host there
        read_device = "cpu" if self.device.type == "mps" else str(self.device)
        with safe_open(self.transcoder_path, framework="pt", device=read_device) as f:
            return f.get_slice("W_dec")[to_read].to(self.device, self.dtype)

    def encode(self, input_acts, apply_activation_function: bool = True):
        W_enc = self.W_enc
        pre_acts = F.linear(input_acts.to(W_enc.dtype), W_enc, self.b_enc)
        if not apply_activation_function:
            return pre_acts
        return self.activation_function(pre_acts)

    def decode(self, acts, input_acts: torch.Tensor | None = None):
        W_dec = self.W_dec
        reconstruction = acts @ W_dec + self.b_dec
        if self.W_skip is not None:
            assert input_acts is not None, (
                "Transcoder has skip connection but no input_acts were provided"
            )
            reconstruction = reconstruction + self.compute_skip(input_acts)
        return reconstruction

    def compute_skip(self, input_acts):
        if self.W_skip is not None:
            return input_acts @ self.W_skip.T
        else:
            raise ValueError("Transcoder has no skip connection")

    def forward(self, input_acts):
        transcoder_acts = self.encode(input_acts)
        decoded = self.decode(transcoder_acts, input_acts)
        # decoded = decoded.detach()
        # decoded.requires_grad = True

        return decoded

    def encode_sparse(
        self, input_acts, zero_positions: slice = slice(0, 1), return_encoders: bool = True
    ):
        """Encode and return sparse activations with active encoder vectors.

        Args:
            input_acts: Input activations
            zero_positions: slice representing the positions to zero out
            return_encoders: if False, skip gathering the encoder vectors (returns None
                in their place)

        Returns:
            sparse_acts: Sparse tensor of activations
            active_encoders: Encoder vectors for active features only
        """
        W_enc = self.W_enc
        pre_acts = F.linear(input_acts.to(W_enc.dtype), W_enc, self.b_enc)
        acts = self.activation_function(pre_acts)

        acts[zero_positions] = 0

        sparse_acts = acts.to_sparse()
        if not return_encoders:
            return sparse_acts, None
        _, feat_idx = sparse_acts.indices()
        active_encoders = W_enc[feat_idx]

        return sparse_acts, active_encoders

    def decode_sparse(
        self,
        sparse_acts,
        input_acts: torch.Tensor | None = None,
        active_decoders: torch.Tensor | None = None,
    ):
        """Decode sparse activations and return reconstruction with scaled decoder vectors.

        Args:
            sparse_acts: Sparse (n_pos, d_transcoder) activations from ``encode_sparse``
            input_acts: Input activations (required with a skip connection)
            active_decoders: optional, already-loaded ``W_dec`` rows of the active features
                (in ``self.dtype``); they are scaled **in place** and returned.

        Returns:
            reconstruction: Decoded output
            scaled_decoders: Decoder vectors scaled by activation values
        """
        pos_idx, feat_idx = sparse_acts.indices()
        values = sparse_acts.values()

        # Get decoder vectors for active features only
        if active_decoders is not None:
            scaled_decoders = active_decoders.mul_(values[:, None])
        else:
            W_dec = self._get_decoder_vectors(feat_idx.cpu())
            scaled_decoders = W_dec * values[:, None]

        n_pos = sparse_acts.shape[0]
        if sparse_acts.is_coalesced() and sparse_acts.is_cuda:
            # Indices are sorted by position: a segmented sum is deterministic (each
            # position's rows are summed in order), unlike index_add_'s CUDA atomics,
            # which made the error vectors (and so every graph) differ run to run.
            # Other devices keep upstream's index_add_: it is deterministic on CPU,
            # and there faster and (in bf16) more accurate than segment_reduce,
            # which has no MPS kernel.
            bounds = torch.searchsorted(
                pos_idx, torch.arange(n_pos + 1, device=pos_idx.device, dtype=pos_idx.dtype)
            )
            reconstruction = torch.segment_reduce(
                scaled_decoders, "sum", lengths=bounds.diff(), axis=0, unsafe=True
            ).to(sparse_acts.dtype)
        else:
            reconstruction = torch.zeros(
                n_pos, self.d_model, device=sparse_acts.device, dtype=sparse_acts.dtype
            ).index_add_(0, pos_idx, scaled_decoders)
        if self.W_skip is not None:
            assert input_acts is not None, (
                "Transcoder has skip connection but no input_acts were provided"
            )
            reconstruction = reconstruction + self.compute_skip(input_acts)
        reconstruction = reconstruction + self.b_dec

        return reconstruction, scaled_decoders

    def to_safetensors(self, save_path: str):
        """Save transcoder to safetensors format compatible with lazy loading.

        Saves the transcoder state dict to a single safetensors file with keys:
        W_enc, W_dec, b_enc, b_dec, and optionally activation_function.threshold and W_skip.

        Args:
            save_path: Path to the safetensors file to save
        """
        state_dict = {
            "W_enc": self.W_enc.cpu(),
            "W_dec": self.W_dec.cpu(),
            "b_enc": self.b_enc.cpu(),
            "b_dec": self.b_dec.cpu(),
        }

        if isinstance(self.activation_function, JumpReLU):
            state_dict["activation_function.threshold"] = self.activation_function.threshold.cpu()

        if isinstance(self.activation_function, TopK):
            state_dict["k"] = torch.tensor(self.activation_function.k)

        if self.W_skip is not None:
            state_dict["W_skip"] = self.W_skip.cpu()

        save_file(state_dict, save_path)


class TranscoderSet(nn.Module):
    """
    A collection of per-layer transcoders that enable construction of a replacement model.

    TranscoderSet manages the collection of SingleLayerTranscoders needed for this substitution,
    where each transcoder replaces the MLP computation at its corresponding layer.

    Attributes:
        transcoders: ModuleList of SingleLayerTranscoder instances, one per layer
        n_layers: Total number of layers covered
        d_transcoder: Common feature dimension across all transcoders
        feature_input_hook: Hook point where features read from (e.g., "hook_resid_mid")
        feature_output_hook: Hook point where features write to (e.g., "hook_mlp_out")
        scan_name: Optional identifier to identify corresponding feature visualization
        skip_connection: Whether transcoders include learned skip connections
    """

    def __init__(
        self,
        transcoders: dict[int, SingleLayerTranscoder],
        feature_input_hook: str,
        feature_output_hook: str,
        scan_name: str | list[str] | None = None,
    ):
        super().__init__()
        # Validate that we have continuous layers from 0 to max
        assert set(transcoders.keys()) == set(range(max(transcoders.keys()) + 1)), (
            f"Each layer should have a transcoder, but got transcoders for layers "
            f"{set(transcoders.keys())}"
        )

        self.transcoders = nn.ModuleList([transcoders[i] for i in range(len(transcoders))])
        self.n_layers = len(self.transcoders)
        self.d_transcoder = self.transcoders[0].d_transcoder

        # Verify all transcoders have the same d_transcoder
        for transcoder in self.transcoders:
            assert transcoder.d_transcoder == self.d_transcoder, (
                f"All transcoders must have the same d_transcoder, but got "
                f"{transcoder.d_transcoder} != {self.d_transcoder}"
            )

        # Store hook configuration
        self.feature_input_hook = feature_input_hook
        self.feature_output_hook = feature_output_hook
        self.scan_name = scan_name
        self.skip_connection = self.transcoders[0].W_skip is not None

    def __len__(self):
        return self.n_layers

    def __getitem__(self, idx: int) -> SingleLayerTranscoder:
        return self.transcoders[idx]  # type: ignore

    def __iter__(self) -> Iterator[SingleLayerTranscoder]:
        return iter(self.transcoders)  # type: ignore

    def apply_activation_function(self, layer_id, features):
        return self.transcoders[layer_id].activation_function(features)  # type: ignore

    def compute_skip(self, layer_id: int, inputs):
        return self.transcoders[layer_id].compute_skip(inputs)  # type: ignore

    def encode(self, input_acts):
        return torch.stack(
            [transcoder.encode(input_acts[i]) for i, transcoder in enumerate(self.transcoders)],  # type: ignore
            dim=0,
        )

    def _get_decoder_vectors(self, layer_id, features):
        return self.transcoders[layer_id]._get_decoder_vectors(features)  # type: ignore

    def select_decoder_vectors(self, features):
        if not features.is_sparse:
            features = features.to_sparse()

        all_layer_idx, all_pos_idx, all_feat_idx = features.indices()
        all_activations = features.values()
        all_scaled_decoder_vectors = []
        for unique_layer in all_layer_idx.unique():
            layer_mask = all_layer_idx == unique_layer
            feat_idx = all_feat_idx[layer_mask]
            activations = all_activations[layer_mask]

            decoder_vectors = self._get_decoder_vectors(unique_layer.item(), feat_idx)

            # Multiply each activation by its corresponding decoder vector
            scaled_decoder_vectors = activations.unsqueeze(-1) * decoder_vectors
            all_scaled_decoder_vectors.append(scaled_decoder_vectors)

        all_scaled_decoder_vectors = torch.cat(all_scaled_decoder_vectors)
        encoder_mapping = torch.arange(features._nnz(), device=features.device)

        return (
            all_pos_idx,
            all_layer_idx,
            all_feat_idx,
            all_scaled_decoder_vectors,
            encoder_mapping,
        )

    def decode(self, acts, input_acts: torch.Tensor | None):
        return torch.stack(
            [
                transcoder.decode(acts[i], None if input_acts is None else input_acts[i])
                for i, transcoder in enumerate[SingleLayerTranscoder](self.transcoders)  # type: ignore
            ],
            dim=0,
        )

    def compute_attribution_components(
        self,
        mlp_inputs: torch.Tensor,
        zero_positions: slice = slice(0, 1),
        lazy_encoders: bool = False,
        dedup_decoders: bool = False,
    ) -> dict[str, Any]:
        """Extract active features and their encoder/decoder vectors for attribution.

        Args:
            mlp_inputs: (n_layers, n_pos, d_model) tensor of MLP inputs
            zero_positions: (slice) slice indicating which positions to zero out
            lazy_encoders: if True, ``encoder_vecs`` is a :class:`LazyEncoderVectors`
                that gathers rows of ``W_enc`` on indexing instead of an
                ``(n_active, d_model)`` tensor (only a few rows are ever used).
            dedup_decoders: if True (and everything is fp32), ``decoder_vecs`` is None
                and the decoders are returned factored as ``decoder_table`` (each
                active (layer, feature)'s unscaled ``W_dec`` row once),
                ``decoder_rows`` (int32 row into the table per active feature) and
                ``decoder_scale`` (its activation), with
                ``decoder_vecs[i] == decoder_table[decoder_rows[i]] * decoder_scale[i]``
                bit-for-bit. A feature active at many positions is stored once.

        Returns:
            Dict containing all components needed for AttributionContext:
                - activation_matrix: Sparse (n_layers, n_pos, d_transcoder) activations
                - reconstruction: (n_layers, n_pos, d_model) reconstructed outputs
                - encoder_vecs: Concatenated encoder vectors for active features
                - decoder_vecs: Concatenated decoder vectors (scaled by activations)
                - encoder_to_decoder_map: Mapping from encoder to decoder indices
                - decoder_table, decoder_rows, decoder_scale: the factored decoders
                  (with ``dedup_decoders``; ``decoder_vecs`` is then None)
        """
        device = mlp_inputs.device

        reconstruction = torch.zeros_like(mlp_inputs)
        sparse_acts_list = []
        encoder_vectors = []

        for layer, transcoder in enumerate[SingleLayerTranscoder](self.transcoders):  # type: ignore
            sparse_acts, active_encoders = transcoder.encode_sparse(
                mlp_inputs[layer], zero_positions=zero_positions, return_encoders=not lazy_encoders
            )
            sparse_acts_list.append(sparse_acts)
            encoder_vectors.append(active_encoders)

        if (
            dedup_decoders
            and all(transcoder.dtype == torch.float32 for transcoder in self.transcoders)
            and mlp_inputs.dtype == torch.float32
        ):
            return self._dedup_attribution_components(
                mlp_inputs,
                sparse_acts_list,
                encoder_vectors,
                reconstruction,
                lazy_encoders,
            )

        # Fill decoder (and encoder) vectors into preallocated buffers layer by layer, so
        # the per-layer chunks and the concatenated result never coexist in full.
        n_active = sum(acts._nnz() for acts in sparse_acts_list)
        d_model = self[0].d_model
        # In the transcoders' dtype (what decode_sparse / W_enc return), not that of
        # mlp_inputs: TransformerLens hands over fp32 inputs for bf16/fp16 models too.
        decoder_vecs = torch.empty(n_active, d_model, device=device, dtype=self[0].dtype)
        encoder_vecs = (
            None
            if lazy_encoders
            else torch.empty(n_active, d_model, device=device, dtype=encoder_vectors[0].dtype)
        )
        # Lazily-loaded decoders: read every layer's active rows straight from the
        # (memory-mapped) files into ``decoder_vecs`` with a pipelined pinned upload,
        # instead of one blocking safetensors slice read + pageable copy per layer.
        jobs = []
        start = 0
        for sparse_acts in sparse_acts_list:
            jobs.append((sparse_acts.indices()[1], start))
            start += sparse_acts._nnz()
        prefetched = self._prefetch_lazy_rows(jobs, decoder_vecs)
        start = 0
        for layer, transcoder in enumerate[SingleLayerTranscoder](self.transcoders):  # type: ignore
            sparse_acts = sparse_acts_list[layer]
            end = start + sparse_acts._nnz()
            reconstruction[layer], active_decoders = transcoder.decode_sparse(
                sparse_acts,
                mlp_inputs[layer],
                active_decoders=decoder_vecs[start:end] if prefetched else None,
            )
            if not prefetched:  # else already scaled in place inside decoder_vecs
                decoder_vecs[start:end] = active_decoders
            del active_decoders
            if encoder_vecs is not None:
                encoder_vecs[start:end] = encoder_vectors[layer]
                encoder_vectors[layer] = None
            start = end

        activation_matrix = torch.stack(sparse_acts_list).coalesce()
        encoder_to_decoder_map = torch.arange(activation_matrix._nnz(), device=device)
        if lazy_encoders:
            encoder_vecs = LazyEncoderVectors(self, activation_matrix.indices())

        return {
            "activation_matrix": activation_matrix,
            "reconstruction": reconstruction,
            "encoder_vecs": encoder_vecs,
            "decoder_vecs": decoder_vecs,
            "encoder_to_decoder_map": encoder_to_decoder_map,
            "decoder_locations": activation_matrix.indices()[:2],
        }

    def _dedup_attribution_components(
        self,
        mlp_inputs,
        sparse_acts_list,
        encoder_vectors,
        reconstruction,
        lazy_encoders,
    ):
        """``compute_attribution_components`` with factored (deduplicated) decoders."""
        device = mlp_inputs.device
        d_model = self[0].d_model
        uniques, rows_list, scales = [], [], []
        n_table = 0
        for sparse_acts in sparse_acts_list:
            uniq, inverse = torch.unique(sparse_acts.indices()[1], return_inverse=True)
            uniques.append((uniq, n_table))
            rows_list.append(inverse.to(torch.int32) + n_table)
            scales.append(sparse_acts.values())
            n_table += uniq.numel()
        table = torch.empty(n_table, d_model, device=device, dtype=mlp_inputs.dtype)
        if not self._prefetch_lazy_rows(uniques, table):
            for transcoder, (uniq, offset) in zip(self, uniques):
                table[offset : offset + uniq.numel()] = transcoder._get_decoder_vectors(uniq.cpu())
        n_active = sum(rows.numel() for rows in rows_list)
        encoder_vecs = None if lazy_encoders else torch.empty(n_active, d_model, device=device)
        start = 0
        for layer, transcoder in enumerate[SingleLayerTranscoder](self.transcoders):  # type: ignore
            sparse_acts = sparse_acts_list[layer]
            end = start + sparse_acts._nnz()
            # same rows, scaled in place by decode_sparse exactly as in the dense path
            active_decoders = table[rows_list[layer]]
            reconstruction[layer], _ = transcoder.decode_sparse(
                sparse_acts, mlp_inputs[layer], active_decoders=active_decoders
            )
            del active_decoders
            if encoder_vecs is not None:
                encoder_vecs[start:end] = encoder_vectors[layer]
                encoder_vectors[layer] = None
            start = end

        activation_matrix = torch.stack(sparse_acts_list).coalesce()
        if lazy_encoders:
            encoder_vecs = LazyEncoderVectors(self, activation_matrix.indices())
        return {
            "activation_matrix": activation_matrix,
            "reconstruction": reconstruction,
            "encoder_vecs": encoder_vecs,
            "decoder_vecs": None,
            "decoder_table": table,
            "decoder_rows": torch.cat(rows_list),
            "decoder_scale": torch.cat(scales),
            "encoder_to_decoder_map": torch.arange(activation_matrix._nnz(), device=device),
            "decoder_locations": activation_matrix.indices()[:2],
        }

    def _prefetch_lazy_rows(self, jobs, out: torch.Tensor) -> bool:
        """``out[offset : offset + len(ids)] = W_dec_layer[ids]`` (unscaled rows) for each
        layer's ``(ids, offset)``, read straight from the memory-mapped files with a
        pipelined pinned upload.

        Only for lazily-loaded decoders on CUDA whose file dtype equals both the
        transcoder dtype and ``out.dtype`` (so the result is bit-identical to the
        per-layer path). Returns False (doing nothing) otherwise.
        """
        if not out.is_cuda:
            return False
        views = []
        for transcoder in self.transcoders:
            if not transcoder.lazy_decoder or transcoder.dtype != out.dtype:
                return False
            view = _safetensors_mmap_view(transcoder, "W_dec")
            if view is None or view.dtype != out.dtype:
                return False
            views.append(view)
        _upload_rows_pipelined(
            [(view, ids.cpu(), offset) for view, (ids, offset) in zip(views, jobs)], out
        )
        return True

    def encode_layer(self, x, layer_id, apply_activation_function=True):
        return self.transcoders[layer_id].encode(
            x, apply_activation_function=apply_activation_function
        )  # type: ignore

    def to_safetensors(self, save_dir: str):
        """Save all transcoders in the set to safetensors files.

        Saves each transcoder as layer_{i}.safetensors in the specified directory.

        Args:
            save_dir: Directory path where the safetensors files will be saved
        """
        os.makedirs(save_dir, exist_ok=True)

        for i, transcoder in enumerate(self.transcoders):
            save_path = os.path.join(save_dir, f"layer_{i}.safetensors")
            transcoder.to_safetensors(save_path)  # type: ignore


def load_gemma_scope_transcoder(
    path: str,
    layer: int,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    revision: str | None = None,
    **kwargs,
) -> SingleLayerTranscoder:
    if device is None:
        device = get_default_device()
    if os.path.isfile(path):
        path_to_params = path
    else:
        path_to_params = hf_hub_download(
            repo_id="google/gemma-scope-2b-pt-transcoders",
            filename=path,
            revision=revision,
            force_download=False,
        )

    # load the parameters, have to rename the threshold key,
    # as ours is nested inside the activation_function module
    param_dict = np.load(path_to_params)
    param_dict = {k: torch.tensor(v, device=device, dtype=dtype) for k, v in param_dict.items()}
    param_dict["activation_function.threshold"] = param_dict["threshold"]
    param_dict["W_enc"] = param_dict["W_enc"].T.contiguous()
    del param_dict["threshold"]

    # create the transcoders
    d_transcoder, d_model = param_dict["W_enc"].shape

    # JumpReLU; will get loaded via load_state_dict
    activation_function = JumpReLU(param_dict["activation_function.threshold"], 0.1)
    with torch.device("meta"):
        transcoder = SingleLayerTranscoder(d_model, d_transcoder, activation_function, layer)
    transcoder.load_state_dict(param_dict, assign=True)
    return transcoder


def load_transcoder(
    path: str,
    layer: int,
    activation_fn=None,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    lazy_encoder: bool = True,
    lazy_decoder: bool = True,
):
    if device is None:
        device = get_default_device()

    param_dict = {}
    with safe_open(path, framework="pt", device=str(device)) as f:
        for k in f.keys():
            if lazy_encoder and k == "W_enc":
                continue
            if lazy_decoder and k == "W_dec":
                continue
            param_dict[k] = f.get_tensor(k)

    d_sae = param_dict["b_enc"].shape[0]
    d_model = param_dict["b_dec"].shape[0]

    # JumpReLU
    if activation_fn is None:
        if "activation_function.threshold" in param_dict:
            activation_function = JumpReLU(param_dict["activation_function.threshold"], 0.1)
        else:
            activation_function = F.relu
    else:
        activation_function = activation_fn

    with torch.device("meta"):
        transcoder = SingleLayerTranscoder(
            d_model,
            d_sae,
            activation_function,
            layer,
            skip_connection=param_dict.get("W_skip") is not None,
            transcoder_path=path,
            lazy_encoder=lazy_encoder,
            lazy_decoder=lazy_decoder,
        )
    transcoder.load_state_dict(param_dict, assign=True)
    return transcoder.to(dtype)


def load_gemma_scope_2_transcoder(
    path: str,
    layer: int,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    lazy_encoder: bool = False,
    lazy_decoder: bool = False,
    **kwargs,
) -> SingleLayerTranscoder:
    """Load a SingleLayerTranscoder from a GemmaScope2 JumpReLUSAE checkpoint.

    Args:
        path: Path to the checkpoint file
        layer: Layer index for the transcoder
        device: Device to load to
        dtype: Data type to use
        lazy_encoder: Whether to use lazy loading for encoder weights (not supported for GemmaScope2 format)
        lazy_decoder: Whether to use lazy loading for decoder weights (not supported for GemmaScope2 format)

    Returns:
        SingleLayerTranscoder: The loaded transcoder
    """
    if device is None:
        device = get_default_device()

    if lazy_encoder or lazy_decoder:
        warnings.warn(
            "Lazy loading is not supported for GemmaScope2 format due to different key naming conventions. "
            "Setting lazy_encoder=False and lazy_decoder=False. If you wish to use lazy loading, please "
            "cache the relevant transcoders via circuit_tracer.utils.caching.save_transcoders_to_cache",
            UserWarning,
        )
        lazy_encoder = False
        lazy_decoder = False

    with safe_open(path, framework="pt", device=device.type) as f:
        state_dict = {k: f.get_tensor(k) for k in f.keys()}

    param_dict = {
        "W_enc": state_dict["w_enc"].T.contiguous().to(device=device, dtype=dtype),
        "W_dec": state_dict["w_dec"].to(device=device, dtype=dtype),
        "b_enc": state_dict["b_enc"].to(device=device, dtype=dtype),
        "b_dec": state_dict["b_dec"].to(device=device, dtype=dtype),
        "activation_function.threshold": state_dict["threshold"].to(device=device, dtype=dtype),
    }

    if "affine_skip_connection" in state_dict:
        param_dict["W_skip"] = (
            state_dict["affine_skip_connection"].T.contiguous().to(device=device, dtype=dtype)
        )

    d_transcoder = param_dict["b_enc"].shape[0]
    d_model = param_dict["b_dec"].shape[0]

    activation_function = JumpReLU(param_dict["activation_function.threshold"], 0.1)

    with torch.device("meta"):
        transcoder = SingleLayerTranscoder(
            d_model,
            d_transcoder,
            activation_function,
            layer,
            skip_connection="W_skip" in param_dict,
        )

    transcoder.load_state_dict(param_dict, assign=True)
    return transcoder


def load_transcoder_set(
    transcoder_paths: dict,
    scan_name: str,
    feature_input_hook: str,
    feature_output_hook: str,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    special_load_fn: Literal["gemma-scope", "gemma-scope-2", None] = None,
    # Activation function config; k is only used when activation="topk"
    activation: str | None = None,
    k: int | None = None,
    lazy_encoder: bool = True,
    lazy_decoder: bool = True,
) -> TranscoderSet:
    if device is None:
        device = get_default_device()
    """Loads either a preset set of transcoders, or a set specified by a file.

    Args:
        transcoder_paths: Dictionary mapping layer indices to transcoder paths
        scan_name: Scan identifier
        feature_input_hook: Hook point where features read from
        feature_output_hook: Hook point where features write to
        device (torch.device | None, optional): Device to load to
        dtype (torch.dtype | None, optional): Data type to use
        special_load_fn: Which special loading function to use
        config: The config file
        lazy_encoder: Whether to use lazy loading for encoder weights
        lazy_decoder: Whether to use lazy loading for decoder weights

    Returns:
        TranscoderSet: The loaded transcoder set with all configuration
    """

    if activation == "topk":
        if scan_name == "facebook/crv-8b-instruct-transcoders":
            warnings.warn(
                """This top-k transcoder (facebook/crv-8b-instruct-transcoders) has a hardcoded value of k = 128.
                In general, k should be set in the config.yaml"""
            )
            k = 128
        assert k is not None, "You must pass k if activation is topk"
        activation_fn = TopK(k)
    elif activation == "relu":
        activation_fn = F.relu
    else:
        # For JumpReLU (and potentially others), we load the log-thresholds from weights
        activation_fn = None

    transcoders = {}
    for layer in range(len(transcoder_paths)):
        npz_format = Path(transcoder_paths[layer]).suffix == ".npz"

        if special_load_fn == "gemma-scope" and npz_format:
            load_fn = load_gemma_scope_transcoder
        elif special_load_fn == "gemma-scope-2":
            load_fn = load_gemma_scope_2_transcoder
        else:
            load_fn = load_transcoder

        transcoders[layer] = load_fn(
            transcoder_paths[layer],
            layer,
            activation_fn=activation_fn,
            device=device,
            dtype=dtype,
            lazy_encoder=lazy_encoder,
            lazy_decoder=lazy_decoder,
        )
    # we don't know how many layers the model has, but we need all layers from 0 to max covered
    assert set(transcoders.keys()) == set(range(max(transcoders.keys()) + 1)), (
        f"Each layer should have a transcoder, but got transcoders for layers "
        f"{set(transcoders.keys())}"
    )

    return TranscoderSet(
        transcoders,
        feature_input_hook=feature_input_hook,
        feature_output_hook=feature_output_hook,
        scan_name=scan_name,
    )


_ST_DTYPES = {"F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16}
# transcoder -> {(path, name, size, mtime): view}; entries go away with their transcoder
_MMAP_VIEWS: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


def _safetensors_mmap_view(transcoder, name: str) -> torch.Tensor | None:
    """Zero-copy CPU view (copy-on-write mmap) of tensor ``name`` in the transcoder's
    safetensors file.

    Views are cached per transcoder (weakly: the mapping is dropped when the transcoder
    is garbage-collected) and per (path, size, mtime), so the page tables are only built
    once per model. Returns None if the file/tensor cannot be mapped (callers fall back).
    """
    path = transcoder.transcoder_path
    if path is None:
        return None
    try:
        stat = os.stat(path)
        key = (path, name, stat.st_size, stat.st_mtime_ns)
        views = _MMAP_VIEWS.setdefault(transcoder, {})
        view = views.get(key)
        if view is None:
            with open(path, "rb") as fh:
                n_header = struct.unpack("<Q", fh.read(8))[0]
                header = json.loads(fh.read(n_header))
            meta = header[name]
            dtype = _ST_DTYPES[meta["dtype"]]
            begin, end = meta["data_offsets"]
            raw = np.memmap(
                path, dtype=np.uint8, mode="c", offset=8 + n_header + begin, shape=(end - begin,)
            )
            view = torch.from_numpy(raw).view(dtype).view(meta["shape"])
            for stale in [k for k in views if k[1] == name]:  # the file was replaced
                del views[stale]
            views[key] = view
        return view
    except Exception:
        return None


def _upload_rows_pipelined(jobs, out: torch.Tensor, chunk_rows: int = 4096, n_bufs: int = 3):
    """``out[offset : offset + len(idx)] = src[idx]`` for each ``(src, idx, offset)`` in
    ``jobs``.

    ``src`` are CPU tensors, ``idx`` CPU index tensors, ``out`` a CUDA tensor. A worker
    thread gathers chunks of rows into a small ring of pinned buffers while the
    previous chunks are copied to the device asynchronously.
    """
    chunks = [
        (src, idx[start : start + chunk_rows], offset + start)
        for src, idx, offset in jobs
        for start in range(0, len(idx), chunk_rows)
    ]
    if not chunks:
        return
    n_bufs = min(n_bufs, len(chunks))
    bufs = [
        torch.empty(chunk_rows, out.shape[1], dtype=out.dtype, pin_memory=True)
        for _ in range(n_bufs)
    ]

    def gather(i):
        src, idx, _ = chunks[i]
        buf = bufs[i % n_bufs][: len(idx)]
        torch.index_select(src, 0, idx, out=buf)
        return buf

    # events and the final sync must use out's device (the model may not be on the
    # current device)
    with torch.cuda.device(out.device), ThreadPoolExecutor(max_workers=1) as executor:
        futures = deque(executor.submit(gather, i) for i in range(n_bufs))
        for i in range(len(chunks)):
            buf = futures.popleft().result()
            _, idx, offset = chunks[i]
            out[offset : offset + len(idx)].copy_(buf, non_blocking=True)
            copied = torch.cuda.Event()
            copied.record()
            next_chunk = i + n_bufs
            if next_chunk < len(chunks):
                copied.synchronize()  # buffer i % n_bufs is reused by chunk next_chunk
                futures.append(executor.submit(gather, next_chunk))
        torch.cuda.current_stream().synchronize()  # pinned buffers are freed on return


_BITS = {1: np.uint8, 2: np.int16, 4: np.int32, 8: np.int64}
_TORCH_BITS = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}


def _bits(tensor: torch.Tensor) -> np.ndarray:
    """A CPU tensor's bits as a numpy array (numpy has no bfloat16)."""
    return tensor.detach().view(_TORCH_BITS[tensor.itemsize]).numpy()


class LazyEncoderVectors:
    """Stand-in for the ``(n_active, d_model)`` encoder-vector tensor of a TranscoderSet.

    Indexing with active-feature indices gathers the corresponding rows of each layer's
    ``W_enc`` (bit-identical to the materialized tensor). Lazily-loaded encoders are read
    row by row from their safetensors file instead of loading the whole ``W_enc``.
    ``materialize()`` builds the full tensor.

    Args:
        transcoder_set (TranscoderSet): The transcoders whose ``W_enc`` rows are read.
        indices (torch.Tensor): (3, n_active) ``(layer, pos, feature)`` indices of the
            active features (the coalesced activation matrix's indices).
    """

    def __init__(self, transcoder_set: "TranscoderSet", indices: torch.Tensor):
        self.transcoder_set = transcoder_set
        self.layers = indices[0]
        self.features = indices[2]
        self.d_model = transcoder_set[0].d_model
        # dtype of W_enc as the eager path returns it (lazy W_enc is cast to the
        # transcoder's dtype on load); read without loading lazy weights
        self.dtype = (
            transcoder_set[0].dtype
            if transcoder_set[0].lazy_encoder
            else transcoder_set[0].W_enc.dtype
        )
        # per layer: None (read the module's W_enc) or (safetensors path, tensor name)
        self._files: list = [
            (transcoder.transcoder_path, "W_enc")
            if transcoder.lazy_encoder and transcoder.transcoder_path
            else None
            for transcoder in transcoder_set
        ]
        self._host_index = None  # (layers, features) as numpy arrays, for host gathers

    def __len__(self):
        return self.layers.shape[0]

    @property
    def shape(self):
        """``(n_active, d_model)``, like the tensor this stands in for."""
        return torch.Size((len(self), self.d_model))

    def read_from_offload_file(self, path: str):
        """Read encoder rows from ``path`` while the transcoders themselves are offloaded.

        Args:
            path (str): A ``disk``-offloaded copy of the whole TranscoderSet's state dict.
        """
        names = {id(module): name for name, module in self.transcoder_set.named_modules()}
        with safe_open(path, framework="pt", device="cpu") as f:
            keys = set(f.keys())
        for layer, transcoder in enumerate(self.transcoder_set.transcoders):
            prefix = names[id(transcoder)]
            key = f"{prefix}.W_enc" if prefix else "W_enc"
            if key in keys:
                self._files[layer] = (path, key)
            elif self._files[layer] is None:
                raise RuntimeError(f"W_enc of layer {layer} is not in the offload file {path}")

    def _rows(self, layer: int, feats: torch.Tensor) -> torch.Tensor:
        src = self._files[layer]
        if src is None:
            W_enc = self.transcoder_set[layer].W_enc
            return W_enc[feats.to(W_enc.device)]
        path, key = src
        with safe_open(path, framework="pt", device="cpu") as f:
            return f.get_slice(key)[feats.cpu()].to(self.dtype)

    def _on_host(self, layer: int) -> bool:
        src = self._files[layer]
        return src is not None or self.transcoder_set[layer].W_enc.device.type == "cpu"

    def __getitem__(self, idx) -> torch.Tensor:
        """Encoder vectors of the active features ``idx``.

        Args:
            idx: Indices into the active features (a tensor or anything
                ``torch.as_tensor`` accepts).

        Returns:
            torch.Tensor: ``(len(idx), d_model)`` rows on the activations' device.
        """
        # rows are returned on the activations' device even if the transcoders have been
        # offloaded (attribute(..., offload="cpu"/"disk"))
        device = self.layers.device
        if device.type != "cpu" and any(self._on_host(layer) for layer in range(len(self._files))):
            return self._gather_on_host(torch.as_tensor(idx).cpu(), device)
        idx = torch.as_tensor(idx, device=device)
        layers, feats = self.layers[idx], self.features[idx]
        out = torch.empty(layers.shape[0], self.d_model, device=device, dtype=self.dtype)
        for layer in layers.unique().tolist():
            mask = layers == layer
            out[mask] = self._rows(layer, feats[mask]).to(out.device)
        return out

    def _gather_on_host(self, idx: torch.Tensor, device) -> torch.Tensor:
        """Rows of host-resident or file-backed encoders (offloaded or lazily loaded),
        gathered on the host and uploaded in one copy.

        The gather runs in numpy on raw bits: a handful of rows per layer is far below
        what is worth a parallel torch kernel, whose thread wake-ups would dominate."""
        if self._host_index is None:
            self._host_index = (self.layers.cpu().numpy(), self.features.cpu().numpy())
        layers, feats = (index[idx.numpy()] for index in self._host_index)
        out = np.empty((len(layers), self.d_model), dtype=_BITS[self.dtype.itemsize])
        on_device = []
        for layer in np.unique(layers).tolist():
            sel = np.flatnonzero(layers == layer)
            if self._files[layer] is None:
                W_enc = self.transcoder_set[layer].W_enc
                if W_enc.device.type != "cpu":  # this layer's encoder is still on the device
                    on_device.append((sel, W_enc[torch.from_numpy(feats[sel]).to(W_enc.device)]))
                    continue
                if W_enc.dtype == self.dtype:
                    out[sel] = _bits(W_enc)[feats[sel]]
                else:
                    out[sel] = _bits(W_enc[torch.from_numpy(feats[sel])].to(self.dtype))
            else:
                out[sel] = _bits(self._rows(layer, torch.from_numpy(feats[sel])))
        rows = torch.from_numpy(out).view(self.dtype).to(device)
        for sel, device_rows in on_device:
            rows[torch.from_numpy(sel).to(device)] = device_rows.to(device)
        return rows

    def materialize(self) -> torch.Tensor:
        """The full encoder-vector tensor.

        Returns:
            torch.Tensor: The ``(n_active, d_model)`` tensor the eager path would have built.
        """
        return self[torch.arange(len(self), device=self.layers.device)]
