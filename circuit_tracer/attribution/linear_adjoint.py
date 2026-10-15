"""
Hand-written adjoint (transposed Jacobian) of the frozen local replacement model.

With attention patterns and normalization scales frozen and the MLPs replaced by
transcoders whose output is detached, every attribution row is the image of an
injected vector under a *fixed linear map* that only involves the residual stream,
the attention OV circuits (``W_V``/``W_O`` with frozen patterns) and diagonal
normalization rescalings. Instead of re-running autograd through a
``batch_size``-replicated forward graph, this module applies that linear map
directly, which buys:

* **No replicated forward** - only the frozen patterns/scales of a single forward
  pass are kept (a few hundred MB at most), so memory no longer grows with the
  batch size or with the stored autograd graph.
* **Layer truncation** - rows are processed sorted by layer, and a row only joins
  the backward sweep once the sweep reaches its own layer, so a target at layer
  ``l`` costs ``l`` layers rather than ``max(layer in batch)``.
* **Causal position truncation** - a target at position ``p`` only has gradient at
  positions ``<= p``; each layer only processes positions up to the largest
  target position among the rows active at that layer.
* **Fused gather-dot** - feature scores are computed by a Triton kernel that reads
  decoder vectors and gradients directly, without materializing the
  ``(batch, n_active, d_model)`` gather.

The result matches the autograd path up to floating-point reassociation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from circuit_tracer.replacement_model.replacement_model_transformerlens import (
        TransformerLensReplacementModel,
    )

_SUPPORTED_INPUT_HOOKS = ("ln2.hook_normalized", "hook_resid_mid", "mlp.hook_in")
_SUPPORTED_NORMS = ("RMS", "RMSPre", "LN", "LNPre")
# Key-position block of the causal pattern product; prompts up to this many positions
# use one dense bmm.
_CAUSAL_BLOCK = 256
# Norm factors (n_pos x d_model) up to this many elements are precomputed.
_NORM_CACHE_ELEMS = 2**21
# Max elements of the gathered operand in the non-Triton gather-dot fallback.
_FALLBACK_ELEMS = 2**26


try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover
    _HAS_TRITON = False
else:
    _HAS_TRITON = True

    @triton.jit
    def _gather_dot_kernel(
        grad,  # (rows, n_pos, d_model) gradients, arbitrary row/pos strides, unit d stride
        dec_vecs,  # (n_dec, d_model) decoder vectors, contiguous
        dec_scale,  # (n_feat,) per-feature scale of its decoder row (only read if HAS_SCALE)
        feat_idx,  # (n_feat,) row into dec_vecs
        feat_pos,  # (n_feat,) position into grad
        out_col,  # (n_feat,) column into out
        out_row,  # (rows,) row of out per row of grad (only read if HAS_ROWS)
        out,  # (rows, row_size)
        n_feat,
        n_rows,
        d_model,
        stride_grad_row,
        stride_grad_pos,
        stride_out_row,
        ACCUMULATE: tl.constexpr,
        HAS_SCALE: tl.constexpr,
        HAS_ROWS: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_F: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_f = tl.program_id(0)
        pid_k = tl.program_id(1)
        offs_f = pid_f * BLOCK_F + tl.arange(0, BLOCK_F)
        mask_f = offs_f < n_feat
        dec_row = tl.load(feat_idx + offs_f, mask=mask_f, other=0).to(tl.int64)
        grad_pos = tl.load(feat_pos + offs_f, mask=mask_f, other=0).to(tl.int64)
        offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_k = offs_k < n_rows
        offs_k64 = offs_k.to(tl.int64)

        acc = tl.zeros((BLOCK_K, BLOCK_F), dtype=tl.float32)
        grad_base = (
            grad
            + offs_k64[:, None, None] * stride_grad_row
            + grad_pos[None, :, None] * stride_grad_pos
        )
        dec_base = dec_vecs + dec_row[:, None] * d_model
        mask_kf = mask_k[:, None, None] & mask_f[None, :, None]
        if HAS_SCALE:
            feat_scale = tl.load(dec_scale + offs_f, mask=mask_f, other=0.0).to(tl.float32)
        for d0 in range(0, d_model, BLOCK_D):
            offs_d = d0 + tl.arange(0, BLOCK_D)
            mask_d = offs_d < d_model
            dec = tl.load(
                dec_base + offs_d[None, :], mask=mask_f[:, None] & mask_d[None, :], other=0.0
            ).to(tl.float32)
            if HAS_SCALE:  # == the prescaled fp32 row W_dec * a, bit for bit
                dec = dec * feat_scale[:, None]  # type: ignore[reportPossiblyUnbound]
            grad_block = tl.load(
                grad_base + offs_d[None, None, :], mask=mask_kf & mask_d[None, None, :], other=0.0
            ).to(tl.float32)
            acc += tl.sum(grad_block * dec[None, :, :], axis=2)

        col = tl.load(out_col + offs_f, mask=mask_f, other=0).to(tl.int64)
        if HAS_ROWS:
            row = tl.load(out_row + offs_k, mask=mask_k, other=0).to(tl.int64)
        else:
            row = offs_k64
        ptrs = out + row[:, None] * stride_out_row + col[None, :]
        mask = mask_k[:, None] & mask_f[None, :]
        if ACCUMULATE:
            acc += tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)
        tl.store(ptrs, acc.to(out.dtype.element_ty), mask=mask)


def _gather_dot(
    grad, dec_vecs, feat_idx, feat_pos, out_col, out, accumulate: bool, scale=None, out_rows=None
):
    """out[r[k], out_col[f]] (+)= <grad[k, feat_pos[f]], dec_vecs[feat_idx[f]] * scale[f]>
    for all rows k of grad, with ``r = out_rows`` (distinct row indices) or the identity
    (``scale`` optional). ``out_col`` must be distinct within one call (the Triton
    kernel's accumulate is a plain read-add-write); a layer group never repeats a column."""
    n_feat = feat_idx.numel()
    n_rows = grad.shape[0]
    if n_feat == 0 or n_rows == 0:
        return
    if _HAS_TRITON and grad.is_cuda:
        assert grad.stride(2) == 1 and dec_vecs.is_contiguous() and out.stride(1) == 1
        BLOCK_K = 16 if n_rows > 8 else 8
        BLOCK_F, BLOCK_D = 32, 16
        grid = (-(-n_feat // BLOCK_F), -(-n_rows // BLOCK_K))
        with torch.cuda.device(grad.device):  # Triton launches on the current device
            _gather_dot_kernel[grid](
                grad,
                dec_vecs,
                scale if scale is not None else dec_vecs,
                feat_idx,
                feat_pos,
                out_col,
                out_rows if out_rows is not None else out_col,
                out,
                n_feat,
                n_rows,
                dec_vecs.shape[1],
                grad.stride(0),
                grad.stride(1),
                out.stride(0),
                ACCUMULATE=accumulate,
                HAS_SCALE=scale is not None,
                HAS_ROWS=out_rows is not None,
                BLOCK_K=BLOCK_K,
                BLOCK_F=BLOCK_F,
                BLOCK_D=BLOCK_D,
                num_warps=4,  # type: ignore[reportCallIssue]
            )
    else:
        # torch fallback (CPU, MPS, no Triton). The gathered (rows, feats, d) operand is
        # built a block of features at a time so it stays <= _FALLBACK_ELEMS floats.
        step = max(1, _FALLBACK_ELEMS // max(1, n_rows * dec_vecs.shape[1]))
        for start in range(0, n_feat, step):
            block = slice(start, start + step)
            idx, pos, cols = feat_idx[block], feat_pos[block], out_col[block]
            dec_block = dec_vecs[idx] if scale is None else dec_vecs[idx] * scale[block, None]
            scores = torch.einsum("kfd,fd->kf", grad[:, pos].to(dec_block.dtype), dec_block)
            scores = scores.to(out.dtype)
            if out_rows is not None:
                if accumulate:
                    sub = out[out_rows]
                    sub.index_add_(1, cols, scores)
                    out[out_rows] = sub
                else:
                    out[out_rows[:, None], cols[None, :]] = scores
            elif accumulate:
                out.index_add_(1, cols, scores)
            else:
                out[:, cols] = scores


class _Norm:
    """Backward of a TransformerLens norm module with a frozen scale.

    In fp32 models the weight and ``1 / scale`` are folded into one factor. In
    bf16/fp16 models TransformerLens computes the norm in fp32 (``hook_scale`` and
    ``hook_normalized`` are fp32) and casts its output back to the model dtype;
    autograd then runs the backward in the same dtypes, and so does this class:
    ``grad * w`` in the model dtype, ``/ scale`` (and the LayerNorm centering) in
    fp32, one rounding back to the model dtype."""

    def __init__(self, module, scale: torch.Tensor, dtype: torch.dtype = torch.float32):
        self.centered = module.__class__.__name__ in ("LayerNorm", "LayerNormPre")
        w = getattr(module, "w", None)
        self.dtype = dtype
        self.low_precision = dtype not in (torch.float32, torch.float64)
        if self.low_precision:
            self.w = w.detach().to(dtype) if w is not None else None
            self.scale = scale.reshape(-1, 1).contiguous()  # fp32, as autograd sees it
            return
        # scale: (n_pos, 1)
        self.inv_scale = (1.0 / scale.reshape(-1, 1)).contiguous()
        self.w = w.detach()[None, :] if w is not None else None
        # The folded (n_pos, d) factor is cached for short prompts only: for long ones
        # it is n_layers * ~3 * n_pos * d floats (1.4 GB for Gemma-2-2B at 2k tokens),
        # so its rows are formed when used (the same products, bit for bit).
        self.factor = None
        if self.w is not None and scale.numel() * self.w.numel() <= _NORM_CACHE_ELEMS:
            self.factor = (self.w * self.inv_scale).contiguous()

    def _factor(self, idx) -> torch.Tensor:
        """``w / scale`` at positions ``idx`` (a slice or an index tensor)."""
        if self.factor is not None:
            return self.factor[idx]
        if self.w is None:
            return self.inv_scale[idx]
        return self.w * self.inv_scale[idx]

    def _low_precision_backward(self, grad, scale, weighted: bool, inplace: bool):
        """bf16/fp16 backward: ``grad`` (model dtype) -> grad wrt the norm input."""
        if weighted and self.w is not None:
            grad = grad.mul_(self.w) if inplace else grad * self.w  # rounded like autograd's
            inplace = True  # grad is now a temporary
        if self.centered:
            grad_f32 = grad.float().div_(scale)
            grad_f32.sub_(grad_f32.mean(-1, keepdim=True))
            return grad.copy_(grad_f32) if inplace else grad_f32.to(self.dtype)
        if inplace:
            return grad.div_(scale)  # computed in fp32, rounded once
        return (grad / scale).to(self.dtype)

    def full(self, grad: torch.Tensor, n_active_pos: int) -> torch.Tensor:
        """grad wrt norm input given grad wrt norm output (after the weight), at the
        first ``n_active_pos`` positions."""
        if self.low_precision:
            return self._low_precision_backward(grad, self.scale[:n_active_pos], True, False)
        return self._center_(grad * self._factor(slice(None, n_active_pos)))

    def full_(self, grad: torch.Tensor, n_active_pos: int) -> torch.Tensor:
        """``full`` computed in place in ``grad`` (a temporary); the same values bit for bit."""
        if self.low_precision:
            return self._low_precision_backward(grad, self.scale[:n_active_pos], True, True)
        return self._center_(grad.mul_(self._factor(slice(None, n_active_pos))))

    def _center_(self, grad):
        return grad.sub_(grad.mean(-1, keepdim=True)) if self.centered else grad

    def normalized(self, grad: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """grad wrt norm input given grad wrt ``hook_normalized`` at positions ``pos``."""
        if self.low_precision:  # hook_normalized is fp32: the injected values are not rounded
            grad_f32 = grad.float().div_(self.scale[pos])
            return self._center_(grad_f32).to(self.dtype)
        return self._center_(grad * self.inv_scale[pos])

    def full_at(self, grad: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """``full`` for one row per position ``pos``."""
        if self.low_precision:
            return self._low_precision_backward(grad.to(self.dtype), self.scale[pos], True, False)
        return self._center_(grad * self._factor(pos))


def _output_hook_supported(model) -> bool:
    """Features must write to a point whose gradient equals the gradient wrt
    ``resid_post``: ``hook_mlp_out``, or the MLP's own output (``mlp.hook_out``, used by
    e.g. the Qwen3 transcoders) when no post-MLP norm (``ln2_post``) sits between the two."""
    hook = model.transcoders.feature_output_hook
    if hook == "hook_mlp_out":
        return True
    return hook == "mlp.hook_out" and not any(hasattr(block, "ln2_post") for block in model.blocks)


def adjoint_supported(model: "TransformerLensReplacementModel") -> bool:
    """Whether the linear adjoint can replace autograd for this model.

    Args:
        model (TransformerLensReplacementModel): The replacement model to attribute.

    Returns:
        bool: True if the model's architecture and transcoder hooks are supported and
            ``model.use_linear_adjoint`` is not False.
    """
    cfg = model.cfg
    transcoders = model.transcoders
    return (
        getattr(model, "use_linear_adjoint", True)
        and model.backend == "transformerlens"
        and transcoders.feature_input_hook in _SUPPORTED_INPUT_HOOKS
        and _output_hook_supported(model)
        and not transcoders.skip_connection
        and cfg.normalization_type in _SUPPORTED_NORMS
        and not cfg.attn_only
        and not cfg.parallel_attn_mlp
        and not cfg.use_attn_in
        and not cfg.use_split_qkv_input
        and not cfg.use_attn_result
        and not getattr(cfg, "post_embedding_ln", False)
        and not getattr(cfg, "load_in_4bit", False)
        and cfg.original_architecture not in ("Olmo2ForCausalLM", "Olmo3ForCausalLM")
        and getattr(cfg, "attention_dir", "causal") == "causal"
    )


def _causal_key_blocks(pattern: torch.Tensor) -> list[torch.Tensor]:
    """Causal pattern (H, q, key) -> per key block [start, end) of _CAUSAL_BLOCK keys, its
    non-zero part key-major: (H, end - start, n_pos - start) = pattern[:, start:, start:end].T.
    About half of the full pattern; each block is the operand of one bmm in
    _causal_pattern_bmm. Made in the forward hook: the forward frees the raw pattern, and
    the copies do not land in (and split) the large blocks the setup frees afterwards."""
    n_pos = pattern.shape[-1]
    return [
        pattern[:, start:, start : start + _CAUSAL_BLOCK].transpose(1, 2).contiguous()
        for start in range(0, n_pos, _CAUSAL_BLOCK)
    ]


def frozen_state_hooks(model: "TransformerLensReplacementModel", store: dict):
    """Forward hooks that record the frozen patterns and norm scales of one forward pass.

    Args:
        model (TransformerLensReplacementModel): The model the forward pass runs on.
        store (dict): Filled with the recorded tensors, keyed by hook name; pass it to
            ``LinearAdjoint`` as ``frozen``.

    Returns:
        list[tuple[str, Callable]]: ``(hook name, hook)`` pairs for ``run_with_hooks``.
    """

    def _save(acts, hook):
        saved = acts.detach()
        if hook.name.endswith("hook_pattern") and saved.shape[-1] > _CAUSAL_BLOCK:
            saved = _causal_key_blocks(saved[0])
        store[hook.name] = saved
        return acts

    names = []
    for layer, block in enumerate(model.blocks):
        names.append(f"blocks.{layer}.attn.hook_pattern")
        names.append(f"blocks.{layer}.ln1.hook_scale")
        names.append(f"blocks.{layer}.ln2.hook_scale")
        if hasattr(block, "ln1_post"):
            names.append(f"blocks.{layer}.ln1_post.hook_scale")
    names.append("ln_final.hook_scale")
    return [(name, _save) for name in names]


class LinearAdjoint:
    """Applies the transposed Jacobian of the frozen replacement model to injected rows."""

    def __init__(
        self,
        model: "TransformerLensReplacementModel",
        frozen: dict,
        *,
        n_pos: int,
        row_size: int,
        n_active: int,
        error_vectors: torch.Tensor,
        token_vectors: torch.Tensor,
        decoder_vecs: torch.Tensor | None,
        encoder_to_decoder_map: torch.Tensor,
        decoder_locations: torch.Tensor,
        decoder_table: torch.Tensor | None = None,
        decoder_rows: torch.Tensor | None = None,
        decoder_scale: torch.Tensor | None = None,
        max_rows_elems: int = 2**27,
    ):
        """Decoders are either ``decoder_vecs`` (n_active, d_model), already scaled by the
        activations, or factored as ``decoder_table[decoder_rows] * decoder_scale``.

        Args:
            model (TransformerLensReplacementModel): The model the forward pass ran on.
            frozen (dict): Patterns and norm scales recorded by ``frozen_state_hooks``.
                The patterns are popped from it.
            n_pos (int): Number of positions in the prompt.
            row_size (int): Length of an attribution row (all source nodes).
            n_active (int): Number of active features.
            error_vectors (torch.Tensor): (n_layers, n_pos, d_model) transcoder errors.
            token_vectors (torch.Tensor): (n_pos, d_model) token embeddings.
            decoder_vecs (torch.Tensor | None): (n_active, d_model) scaled decoder
                vectors, or None to use the factored form.
            encoder_to_decoder_map (torch.Tensor): Active feature (column) of each
                decoder vector.
            decoder_locations (torch.Tensor): (2, n_decoders) layer and position each
                decoder vector writes to.
            decoder_table (torch.Tensor, optional): Deduplicated decoder rows.
            decoder_rows (torch.Tensor, optional): Row of ``decoder_table`` per decoder.
            decoder_scale (torch.Tensor, optional): Activation scaling each decoder.
            max_rows_elems (int): Max elements of the per-call gradient buffer; rows
                beyond it are processed in chunks.
        """
        cfg = model.cfg
        # The residual-stream gradient lives in the model dtype, as in autograd.
        dtype = cfg.dtype
        self.dtype = dtype
        self.n_layers = cfg.n_layers
        self.n_pos = n_pos
        self.d_model = cfg.d_model
        self.row_size = row_size
        self.n_active = n_active
        self.input_hook = model.transcoders.feature_input_hook
        self.error_vectors = error_vectors
        self.token_vectors = token_vectors
        if decoder_vecs is None:
            assert decoder_table is not None and decoder_rows is not None
            assert decoder_scale is not None
            self.decoder_vecs = decoder_table.contiguous()
        else:
            decoder_rows = decoder_scale = None
            self.decoder_vecs = decoder_vecs.contiguous()
        self.max_rows_elems = max_rows_elems

        self.blocks = []
        for layer, block in enumerate(model.blocks):
            attn: Any = block.attn
            if hasattr(attn, "_W_V"):
                W_V = attn._W_V  # (n_kv, d_model, d_head)
            else:
                W_V = attn.W_V
            n_kv = W_V.shape[0]
            heads_per_kv = cfg.n_heads // n_kv
            # popped so each recorded pattern is freed once its layout below exists
            # (n_layers * H * n_pos^2 floats: 3.5 GB at 2k tokens). TransformerLens casts
            # the pattern to the value dtype after the hook.
            pattern = frozen.pop(f"blocks.{layer}.attn.hook_pattern")
            if n_pos > _CAUSAL_BLOCK:
                # recorded as causal key blocks (see _causal_key_blocks): the causal
                # product batches all heads in one bmm per key block
                pattern_t = [key_block.to(W_V.dtype) for key_block in pattern]
            else:  # (n_kv, key, heads_per_kv, q): (heads_per_kv, q) merge into one bmm dim
                pattern = pattern[0].to(W_V.dtype)
                pattern_t = self._transpose_pattern(pattern, n_kv, heads_per_kv, n_pos)
                pattern_t = pattern_t.contiguous()
            del pattern
            self.blocks.append(
                dict(
                    pattern_t=pattern_t,
                    n_kv=n_kv,
                    heads_per_kv=heads_per_kv,
                    W_O=attn.W_O,  # (n_heads, d_head, d_model)
                    W_V=W_V,
                    ln1=_Norm(block.ln1, frozen[f"blocks.{layer}.ln1.hook_scale"][0], dtype),
                    ln1_post=(
                        _Norm(
                            block.ln1_post, frozen[f"blocks.{layer}.ln1_post.hook_scale"][0], dtype
                        )
                        if hasattr(block, "ln1_post")
                        else None
                    ),
                    ln2=_Norm(block.ln2, frozen[f"blocks.{layer}.ln2.hook_scale"][0], dtype),
                )
            )
        self.ln_final = _Norm(model.ln_final, frozen["ln_final.hook_scale"][0], dtype)

        # Feature decoders grouped by the layer they write to, sorted by position.
        dec_layers, dec_pos = decoder_locations
        self.dec_groups = []
        self.accumulate = bool(
            encoder_to_decoder_map.numel() != torch.unique(encoder_to_decoder_map).numel()
        )
        for layer in range(self.n_layers):
            idx = torch.nonzero(dec_layers == layer).squeeze(1)
            pos = dec_pos[idx]
            order = torch.argsort(pos, stable=True)
            idx, pos = idx[order], pos[order]
            rows = idx if decoder_rows is None else decoder_rows[idx]
            self.dec_groups.append(
                dict(
                    idx=rows.to(torch.int32).contiguous(),
                    scale=None if decoder_scale is None else decoder_scale[idx].contiguous(),
                    pos=pos.to(torch.int32).contiguous(),
                    pos_sorted=pos.cpu(),
                    col=encoder_to_decoder_map[idx].to(torch.int32).contiguous(),
                )
            )

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _transpose_pattern(pattern, n_kv, heads_per_kv, n_pos):
        """``(H, q, key)`` -> view ``A[j, key, r, q] = pattern[j * heads_per_kv + r, q, key]``."""
        return pattern.reshape(n_kv, heads_per_kv, n_pos, n_pos).permute(0, 3, 1, 2)

    def _attn_bwd(
        self, blk, grad: torch.Tensor, n_active_pos: int, n_dense: int, fresh_pos: torch.Tensor
    ) -> torch.Tensor:
        """grad: (n_rows, n_active_pos, d_model) grad wrt resid_mid/attn_out. Returns grad
        wrt resid_pre via attn.

        Rows ``[n_dense, n_rows)`` are *fresh*: they have not passed through any attention
        layer yet, so their gradient is non-zero only at their own position
        ``fresh_pos`` and their ``W_O`` product is computed at that position only."""
        n_rows = grad.shape[0]
        n_kv, heads_per_kv = blk["n_kv"], blk["heads_per_kv"]
        W_O = blk["W_O"]
        n_heads, d_head, d_model = W_O.shape
        if blk["ln1_post"] is not None:
            grad = blk["ln1_post"].full(grad, n_active_pos)
        # grad wrt z: (n_rows, n_active_pos, n_heads * d_head)
        w_o_t = W_O.reshape(n_heads * d_head, d_model).T
        grad_z = torch.empty(
            n_rows, n_active_pos, n_heads * d_head, dtype=grad.dtype, device=grad.device
        )
        if n_dense:
            torch.matmul(grad[:n_dense], w_o_t, out=grad_z[:n_dense])
        if n_dense < n_rows:
            fresh = torch.arange(n_dense, n_rows, device=grad.device)
            grad_z[n_dense:].zero_()
            grad_z[fresh, fresh_pos] = grad[fresh, fresh_pos] @ w_o_t
        # Each (n_rows, n_active_pos, *) temporary is dropped as soon as the next one
        # exists, and the ln1 backward runs in place: the call's working set is ~3 x
        # n_rows * n_active_pos * d_model floats (the gradient buffer included) instead of
        # ~6.5, which leaves more room for the ranker rows.
        del grad
        # -> (n_kv, heads_per_kv, q, n_rows, d_head) -> (n_kv, heads_per_kv * q, n_rows * d_head)
        grad_z = grad_z.view(n_rows, n_active_pos, n_kv, heads_per_kv, d_head)
        grad_z = grad_z.permute(2, 3, 1, 0, 4)
        grad_z = grad_z.reshape(n_kv, heads_per_kv * n_active_pos, n_rows * d_head)
        pattern_t = blk["pattern_t"]
        if n_active_pos <= _CAUSAL_BLOCK:
            if isinstance(pattern_t, list):  # long prompt: key block 0 as (n_kv, key, head, q)
                _, n_key, n_q = pattern_t[0].shape
                pattern_t = pattern_t[0].view(n_kv, heads_per_kv, n_key, n_q).permute(0, 2, 1, 3)
            pattern_prefix = pattern_t[:, :n_active_pos, :, :n_active_pos].reshape(
                n_kv, n_active_pos, heads_per_kv * n_active_pos
            )
            grad_v = torch.bmm(pattern_prefix, grad_z)  # (n_kv, key, n_rows * d_head)
        else:
            grad_v = self._causal_pattern_bmm(
                pattern_t,
                grad_z.view(n_kv, heads_per_kv, n_active_pos, n_rows * d_head),
                n_active_pos,
            )
        del grad_z, pattern_t
        grad_v = grad_v.view(n_kv, n_active_pos, n_rows, d_head).permute(2, 1, 0, 3)
        grad_v = grad_v.reshape(n_rows, n_active_pos, n_kv * d_head)
        W_V = blk["W_V"]  # (n_kv, d_model, d_head) -> (n_kv * d_head, d_model)
        w_v = W_V.permute(0, 2, 1).reshape(n_kv * d_head, d_model)
        grad_resid = torch.matmul(grad_v, w_v)  # (n_rows, n_active_pos, d_model)
        del grad_v
        return blk["ln1"].full_(grad_resid, n_active_pos)

    @staticmethod
    def _causal_pattern_bmm(
        pattern_blocks: list, grad_z: torch.Tensor, n_active_pos: int
    ) -> torch.Tensor:
        """grad_v[j, key] = sum_{r, q} pattern[j * heads_per_kv + r, q, key] * grad_z[j, r, q]
        for key, q < n_active_pos, with ``pattern_blocks`` the causal key blocks of
        ``_causal_key_blocks``.

        The pattern is causal (zero for q < key), so key block [start, end) only needs
        queries q >= start: the upper-triangular blocks are skipped (about half the
        products for long prompts). Each key block is one bmm over all heads; the
        heads_per_kv partial products of a kv head are then summed in order."""
        n_kv, heads_per_kv, _, n_cols = grad_z.shape
        grad_v = torch.empty(n_kv, n_active_pos, n_cols, dtype=grad_z.dtype, device=grad_z.device)
        grad_z_heads = grad_z.view(n_kv * heads_per_kv, n_active_pos, n_cols)
        for i, start in enumerate(range(0, n_active_pos, _CAUSAL_BLOCK)):
            end = min(n_active_pos, start + _CAUSAL_BLOCK)
            # (H, key in [start, end), q in [start, n_active_pos))
            pattern_block = pattern_blocks[i][:, : end - start, : n_active_pos - start]
            partial = torch.bmm(pattern_block, grad_z_heads[:, start:n_active_pos])
            partial = partial.view(n_kv, heads_per_kv, end - start, n_cols)
            if heads_per_kv == 1:
                grad_v[:, start:end] = partial[:, 0]
                continue
            torch.add(partial[:, 0], partial[:, 1], out=grad_v[:, start:end])
            for head in range(2, heads_per_kv):
                grad_v[:, start:end] += partial[:, head]
        return grad_v

    def _inject_feature_rows(self, grad, layer, rows, pos, values):
        blk = self.blocks[layer]
        if self.input_hook == "hook_resid_mid":
            grad[rows, pos] += values.to(grad.dtype)
        elif self.input_hook == "ln2.hook_normalized":
            grad[rows, pos] += blk["ln2"].normalized(values, pos)
        else:  # mlp.hook_in == ln2 output (after the weight)
            grad[rows, pos] += blk["ln2"].full_at(values, pos)

    def _score_outputs(self, grad, layer, n_rows, n_active_pos, out, rows):
        """Contract grads at the output of ``layer`` with features/errors writing there
        (row ``i`` of grad goes to row ``rows[i]`` of ``out``)."""
        if n_rows == 0:
            return
        grad_rows = grad[:n_rows, :n_active_pos]
        if grad_rows.dtype != self.error_vectors.dtype:  # autograd: grads.to(output_vecs.dtype)
            grad_rows = grad_rows.to(self.error_vectors.dtype)
        group = self.dec_groups[layer]
        n_feat = (
            int(torch.searchsorted(group["pos_sorted"], n_active_pos))
            if n_active_pos < self.n_pos
            else None
        )
        feats = slice(None, n_feat)
        _gather_dot(
            grad_rows,
            self.decoder_vecs,
            group["idx"][feats],
            group["pos"][feats],
            group["col"][feats],
            out,
            self.accumulate,
            None if group["scale"] is None else group["scale"][feats],
            rows[:n_rows],
        )
        error_col = self.n_active + layer * self.n_pos
        out[rows[:n_rows], error_col : error_col + n_active_pos] = torch.einsum(
            "kpd,pd->kp", grad_rows, self.error_vectors[layer, :n_active_pos]
        ).to(out.dtype)

    # --------------------------------------------------------------------- main
    @torch.no_grad()
    def compute(
        self, layers: torch.Tensor, positions: torch.Tensor, inject_values: torch.Tensor
    ) -> torch.Tensor:
        """Attribution rows for values injected at the given layers and positions.

        Args:
            layers (torch.Tensor): (n_rows,) layer of each row's target; ``n_layers``
                for logit rows.
            positions (torch.Tensor): (n_rows,) position of each row's target.
            inject_values (torch.Tensor): (n_rows, d_model) vector injected per row.

        Returns:
            torch.Tensor: ``(n_rows, row_size)`` attribution rows, in the input order.
        """
        n_rows = len(layers)
        device = inject_values.device
        # Each chunk writes its rows straight into ``out`` (no per-chunk row buffer and
        # scatter); columns no layer reaches stay zero.
        out = torch.zeros(n_rows, self.row_size, dtype=inject_values.dtype, device=device)
        layers = layers.to(device)
        positions = positions.to(device)
        # Sort by layer (desc) then position (desc) so active rows form a prefix.
        sort_key = layers.long() * (self.n_pos + 1) + positions.long()
        order = torch.argsort(sort_key, descending=True, stable=True)
        layers_cpu = layers[order].cpu()
        pos_cpu = positions[order].cpu()
        max_rows = max(1, self.max_rows_elems // (self.n_pos * self.d_model))
        for start in range(0, n_rows, max_rows):
            chunk = order[start : start + max_rows]
            self._compute_sorted(
                layers_cpu[start : start + max_rows],
                pos_cpu[start : start + max_rows],
                inject_values[chunk],
                positions[chunk],
                out,
                chunk,
            )
        return out

    def _compute_sorted(self, layers, positions, values, positions_dev, out, out_rows):
        """Rows already sorted by layer descending (CPU layers/positions); row ``i`` is
        written to ``out[out_rows[i]]`` (zero-initialized, distinct rows)."""
        n_rows = len(layers)
        top_layer = int(layers[0])
        n_pos_max = int(positions.max()) + 1
        device = values.device
        grad = torch.zeros(n_rows, n_pos_max, self.d_model, dtype=self.dtype, device=device)

        layer_list = layers.tolist()
        pos_list = positions.tolist()

        def active(min_layer):
            """Number of rows with layer >= min_layer (a prefix) and their max position + 1."""
            n_active_rows = 0
            n_active_pos = 0
            while n_active_rows < n_rows and layer_list[n_active_rows] >= min_layer:
                n_active_pos = max(n_active_pos, pos_list[n_active_rows] + 1)
                n_active_rows += 1
            return n_active_rows, n_active_pos

        # Logit rows: injected at the output of ln_final.
        n_logit_rows, _ = active(self.n_layers)
        if n_logit_rows:
            rows = torch.arange(n_logit_rows, device=device)
            grad[rows, positions_dev[:n_logit_rows]] += self.ln_final.full_at(
                values[:n_logit_rows], positions_dev[:n_logit_rows]
            )

        n_rows_above, n_pos_above = active(self.n_layers)
        n_dense = 0  # rows [0, n_dense) have passed through an attention layer
        for layer in range(min(top_layer, self.n_layers - 1), -1, -1):
            # grads at this layer's MLP output == grads at resid_post (rows above)
            self._score_outputs(grad, layer, n_rows_above, n_pos_above, out, out_rows)
            n_rows_here, n_pos_here = active(layer)
            if n_rows_here > n_rows_above:
                rows = torch.arange(n_rows_above, n_rows_here, device=device)
                self._inject_feature_rows(
                    grad,
                    layer,
                    rows,
                    positions_dev[n_rows_above:n_rows_here],
                    values[n_rows_above:n_rows_here],
                )
            if n_rows_here:
                grad_here = grad[:n_rows_here, :n_pos_here]
                grad_here += self._attn_bwd(
                    self.blocks[layer],
                    grad_here,
                    n_pos_here,
                    n_dense,
                    positions_dev[n_dense:n_rows_here],
                )
                n_dense = n_rows_here
            n_rows_above, n_pos_above = n_rows_here, n_pos_here

        # token embeddings
        token_col = self.n_active + self.n_layers * self.n_pos
        out[out_rows, token_col : token_col + n_pos_above] = torch.einsum(
            "kpd,pd->kp",
            grad[:, :n_pos_above].to(self.token_vectors.dtype),
            self.token_vectors[:n_pos_above],
        ).to(out.dtype)
