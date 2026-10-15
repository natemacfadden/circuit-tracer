"""The linear-adjoint attribution path (circuit_tracer/attribution/linear_adjoint.py)
on small random models: graphs must match the autograd path
(``model.use_linear_adjoint = False``), plus unit checks of its building blocks."""

import gc
from typing import Any

import pytest
import torch
import torch.nn as nn
from transformer_lens import HookedTransformerConfig

import circuit_tracer.attribution.attribute_transformerlens as atl
from circuit_tracer.attribution import linear_adjoint as la
from circuit_tracer.replacement_model.replacement_model_transformerlens import (
    TransformerLensReplacementModel,
)
from circuit_tracer.transcoder import SingleLayerTranscoder, TranscoderSet
from circuit_tracer.transcoder.activation_functions import TopK
from circuit_tracer.transcoder.cross_layer_transcoder import CrossLayerTranscoder
from circuit_tracer.utils import get_default_device
from tests.test_attributions_gemma import patch_tokenizer_special_ids

PROMPT = torch.tensor([0, 3, 4, 3, 2, 5, 3, 8, 9, 1, 7, 4])


@pytest.fixture(autouse=True)
def cleanup_cuda():
    yield
    torch.cuda.empty_cache()
    gc.collect()


def toy_model(
    dtype=torch.float32,
    clt=False,
    input_hook="mlp.hook_in",
    output_hook="hook_mlp_out",
    **cfg_overrides,
) -> TransformerLensReplacementModel:
    """A 3-layer Gemma-2-like model (GQA, soft cap, local attention) with random
    transcoders; ``cfg_overrides`` change the architecture."""
    params: dict[str, Any] = dict(
        n_layers=3,
        d_model=16,
        d_head=4,
        n_heads=4,
        n_key_value_heads=2,
        d_mlp=32,
        d_vocab=16,
        n_ctx=64,
        act_fn="gelu_pytorch_tanh",
        normalization_type="RMS",
        positional_embedding_type="rotary",
        rotary_dim=4,
        gated_mlp=True,
        final_rms=True,
        attn_scores_soft_cap=50.0,
        attn_types=["global", "local", "global"],
        window_size=5,
        use_local_attn=True,
        tokenizer_name="gpt2",
        original_architecture="Gemma2ForCausalLM",
        device=get_default_device(),
        dtype=dtype,
    )
    cfg = HookedTransformerConfig(**(params | cfg_overrides))
    assert cfg.device is not None
    torch.manual_seed(0)
    if clt:
        transcoders = CrossLayerTranscoder(
            n_layers=cfg.n_layers,
            d_transcoder=64,
            d_model=cfg.d_model,
            activation_function="relu",
            lazy_decoder=False,
            feature_input_hook=input_hook,
            feature_output_hook=output_hook,
            dtype=dtype,
        )
    else:
        transcoders = TranscoderSet(
            {
                layer: SingleLayerTranscoder(
                    cfg.d_model,
                    64,
                    TopK(6),
                    layer,
                    skip_connection=False,
                    dtype=dtype,
                    device=torch.device(cfg.device),
                )
                for layer in range(cfg.n_layers)
            },
            feature_input_hook=input_hook,
            feature_output_hook=output_hook,
        )
    for param in transcoders.parameters():
        nn.init.uniform_(param, a=-1, b=1)
    model = TransformerLensReplacementModel.from_config(cfg, transcoders)
    for param in model.parameters():
        nn.init.uniform_(param, a=-1, b=1)
    return model


def run(model, batch_size=4, **kwargs):
    with patch_tokenizer_special_ids(model, [0]):
        return atl.attribute(PROMPT, model, batch_size=batch_size, **kwargs)


def rel_err(a, b):
    a, b = a.double(), b.double()
    return float((a - b).norm() / b.norm())


# Both paths round to the model dtype at the same points but sum in different orders:
# 2x the unit roundoff for bf16/fp16.
_TOL = {torch.float32: 1e-5, torch.bfloat16: 2**-7, torch.float16: 2**-10}

CASES = {
    "mha_ln_normalized": dict(
        normalization_type="LN", n_key_value_heads=None, input_hook="ln2.hook_normalized"
    ),
    "mqa_resid_mid": dict(
        normalization_type="RMSPre", n_key_value_heads=1, input_hook="hook_resid_mid"
    ),
    "bf16_post_norms": dict(dtype=torch.bfloat16, use_normalization_before_and_after=True),
    "fp16_ln": dict(dtype=torch.float16, normalization_type="LN"),
    # Qwen3-like: QK norm, n_heads * d_head != d_model, transcoders writing to mlp.hook_out
    "qwen3": dict(
        d_head=8,
        rotary_dim=8,
        rotary_adjacent_pairs=False,
        use_qk_norm=True,
        act_fn="silu",
        attn_scores_soft_cap=-1.0,
        attn_types=None,
        window_size=None,
        use_local_attn=False,
        original_architecture="Qwen3ForCausalLM",
        output_hook="mlp.hook_out",
    ),
    "clt": dict(clt=True),
}


@pytest.mark.parametrize("case", list(CASES))
def test_adjoint_matches_autograd(case):
    model = toy_model(**CASES[case])
    assert la.adjoint_supported(model)
    adjoint = run(model, max_feature_nodes=None)
    model.use_linear_adjoint = False
    autograd = run(model, max_feature_nodes=None)

    assert torch.equal(adjoint.active_features, autograd.active_features)
    assert torch.equal(adjoint.selected_features, autograd.selected_features)
    assert torch.isfinite(adjoint.adjacency_matrix).all()
    err = rel_err(adjoint.adjacency_matrix, autograd.adjacency_matrix)
    assert err < _TOL[model.cfg.dtype], err


def test_causal_blocks_match_single_block():
    """Tiny key blocks send the 12-token prompt down the long-prompt path (causal block
    skipping, head-batched products, key-major pattern layout); 4 heads per kv head.
    One row per call, so rows near the start also take the short branch on that layout."""
    model = toy_model(n_key_value_heads=1)
    ref = run(model)
    block, budget = la._CAUSAL_BLOCK, atl._device_budget
    try:
        la._CAUSAL_BLOCK, atl._device_budget = 4, lambda device: 1
        graph = run(model)
    finally:
        la._CAUSAL_BLOCK, atl._device_budget = block, budget
    assert torch.equal(graph.selected_features, ref.selected_features)
    assert rel_err(graph.adjacency_matrix, ref.adjacency_matrix) < 1e-5


def test_decode_sparse_reconstruction():
    """CUDA: a segmented sum (positions without features included); elsewhere upstream's
    index_add_, bit for bit (segment_reduce would be slower and, in bf16, less exact)."""
    torch.manual_seed(0)
    pos = torch.tensor([0, 0, 2, 2, 2, 5, 3, 3, 3])
    feat = torch.tensor([1, 7, 0, 4, 31, 9, 2, 5, 6])
    for dtype, device in ((torch.float32, get_default_device()), (torch.bfloat16, "cpu")):
        tc = SingleLayerTranscoder(8, 32, TopK(3), 0, dtype=dtype, device=torch.device(device))
        for param in tc.parameters():
            nn.init.uniform_(param, a=-1, b=1)
        vals = torch.rand(len(pos), dtype=dtype)
        acts = torch.sparse_coo_tensor(torch.stack([pos, feat]), vals, (8, 32))
        acts = acts.coalesce().to(device)
        with torch.no_grad():
            recon, scaled = tc.decode_sparse(acts)
            zeros = torch.zeros(8, 8, dtype=dtype, device=device)
            index_add = zeros.index_add_(0, acts.indices()[0], scaled) + tc.b_dec
            dense = acts.to_dense() @ tc.W_dec + tc.b_dec
        torch.testing.assert_close(recon, dense, rtol=1e-2, atol=1e-2)
        if recon.device.type == "cpu":
            assert torch.equal(recon, index_add)
        else:
            torch.testing.assert_close(recon, index_add)


def test_gather_dot():
    """Against an explicit reference: the torch fallback (in several feature blocks)
    and, on CUDA, the Triton kernel; with/without accumulation and output rows."""
    torch.manual_seed(0)
    device = get_default_device()
    n_rows, n_pos, d_model, n_dec, n_feat, width = 5, 7, 16, 40, 60, 80
    grad = torch.randn(n_rows, n_pos, d_model, device=device)
    dec_vecs = torch.randn(n_dec, d_model, device=device)
    feat_idx = torch.randint(0, n_dec, (n_feat,), device=device)
    feat_pos = torch.randint(0, n_pos, (n_feat,), device=device)
    out_col = torch.randperm(width, device=device)[:n_feat]  # distinct within a call
    scale = torch.rand(n_feat, device=device)
    scores = torch.einsum("kfd,fd->kf", grad[:, feat_pos], dec_vecs[feat_idx] * scale[:, None])
    old = (la._FALLBACK_ELEMS, la._HAS_TRITON)
    for accumulate in (False, True):
        for rows in (None, torch.tensor([6, 0, 3, 9, 2], device=device)):
            init = torch.randn(n_rows if rows is None else 10, width, device=device)
            target_rows = torch.arange(n_rows, device=device) if rows is None else rows
            expected = init.clone() if accumulate else torch.zeros_like(init)
            expected[target_rows[:, None], out_col] += scores
            for triton in (False, True):
                if triton and not (la._HAS_TRITON and device.type == "cuda"):
                    continue
                idx = (lambda t: t.to(torch.int32)) if triton else (lambda t: t)
                out = init.clone() if accumulate else torch.zeros_like(init)
                try:  # fallback: 7 features per block
                    la._FALLBACK_ELEMS, la._HAS_TRITON = n_rows * d_model * 7, triton
                    la._gather_dot(
                        grad,
                        dec_vecs,
                        idx(feat_idx),
                        idx(feat_pos),
                        idx(out_col),
                        out,
                        accumulate,
                        scale,
                        None if rows is None else idx(rows),
                    )
                finally:
                    la._FALLBACK_ELEMS, la._HAS_TRITON = old
                torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)
