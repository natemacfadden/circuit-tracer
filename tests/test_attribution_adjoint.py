"""attribute() on the linear-adjoint path: memory-bounded calls, influence ranking,
offload / lazily-loaded encoders, no side effects, and the context's public tensors."""

import gc
import os
from contextlib import nullcontext
from typing import Any
from unittest.mock import patch

import pytest
import torch
from safetensors.torch import save_file

import circuit_tracer.attribution.attribute_transformerlens as atl
from circuit_tracer.attribution import influence
from circuit_tracer.attribution.linear_adjoint import LinearAdjoint
from circuit_tracer.replacement_model.replacement_model_transformerlens import (
    TransformerLensReplacementModel,
)
from circuit_tracer.transcoder import TranscoderSet
from circuit_tracer.transcoder import single_layer_transcoder as slt
from circuit_tracer.transcoder.activation_functions import TopK
from circuit_tracer.utils import get_default_device
from tests.test_linear_adjoint import PROMPT, rel_err, run, toy_model


@pytest.fixture(autouse=True)
def cleanup_cuda():
    yield
    torch.cuda.empty_cache()
    gc.collect()


def run_recording(model, budget=None, **kwargs):
    """attribute() with an optional per-call byte budget; records rows per adjoint call,
    rankers built and the transcoders' device during each call."""
    rec: dict[str, Any] = dict(rows=[], rankers=0, devices=set())
    compute, ranker_init = LinearAdjoint.compute, influence.BucketedRanker.__init__

    def compute_(adj, layers, *a):
        rec["rows"].append(len(layers))
        rec["devices"].add(next(model.transcoders.parameters()).device.type)
        return compute(adj, layers, *a)

    def ranker_init_(r, *a, **k):
        rec["rankers"] += 1
        ranker_init(r, *a, **k)

    with (
        patch.object(LinearAdjoint, "compute", compute_),
        patch.object(influence.BucketedRanker, "__init__", ranker_init_),
        (
            patch.object(atl, "_device_budget", lambda device: budget)
            if budget is not None
            else nullcontext()
        ),
    ):
        return run(model, **kwargs), rec


def row_bytes(model, graph):
    n_active = graph.active_features.shape[0]
    return 4 * (n_active + (model.cfg.n_layers + 1) * len(PROMPT))


def test_all_features_selected_bounded_calls():
    """max_feature_nodes=None: no ranker, and every call stays within the budget."""
    model = toy_model()
    model.use_linear_adjoint = False
    ref = run(model)
    model.use_linear_adjoint = True
    graph, rec = run_recording(model, budget=5 * row_bytes(model, ref), batch_size=512)
    assert rec["rankers"] == 0
    assert max(rec["rows"]) <= 5
    assert sum(rec["rows"]) == ref.active_features.shape[0] + len(graph.logit_targets)
    assert torch.equal(graph.selected_features, ref.selected_features)
    assert rel_err(graph.adjacency_matrix, ref.adjacency_matrix) < 1e-5


def test_ranker_matches_host_ranking():
    """Pruned graphs over several ranking rounds: the on-device ranker (adjacency in one
    piece, or piecewise under a small budget) == upstream's host ranking (used when the
    ranker's matrix would exceed its limit)."""
    model = toy_model(normalization_type="LN")
    kw = dict(batch_size=2, max_feature_nodes=30)
    ranked, rec = run_recording(model, **kw)
    with patch.object(atl, "_ranker_limit", lambda device: 0):
        host, rec_host = run_recording(model, **kw)
    n_rows = 30 + len(ranked.logit_targets)
    small = max(3 * (row_bytes(model, ranked) + 4 * n_rows), 2 * 4 * n_rows**2)
    assert 4 * n_rows * ranked.adjacency_matrix.shape[0] > small  # piecewise adjacency
    piecewise, rec_piece = run_recording(model, budget=small, **kw)
    assert (rec["rankers"], rec_host["rankers"], rec_piece["rankers"]) == (1, 0, 1)
    for graph in (host, piecewise):
        assert torch.equal(graph.selected_features, ranked.selected_features)
        assert rel_err(graph.adjacency_matrix, ranked.adjacency_matrix) < 1e-5


def test_ranker_rows_in_host_memory_bit_identical():
    """Ranker rows kept in host memory when the device is full (none, or only a few,
    allowed on the device) give exactly the graph of the all-device ranker."""
    model = toy_model(normalization_type="LN")
    kw = dict(batch_size=2, max_feature_nodes=30)
    ref = run(model, **kw)
    init = influence.BucketedRanker.__init__
    for limit in (0, 2000):

        def init_(r, *a, **k):
            init(r, *a, **{**k, "device_limit": limit})

        with patch.object(influence.BucketedRanker, "__init__", init_):
            graph = run(model, **kw)
        assert torch.equal(graph.selected_features, ref.selected_features)
        if next(model.parameters()).device.type == "mps":  # MPS matmuls aren't bit-reproducible
            torch.testing.assert_close(graph.adjacency_matrix, ref.adjacency_matrix)
        else:
            assert torch.equal(graph.adjacency_matrix, ref.adjacency_matrix)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_ranker_low_precision_zero_row(dtype):
    """An all-zero edge row (a feature with nothing upstream) in a half-precision model
    must not turn the influence iteration into NaN."""
    ranker = influence.BucketedRanker(3, 8, 4, [0, 2, 4, 4], torch.device("cpu"))
    logit_row = torch.tensor([[0.5, 0.2, 0.1, 0.0, 0.3, 0.1, 0.0]], dtype=dtype)
    ranker.add(0, logit_row, torch.tensor([7]), torch.tensor([3]))
    feat_rows = torch.tensor([[0.0] * 7, [0.4, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0]], dtype=dtype)
    ranker.add(1, feat_rows, torch.tensor([0, 1]), torch.tensor([0, 1]))
    inf = ranker.influences(3, torch.tensor([1.0]), 1, 4, torch.zeros(4, dtype=torch.bool), 2)
    assert torch.isfinite(inf).all()


def test_top_unvisited_breaks_ties_like_upstream():
    """Upstream: argsort(influences, descending) minus visited, ties to the lower index."""
    torch.manual_seed(0)
    inf = torch.randint(0, 4, (1000,)).float()  # many exact ties
    visited = torch.rand(1000) < 0.3
    device = get_default_device()
    for k in (1, 300, int((~visited).sum())):
        got = atl._top_unvisited(inf.to(device), visited.to(device), k).cpu()
        rank = torch.argsort(inf, descending=True, stable=True)
        assert torch.equal(got, rank[~visited[rank]][:k])


def lazy_copy(model, tmp_path) -> TransformerLensReplacementModel:
    """The same model with transcoders loaded lazily (W_enc/W_dec) from safetensors."""
    device = next(model.transcoders.parameters()).device
    transcoders = {}
    for layer, tc in enumerate(model.transcoders.transcoders):
        path = str(tmp_path / f"layer_{layer}.safetensors")
        save_file({k: v.detach().cpu().contiguous() for k, v in tc.state_dict().items()}, path)
        transcoders[layer] = slt.load_transcoder(
            path,
            layer,
            activation_fn=TopK(6),
            device=device,
            lazy_encoder=True,
            lazy_decoder=True,
        )
    tset = TranscoderSet(transcoders, "mlp.hook_in", "hook_mlp_out")
    lazy = TransformerLensReplacementModel.from_config(model.cfg, tset)
    state = {k: v for k, v in model.state_dict().items() if not k.startswith("transcoders.")}
    lazy.load_state_dict(state, strict=False)
    return lazy


def test_disk_offload_with_lazy_encoders(tmp_path):
    """offload="disk" really offloads the transcoders during the adjoint calls; encoder
    rows are then read from the offload file, never by loading a whole W_enc."""
    model = toy_model()
    ref = run(model, max_feature_nodes=12)
    lazy = lazy_copy(model, tmp_path)
    device = next(lazy.transcoders.parameters()).device
    full_loads = []
    getattr_ = slt.SingleLayerTranscoder.__getattr__

    def counting_getattr(self, name):
        if name == "W_enc" and self.lazy_encoder:
            full_loads.append(name)
        return getattr_(self, name)

    with patch.object(slt.SingleLayerTranscoder, "__getattr__", counting_getattr):
        graph, rec = run_recording(lazy, max_feature_nodes=12, offload="disk")
    assert rec["devices"] == {"meta"}
    assert next(lazy.transcoders.parameters()).device == device  # reloaded
    assert len(full_loads) == model.cfg.n_layers  # once per layer, in setup
    assert torch.equal(graph.selected_features, ref.selected_features)
    if device.type == "mps":  # MPS matmuls aren't bit-reproducible run to run
        torch.testing.assert_close(graph.adjacency_matrix, ref.adjacency_matrix)
    else:
        assert torch.equal(graph.adjacency_matrix, ref.adjacency_matrix)
    del lazy
    gc.collect()
    assert len(slt._MMAP_VIEWS) == 0  # the decoder files' memory maps go with the model


class Boom(RuntimeError):
    pass


def snapshot(model):
    """Global torch state, and every parameter/buffer and hook of the model."""
    return dict(
        grad=torch.is_grad_enabled(),
        tf32=(torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32),
        f32_precision=torch.get_float32_matmul_precision(),
        deterministic=torch.are_deterministic_algorithms_enabled(),
        default_dtype=torch.get_default_dtype(),
        env=dict(os.environ),
        tensors={
            name: (t.device, t.dtype, t.requires_grad, t.detach().cpu().clone())
            for name, t in model.state_dict(keep_vars=True).items()
        },
        hooks={
            name: (
                len(m._forward_hooks),
                len(m._forward_pre_hooks),
                len(m._backward_hooks),
                len(getattr(m, "fwd_hooks", [])),
                len(getattr(m, "bwd_hooks", [])),
            )
            for name, m in model.named_modules()
        },
    )


def assert_same(before, after):
    assert before.keys() == after.keys()
    for key, value_before in before.items():
        if key != "tensors":
            assert value_before == after[key], key
            continue
        assert value_before.keys() == after[key].keys()
        for name, (*meta, value) in value_before.items():
            *meta_after, value_after = after[key][name]
            assert meta == meta_after and torch.equal(value, value_after), name


@pytest.mark.parametrize("fail", [False, True])
def test_attribute_leaves_no_trace(fail):
    """After attribute() - also when it raises mid-run with the transcoders offloaded -
    the model and global torch state are as they were."""
    torch.backends.cuda.matmul.allow_tf32 = False
    model = toy_model()
    before = snapshot(model)
    compute, calls = LinearAdjoint.compute, []

    def failing_compute(adj, *a):
        calls.append(1)
        if len(calls) == 2:  # the first feature rows, after the logit rows
            raise Boom()
        return compute(adj, *a)

    with (
        patch.object(LinearAdjoint, "compute", failing_compute) if fail else nullcontext(),
        pytest.raises(Boom) if fail else nullcontext(),
    ):
        run(model, max_feature_nodes=10, offload="disk")
    assert_same(before, snapshot(model))


def test_context_keeps_upstream_tensors():
    """ctx.logits (all positions), decoder_vecs and encoder_vecs are upstream's tensors,
    though the adjoint context stores last-position logits, a decoder table and lazy
    encoder rows."""
    model = toy_model()
    tokens = PROMPT.to(model.cfg.device)
    lazy = model.setup_attribution(tokens)
    model.use_linear_adjoint = False
    eager = model.setup_attribution(tokens)
    assert lazy.adjoint is not None and eager.adjoint is None
    with torch.no_grad():
        logits = model(tokens)
    for ctx in (lazy, eager):
        assert torch.equal(ctx.logits, logits)
    assert torch.equal(lazy.decoder_vecs, eager.decoder_vecs)
    assert torch.equal(lazy.encoder_vecs, eager.encoder_vecs)
    fresh = model.setup_attribution(tokens)
    fresh.logits = new = torch.zeros(1, 3, 5)
    assert fresh.logits is new
