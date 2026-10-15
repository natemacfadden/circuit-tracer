"""
Build an **attribution graph** that captures the *direct*, *linear* effects
between features and next-token logits for a *prompt-specific*
**local replacement model**.

High-level algorithm (matches the 2025 ``Attribution Graphs`` paper):
https://transformer-circuits.pub/2025/attribution-graphs/methods.html

1. **Local replacement model** - we configure gradients to flow only through
   linear components of the network, effectively bypassing attention mechanisms,
   MLP non-linearities, and layer normalization scales.
2. **Forward pass** - record residual-stream activations and mark every active
   feature.
3. **Backward passes** - for each source node (feature or logit), inject a
   *custom* gradient that selects its encoder/decoder direction.  Because the
   model is linear in the residual stream under our freezes, this contraction
   equals the *direct effect* A_{s->t}.
4. **Assemble graph** - store edge weights in a dense matrix and package a
   ``Graph`` object.  Downstream utilities can *prune* the graph to the subset
   needed for interpretation.
"""

import logging
import time
from collections.abc import Sequence
from typing import Literal

import torch
from tqdm import tqdm

from circuit_tracer.attribution.targets import (
    AttributionTargets,
    TargetSpec,
    log_attribution_target_info,
)
from circuit_tracer.attribution.influence import BucketedRanker
from circuit_tracer.graph import Graph, compute_partial_influences
from circuit_tracer.replacement_model.replacement_model_transformerlens import (
    TransformerLensReplacementModel,
)
from circuit_tracer.transcoder.single_layer_transcoder import LazyEncoderVectors
from circuit_tracer.utils.disk_offload import offload_modules


def attribute(
    prompt: str | torch.Tensor | list[int],
    model: TransformerLensReplacementModel,
    *,
    attribution_targets: Sequence[str] | Sequence[TargetSpec] | torch.Tensor | None = None,
    max_n_logits: int = 10,
    desired_logit_prob: float = 0.95,
    batch_size: int = 512,
    max_feature_nodes: int | None = None,
    offload: Literal["cpu", "disk", None] = None,
    verbose: bool = False,
    update_interval: int = 4,
) -> Graph:
    """Compute an attribution graph for *prompt* using TransformerLens backend.

    Args:
        prompt: Text, token ids, or tensor - will be tokenized if str.
        model: Frozen ``TransformerLensReplacementModel``
        attribution_targets: Target specification in one of four formats:
                          - None: Auto-select salient logits based on probability threshold
                          - torch.Tensor: Tensor of token indices
                          - Sequence[str]: Token strings (tokenized, auto-computes probability
                            and unembed vector)
                          - Sequence[TargetSpec]: Fully specified custom targets (CustomTarget or tuple)
                            with arbitrary unembed directions
        max_n_logits: Max number of logit nodes (used when attribution_targets is None).
        desired_logit_prob: Keep logits until cumulative prob >= this value
                           (used when attribution_targets is None).
        batch_size: How many source nodes to process per backward pass. With the
            linear adjoint (used automatically when ``adjoint_supported(model)``; set
            ``model.use_linear_adjoint = False`` to force autograd) there is no
            backward pass: ``batch_size`` only sets the ranking queue
            (``update_interval * batch_size`` features per ranking round), and rows are
            computed in calls bounded by device memory, not by ``batch_size``.
        max_feature_nodes: Max number of feature nodes to include in the graph.
        offload: Method for offloading model parameters to save memory.
                 Options are "cpu" (move to CPU), "disk" (save to disk),
                 or None (no offloading).
        verbose: Whether to show progress information.
        update_interval: Number of batches to process before updating the feature ranking.

    Returns:
        Graph: Fully dense adjacency (unpruned).
    """

    logger = logging.getLogger("attribution")
    logger.propagate = False
    handler = None
    if verbose and not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.WARNING)

    offload_handles = []
    try:
        return _run_attribution(
            model=model,
            prompt=prompt,
            attribution_targets=attribution_targets,
            max_n_logits=max_n_logits,
            desired_logit_prob=desired_logit_prob,
            batch_size=batch_size,
            max_feature_nodes=max_feature_nodes,
            offload=offload,
            verbose=verbose,
            offload_handles=offload_handles,
            update_interval=update_interval,
            logger=logger,
        )
    finally:
        for reload_handle in offload_handles:
            reload_handle()

        if handler:
            logger.removeHandler(handler)


def _device_budget(device: torch.device) -> int:
    """Bytes one linear-adjoint call (its output rows) or one ranker update (its
    temporaries) may use: 1/32 of the device's total memory, clamped to [256 MiB, 2 GiB]
    (1 GiB off CUDA). It depends only on the device, not on its momentary free memory,
    so the chunking - and hence the graph - is reproducible."""
    if device.type == "cuda":
        total = torch.cuda.get_device_properties(device).total_memory
        return int(min(max(total // 32, 2**28), 2**31))
    return 2**30


def _ranker_limit(device: torch.device) -> int:
    """Bytes the ranker's ``(n_rows, n_rows)`` matrix may take: 1/8 of the device's total
    memory (2 GiB off CUDA), about 32k rows on a 32 GB GPU. Below that the ranker is
    several times faster than the host ranking, which also needs an
    ``(n_rows, n_nodes)`` matrix in host memory."""
    if device.type == "cuda":
        return torch.cuda.get_device_properties(device).total_memory // 8
    return 2**31


def _top_unvisited(influences: torch.Tensor, visited: torch.Tensor, k: int) -> torch.Tensor:
    """Indices of the ``k`` unvisited features of highest influence, in descending
    order, selected on the device. Exact ties (common in bf16/fp16, and among
    zero-influence features) go to the lower index, as with upstream's argsort of all
    influences followed by dropping the visited ones."""
    masked = influences.masked_fill(visited, float("-inf"))
    return torch.argsort(masked, descending=True, stable=True)[:k]


def _make_ranker(ctx, n_rows: int, n_cols: int, feat_layers: torch.Tensor, device):
    """Incremental on-device ranker for the linear-adjoint path.

    Returns None (rows go to the host edge matrix and the ranking uses upstream's
    ``compute_partial_influences``) when the adjoint is off, or when the ranker's
    ``(n_rows, n_rows)`` matrix would exceed ``_ranker_limit``, i.e. for very large
    ``max_feature_nodes``."""
    if ctx.adjoint is None or 4 * n_rows * n_rows > _ranker_limit(device):
        return None
    # number of active features at layers < l (feature columns are layer-ordered)
    widths = torch.searchsorted(
        feat_layers.contiguous(), torch.arange(ctx.n_layers + 1, device=feat_layers.device)
    ).tolist()
    widths[ctx.n_layers] = len(feat_layers)  # logit rows read every feature
    budget = _device_budget(device)
    # Device memory the ranker leaves for the attribution calls' temporaries: an output
    # of up to ``budget`` plus the adjoint's gradient chunk and its ~3 same-sized
    # temporaries.
    adjoint = ctx.adjoint
    headroom = budget + 4 * adjoint.max_rows_elems * adjoint.dtype.itemsize
    return BucketedRanker(
        n_rows, n_cols, len(feat_layers), widths, device, budget_bytes=budget, headroom=headroom
    )


def _dense_adjacency(
    edge_matrix,
    selected_features,
    row_to_node_index,
    max_feature_nodes,
    total_active_feats,
    total_nodes,
    n_logits,
):
    edge_device = edge_matrix.device
    if max_feature_nodes < total_active_feats:
        non_feature_nodes = torch.arange(total_active_feats, total_nodes)
        col_read = torch.cat([selected_features, non_feature_nodes])
        edge_matrix = edge_matrix[:, col_read.to(edge_device)]

    # sort rows such that features are in order
    edge_matrix = edge_matrix[row_to_node_index.argsort().to(edge_device)]
    final_node_count = edge_matrix.shape[1]
    full_edge_matrix = torch.zeros(final_node_count, final_node_count, device=edge_device)
    full_edge_matrix[:max_feature_nodes] = edge_matrix[:max_feature_nodes]
    full_edge_matrix[-n_logits:] = edge_matrix[max_feature_nodes:]
    return full_edge_matrix.cpu()


def _run_attribution(
    model,
    prompt,
    attribution_targets,
    max_n_logits,
    desired_logit_prob,
    batch_size,
    max_feature_nodes,
    offload,
    verbose,
    offload_handles,
    logger,
    update_interval=4,
):
    start_time = time.time()
    # Phase 0: precompute
    logger.info("Phase 0: Precomputing activations and vectors")
    phase_start = time.time()
    input_ids = model.ensure_tokenized(prompt)

    ctx = model.setup_attribution(input_ids)
    activation_matrix = ctx.activation_matrix

    logger.info(f"Precomputation completed in {time.time() - phase_start:.2f}s")
    logger.info(f"Found {ctx.activation_matrix._nnz()} active features")
    logger.info(
        "Attribution rows: "
        + ("linear adjoint" if ctx.adjoint is not None else "autograd (adjoint unsupported)")
    )

    if offload:
        handles = offload_modules(model.transcoders, offload)
        offload_handles += handles
        if offload == "disk" and isinstance(ctx._encoder_vecs, LazyEncoderVectors):
            # the selected features' W_enc rows are gathered on demand: read them from
            # the offload file while the transcoders are on disk
            ctx._encoder_vecs.read_from_offload_file(handles[0].path)

    # Phase 1: forward pass
    logger.info("Phase 1: Running forward pass")
    phase_start = time.time()
    if ctx.adjoint is None:
        with ctx.install_hooks(model):
            residual = model.forward(
                input_ids.expand(batch_size, -1), stop_at_layer=model.cfg.n_layers
            )
            ctx._resid_activations[-1] = model.ln_final(residual)
    # else: the frozen patterns/scales were already captured by setup_attribution and the
    # linear adjoint needs no replicated forward graph.
    logger.info(f"Forward pass completed in {time.time() - phase_start:.2f}s")

    if offload:
        offload_handles += offload_modules([block.mlp for block in model.blocks], offload)

    # Phase 2: build input vector list
    logger.info("Phase 2: Building input vectors")
    phase_start = time.time()
    feat_layers, feat_pos, _ = activation_matrix.indices()
    n_layers, n_pos, _ = activation_matrix.shape
    total_active_feats = activation_matrix._nnz()

    targets = AttributionTargets(
        attribution_targets=attribution_targets,
        logits=ctx.last_logits[0, -1],
        unembed_proj=model.unembed.W_U,
        tokenizer=model.tokenizer,
        max_n_logits=max_n_logits,
        desired_logit_prob=desired_logit_prob,
    )

    log_attribution_target_info(targets, attribution_targets, logger)

    if offload:
        offload_handles += offload_modules([model.unembed, model.embed], offload)

    logit_offset = len(feat_layers) + (n_layers + 1) * n_pos
    n_logits = len(targets)
    total_nodes = logit_offset + n_logits

    max_feature_nodes = min(max_feature_nodes or total_active_feats, total_active_feats)
    logger.info(f"Will include {max_feature_nodes} of {total_active_feats} feature nodes")

    # With the linear adjoint, rows are produced on the GPU; the ranker keeps them there
    # so influence ranking never re-uploads the growing edge matrix. When every active
    # feature is selected there is nothing to rank: rows go straight to the host matrix.
    device = ctx.error_vectors.device
    needs_ranking = max_feature_nodes < total_active_feats
    n_rows = max_feature_nodes + n_logits
    ranker = _make_ranker(ctx, n_rows, total_nodes, feat_layers, device) if needs_ranking else None
    # Rows per linear-adjoint call: whole queues when they fit the per-call budget,
    # otherwise budget-sized pieces, so no call's output (nor the ranker's update
    # temporaries, which scale with rows x n_rows) grows with the number of active features.
    row_bytes = 4 * (logit_offset + (n_rows if ranker is not None else 0))
    adjoint_rows = max(1, _device_budget(device) // row_bytes)
    edge_matrix = None
    if ranker is None:
        edge_matrix = torch.zeros(max_feature_nodes + n_logits, total_nodes)
    # Maps row indices in edge_matrix to original feature/node indices
    # First populated with logit node IDs, then feature IDs in attribution order
    row_to_node_index = torch.zeros(max_feature_nodes + n_logits, dtype=torch.int32)
    logger.info(f"Input vectors built in {time.time() - phase_start:.2f}s")

    # Phase 3: logit attribution
    logger.info("Phase 3: Computing logit attributions")
    phase_start = time.time()
    # The linear adjoint handles many rows per call (it chunks and layer-sorts
    # internally), so it gets up to ``adjoint_rows`` at once.
    phys_batch = max(min(len(targets), adjoint_rows), 1) if ctx.adjoint is not None else batch_size
    for i in range(0, len(targets), phys_batch):
        batch = targets.logit_vectors[i : i + phys_batch]
        rows = ctx.compute_batch(
            layers=torch.full((batch.shape[0],), n_layers),
            positions=torch.full((batch.shape[0],), n_pos - 1),
            inject_values=batch,
        )
        row_to_node_index[i : i + batch.shape[0]] = (
            torch.arange(i, i + batch.shape[0]) + logit_offset
        )
        if ranker is not None:
            ranker.add(
                i,
                rows,
                row_to_node_index[i : i + batch.shape[0]],
                torch.full((batch.shape[0],), n_layers),
            )
        else:
            assert edge_matrix is not None
            edge_matrix[i : i + batch.shape[0], :logit_offset] = rows.cpu()
    logger.info(f"Logit attributions completed in {time.time() - phase_start:.2f}s")

    # Phase 4: feature attribution
    logger.info("Phase 4: Computing feature attributions")
    phase_start = time.time()
    st = n_logits
    visited = torch.zeros(total_active_feats, dtype=torch.bool)
    visited_dev = visited.to(device) if ranker is not None else None
    n_visited = 0

    pbar = tqdm(total=max_feature_nodes, desc="Feature influence computation", disable=not verbose)

    while n_visited < max_feature_nodes:
        if max_feature_nodes == total_active_feats:
            pending = torch.arange(total_active_feats)
        else:
            queue_size = min(update_interval * batch_size, max_feature_nodes - n_visited)
            if ranker is not None:
                assert visited_dev is not None
                influences = ranker.influences(
                    st,
                    targets.logit_probabilities,
                    n_logits,
                    total_active_feats,
                    visited,
                    queue_size,
                )
                pending = _top_unvisited(
                    influences[:total_active_feats], visited_dev, queue_size
                ).cpu()
            else:
                assert edge_matrix is not None
                influences = compute_partial_influences(
                    edge_matrix[:st], targets.logit_probabilities, row_to_node_index[:st]
                )
                feature_rank = torch.argsort(influences[:total_active_feats], descending=True).cpu()
                pending = feature_rank[~visited[feature_rank]][:queue_size]

        # Rows within a queue are independent, so the linear adjoint processes the whole
        # queue (up to ``adjoint_rows``) in one call; the queue itself (and hence feature
        # selection) is unchanged.
        step = min(len(pending), adjoint_rows) if ctx.adjoint is not None else batch_size
        queue = [pending[i : i + step] for i in range(0, len(pending), step)]

        for idx_batch in queue:
            n_visited += len(idx_batch)

            rows = ctx.compute_batch(
                layers=feat_layers[idx_batch],
                positions=feat_pos[idx_batch],
                inject_values=ctx.encoder_rows(idx_batch),
                retain_graph=n_visited < max_feature_nodes,
            )

            end = st + rows.shape[0]
            row_to_node_index[st:end] = idx_batch
            if ranker is not None:
                ranker.add(st, rows, idx_batch, feat_layers[idx_batch])
            else:
                assert edge_matrix is not None
                edge_matrix[st:end, :logit_offset] = rows.cpu()
            visited[idx_batch] = True
            if visited_dev is not None:
                visited_dev[idx_batch.to(visited_dev.device)] = True
            st = end
            pbar.update(len(idx_batch))

    pbar.close()
    logger.info(f"Feature attributions completed in {time.time() - phase_start:.2f}s")
    if ranker is not None:
        logger.info(
            f"Ranker rows: {ranker.device_bytes / 2**30:.2f} GiB on the device, "
            f"{ranker.host_bytes / 2**30:.2f} GiB in host memory"
        )

    # Phase 5: packaging graph
    selected_features = torch.where(visited)[0]
    if ranker is not None:
        full_edge_matrix = ranker.adjacency(
            row_to_node_index, selected_features, n_logits, max_feature_nodes, total_nodes
        ).cpu()
        del ranker
    else:
        full_edge_matrix = _dense_adjacency(
            edge_matrix,
            selected_features,
            row_to_node_index,
            max_feature_nodes,
            total_active_feats,
            total_nodes,
            n_logits,
        )
        del edge_matrix, ranker

    graph = Graph(
        input_string=model.tokenizer.decode(input_ids),
        input_tokens=input_ids,
        logit_targets=targets.logit_targets,
        logit_probabilities=targets.logit_probabilities,
        vocab_size=targets.vocab_size,
        active_features=activation_matrix.indices().T,
        activation_values=activation_matrix.values(),
        selected_features=selected_features,
        adjacency_matrix=full_edge_matrix,
        cfg=model.cfg,
        scan_name=model.scan_name,
    )

    total_time = time.time() - start_time
    logger.info(f"Attribution completed in {total_time:.2f}s")

    return graph
