"""
Incremental influence ranking for the attribution driver.

``compute_partial_influences`` re-uploads and re-normalizes the whole (growing) edge
matrix for every ranking update, which is O(n_rows^2 * n_nodes) traffic over a run.
``BucketedRanker`` keeps the edge rows on the device as they are produced and uses

* the **row-node trick**: the power iteration ``prod <- prod[rows] @ N`` only reads
  ``prod`` at the row nodes, so it runs on the small square ``row_node_matrix =
  N[:, rows]`` (``u <- u @ row_node_matrix``, maintained incrementally) and the
  influences are a single pass ``(sum_t u_t) @ N`` over the full matrix;
* **bucketed storage**: rows grouped by target layer and stored only up to their
  structural-zero boundary, exact fp32 on the device (or, when the device is full, in
  host memory: see ``BucketedRanker``).
"""

from __future__ import annotations

import contextlib
from typing import Any

import torch

# Host chunks are stored in pages of about this many bytes, allocated as rows arrive.
_PAGE_BYTES = 2**26


def _power_weights(
    row_node_matrix: torch.Tensor, logit_p: torch.Tensor, n_logits: int, max_iter: int = 128
):
    """sum_t u_t with u_0 = logit_p on the logit rows and u_{t+1} = u_t @ row_node_matrix."""
    step = torch.zeros(
        row_node_matrix.shape[0], device=row_node_matrix.device, dtype=row_node_matrix.dtype
    )
    step[:n_logits] = logit_p.to(device=row_node_matrix.device, dtype=row_node_matrix.dtype)
    total = torch.zeros_like(step)
    for _ in range(max_iter):
        if not step.any():
            break
        total += step
        step = step @ row_node_matrix
    else:
        raise RuntimeError("Failed to converge")
    return total


class _RankerBase:
    def __init__(self, n_rows: int, n_cols: int, device: torch.device):
        self.device = device
        self.n_cols = n_cols
        # 1 / (sum of each row's absolute edges)
        self.inv_row_sums = torch.zeros(n_rows, device=device)
        # normalized |edges| between the row nodes: row_node_matrix[i, j] = |E[i, nodes[j]]|
        self.row_node_matrix = torch.zeros(n_rows, n_rows, device=device)
        self.nodes = torch.zeros(n_rows, dtype=torch.long, device=device)

    def _update_row_node_matrix(
        self, start: int, abs_rows: torch.Tensor, old_cols_at_new: torch.Tensor | None
    ):
        """abs_rows: |new rows| (fp32, on device). old_cols_at_new: E[:start, new_nodes]."""
        end = start + abs_rows.shape[0]
        inv_new = 1.0 / abs_rows.sum(1).clamp(min=1e-8)
        self.inv_row_sums[start:end] = inv_new
        # new rows against every row node so far (including themselves)
        idx = self.nodes[:end]
        # logit nodes have no column in the rows (they are never sources)
        vals = abs_rows[:, idx.clamp(max=abs_rows.shape[1] - 1)]
        vals[:, idx >= abs_rows.shape[1]] = 0
        self.row_node_matrix[start:end, :end] = vals.mul_(inv_new[:, None])
        if start:
            assert old_cols_at_new is not None
            self.row_node_matrix[:start, start:end] = old_cols_at_new.abs_().mul_(
                self.inv_row_sums[:start, None]
            )

    def weights(self, n_rows_done: int, logit_p: torch.Tensor, n_logits: int) -> torch.Tensor:
        """Per-row coefficients c such that influences = c @ |E[:n_rows_done]|."""
        total = _power_weights(self.row_node_matrix[:n_rows_done, :n_rows_done], logit_p, n_logits)
        return total * self.inv_row_sums[:n_rows_done]


class _HostChunk:
    """A chunk whose rows are (partly) in host memory: an optional device ``head`` (the
    filled rows of a device chunk that could not grow) followed by rows in fixed-size
    host pages, allocated as rows arrive (so only stored rows are resident)."""

    def __init__(self, width: int, pin: bool, head: torch.Tensor | None = None):
        self.width = width
        self.head = head
        self.page_rows = max(1, _PAGE_BYTES // (4 * max(width, 1)))
        self.pin = pin
        self.pages: list[torch.Tensor] = []
        self.n_rows = 0  # rows in the pages

    @property
    def nbytes(self) -> int:
        """Host bytes."""
        return sum(4 * page.numel() for page in self.pages)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.n_rows + (0 if self.head is None else self.head.shape[0]), self.width)

    def append(self, rows: torch.Tensor):
        n_done = 0
        while n_done < rows.shape[0]:
            if self.n_rows == len(self.pages) * self.page_rows:
                self.pages.append(torch.empty(self.page_rows, self.width, pin_memory=self.pin))
            page, offset = divmod(self.n_rows, self.page_rows)
            take = min(self.page_rows - offset, rows.shape[0] - n_done)
            self.pages[page][offset : offset + take].copy_(rows[n_done : n_done + take])
            n_done += take
            self.n_rows += take

    def _filled(self):
        for i, page in enumerate(self.pages):
            yield page[: min(self.page_rows, self.n_rows - i * self.page_rows)]

    def to(self, device: torch.device) -> torch.Tensor:
        """The rows as one contiguous (rows, width) tensor on ``device``."""
        out = torch.empty(self.shape, device=device)
        offset = 0
        if self.head is not None:
            out[: self.head.shape[0]].copy_(self.head)
            offset = self.head.shape[0]
        for page in self._filled():
            out[offset : offset + page.shape[0]].copy_(page, non_blocking=self.pin)
            offset += page.shape[0]
        return out

    def cols(self, cols: torch.Tensor) -> torch.Tensor:
        """``rows[:, cols]``, on the host (``cols`` on the host)."""
        parts = [page[:, cols] for page in self._filled()]
        if self.head is not None:
            parts.insert(0, self.head[:, cols.to(self.head.device)].cpu())
        return torch.cat(parts)


class BucketedRanker(_RankerBase):
    """Exact fp32 edge rows, stored without their structural zeros.

    A row whose target sits at layer ``l`` can only have non-zero edges from features
    at layers ``< l`` (and logit rows from all features). Feature columns are ordered
    by layer, so the feature part of such a row is a *prefix* of width
    ``prefix_width[l]``. Rows are kept in one growable bucket per target layer with
    exactly that width; the error/token columns are kept densely. For typical graphs
    (most selected features sit in early layers) this is a fraction of the dense
    matrix, and the influence pass shrinks by the same factor.

    **Host chunks.** Row chunks are kept on the device while it has room: ``headroom``
    bytes stay available for the attribution calls' temporaries (and at most
    ``device_limit`` bytes of rows go on the device, if given). Further chunks are kept
    in host memory owned by the ranker and freed with it (page-locked if ``pin_host``);
    the influence pass copies each one to the device before the same product. Chunk
    boundaries and every product are the same wherever a chunk lives, so the
    influences, the selected features and the adjacency are bit for bit those of the
    all-device ranker: placement only changes speed, never the result. The
    error/token part (written once per row, read only for the final adjacency) is kept
    on the host when it is larger than half the per-call budget.

    Args:
        n_rows (int): Maximum number of rows (logit rows plus feature rows).
        n_cols (int): Number of source nodes per row.
        n_feats (int): Number of active features (the leading, layer-ordered columns).
        prefix_width (list[int]): Feature-prefix width of a row per target layer
            (``n_layers + 1`` entries; the last is for logit rows).
        device (torch.device): Device the rows and the ranking live on.
        chunk_rows (int): Rows per stored chunk (fewer for very wide rows).
        budget_bytes (int): Size bound for temporaries of the ranker's own operations.
        headroom (int, optional): Device bytes to leave free for the attribution
            calls. Defaults to ``4 * budget_bytes``.
        device_limit (int, optional): Max bytes of rows stored on the device.
        pin_host (bool): Page-lock the rows kept in host memory.
    """

    def __init__(
        self,
        n_rows: int,
        n_cols: int,
        n_feats: int,
        prefix_width: list[int],
        device: torch.device,
        chunk_rows: int = 256,
        budget_bytes: int = 2**30,
        headroom: int | None = None,
        device_limit: int | None = None,
        pin_host: bool = False,
    ):
        device = torch.device(device)
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        super().__init__(n_rows, n_cols, device)
        self.n_feats = n_feats
        self.budget_bytes = budget_bytes
        self.prefix_width = prefix_width
        self.chunk_rows = chunk_rows
        self.headroom = 4 * budget_bytes if headroom is None else headroom
        self.device_limit = device_limit
        self.pin_host = pin_host and device.type == "cuda"
        self.device_bytes = 0  # rows stored on the device
        # Row chunks are long-lived and of many sizes: a private pool keeps them out of
        # the allocator segments the per-call temporaries are cached in (which they
        # would fragment). The pool and its memory are released with the ranker.
        self._pool = None
        if device.type == "cuda" and hasattr(torch.cuda, "use_mem_pool"):
            self._pool = torch.cuda.MemPool()
        dense_shape = (n_rows, n_cols - n_feats)
        dense_bytes = 4 * dense_shape[0] * dense_shape[1]
        dense = None
        if dense_bytes <= budget_bytes // 2 and self._device_room(dense_bytes):
            dense = self._device_empty(dense_shape)
        self.dense_on_host = dense is None
        if dense is None:
            dense = torch.empty(dense_shape, pin_memory=self.pin_host)
        dense.zero_()
        self.dense: torch.Tensor | None = dense
        self.buckets: dict[int, dict[str, Any]] = {}

    def release(self):
        """Free the stored rows (before the pool they live in)."""
        self.buckets = {}
        self.dense = None
        self._pool = None

    def __del__(self):
        self.release()

    @property
    def host_bytes(self) -> int:
        """Bytes of rows kept in host memory."""
        total = 4 * self.dense.numel() if self.dense_on_host and self.dense is not None else 0
        for bucket in self.buckets.values():
            total += sum(c.nbytes for c in bucket["chunks"] if isinstance(c, _HostChunk))
        return total

    # -------------------------------------------------------------- placement
    def _device_room(self, nbytes: int) -> bool:
        """Whether ``nbytes`` more rows may be stored on the device, leaving ``headroom``
        bytes the attribution calls' temporaries can get: free device memory plus
        cached allocator segments that are wholly unused (the allocator releases those
        when it needs room). Free blocks inside partly used segments are not counted
        (an allocation may not fit in them), nor is memory cached in the rows' pool,
        which only rows can reuse."""
        if self.device_limit is not None and self.device_bytes + nbytes > self.device_limit:
            return False
        if self.device.type != "cuda":
            return True
        free, _ = torch.cuda.mem_get_info(self.device)
        if free - nbytes >= self.headroom:  # the common case: no need to look closer
            return True
        stats = torch.cuda.memory_stats(self.device)
        unused = (
            stats.get("reserved_bytes.all.current", 0)
            - stats.get("allocated_bytes.all.current", 0)
            - stats.get("inactive_split_bytes.all.current", 0)
        )
        largest_pool_block = 0
        if self._pool is not None:
            for seg in self._pool.snapshot(include_traces=False):
                if seg["allocated_size"] == 0:
                    unused -= seg["total_size"]
                for blk in seg["blocks"]:
                    if blk["state"] == "inactive":
                        largest_pool_block = max(largest_pool_block, blk["size"])
        new = 0 if nbytes <= largest_pool_block else nbytes  # else the pool reuses a block
        return free + max(unused, 0) - new >= self.headroom

    def _device_empty(self, shape) -> torch.Tensor | None:
        pool = self._pool
        ctx = torch.cuda.use_mem_pool(pool, self.device) if pool else contextlib.nullcontext()
        try:
            with ctx:
                storage = torch.empty(shape, device=self.device)
        except torch.OutOfMemoryError:  # e.g. fragmentation: the host takes the rows
            return None
        self.device_bytes += 4 * storage.numel()
        return storage

    def _new_chunk(self, rows: int, width: int) -> torch.Tensor | _HostChunk:
        """Storage for ``rows`` rows of a chunk: on the device if it has room."""
        storage = self._device_empty((rows, width)) if self._device_room(4 * rows * width) else None
        return _HostChunk(width, self.pin_host) if storage is None else storage

    # ------------------------------------------------------------------ rows
    def _append(self, layer: int, rows: torch.Tensor, sel: torch.Tensor, ids: torch.Tensor):
        """Append ``rows[sel]`` to the layer's bucket (row ids ``ids``).

        A bucket is a list of row chunks of at most ``chunk_rows`` rows (the same blocks
        the influence pass iterates over). Only the last chunk grows, by 1.25x, so the
        slack is a fraction of one chunk and growth never copies the whole bucket."""
        width = self.prefix_width[layer]
        # fewer rows per chunk for very wide rows: one chunk (and the influence pass's
        # |chunk| temporary) stays <= 1 GiB
        chunk_rows = max(16, min(self.chunk_rows, 2**30 // max(4 * width, 1)))
        bucket = self.buckets.get(layer)
        n_new = sel.shape[0]
        if bucket is None:
            bucket = self.buckets[layer] = dict(
                chunk_rows=chunk_rows,
                chunks=[],
                ids=torch.empty(max(64, n_new), dtype=torch.long, device=self.device),
                n_rows=0,
            )
        n_rows = bucket["n_rows"]
        if n_rows + n_new > bucket["ids"].shape[0]:
            ids_new = torch.empty(
                max(2 * bucket["ids"].shape[0], n_rows + n_new),
                dtype=torch.long,
                device=self.device,
            )
            ids_new[:n_rows] = bucket["ids"][:n_rows]
            bucket["ids"] = ids_new
        bucket["ids"][n_rows : n_rows + n_new] = ids
        chunks = bucket["chunks"]
        src = rows[:, :width]
        n_done = 0
        while n_done < n_new:
            used = n_rows - (len(chunks) - 1) * chunk_rows if chunks else chunk_rows
            if used == chunk_rows:  # start a new chunk
                chunks.append(self._new_chunk(min(chunk_rows, n_new - n_done), width))
                used = 0
            take = min(chunk_rows - used, n_new - n_done)
            last = chunks[-1]
            if not isinstance(last, _HostChunk) and used + take > last.shape[0]:
                capacity = min(chunk_rows, max(used + take, -(-last.shape[0] * 5 // 4)))
                grown = self._new_chunk(capacity, width)
                if isinstance(grown, _HostChunk):  # its rows so far stay on the device
                    grown.head = last[:used]
                else:
                    grown[:used] = last[:used]
                    self.device_bytes -= 4 * last.numel()
                chunks[-1] = last = grown
            take_idx = sel[n_done : n_done + take]
            if isinstance(last, _HostChunk):  # gathered on the device in bounded pieces
                step = max(1, self.budget_bytes // (16 * max(width, 1)))
                for j in range(0, take, step):
                    last.append(torch.index_select(src, 0, take_idx[j : j + step]))
            else:
                torch.index_select(src, 0, take_idx, out=last[used : used + take])
            n_done += take
            n_rows += take
        bucket["n_rows"] = n_rows

    def _chunks(self, bucket):
        """(offset, chunk) of each chunk of ``bucket``; device chunks are sliced to their
        filled rows."""
        for i, chunk in enumerate(bucket["chunks"]):
            offset = i * bucket["chunk_rows"]
            if isinstance(chunk, _HostChunk):
                yield offset, chunk
            else:
                yield offset, chunk[: min(bucket["chunk_rows"], bucket["n_rows"] - offset)]

    def add(self, start: int, rows: torch.Tensor, nodes: torch.Tensor, layers: torch.Tensor):
        """Store new edge rows.

        Args:
            start (int): Row id of the first new row (the number of rows stored so far).
            rows (torch.Tensor): (n_new, n_cols) edge rows. Consumed: it is overwritten
                with its absolute values.
            nodes (torch.Tensor): (n_new,) node index of each row's target.
            layers (torch.Tensor): (n_new,) layer of each row's target (``n_layers`` for
                logit rows).
        """
        rows = rows.float()  # fp16 can't hold the 1e-8 normalization floor (zero rows -> NaN)
        n_new = rows.shape[0]
        end = start + n_new
        nodes_dev = nodes.to(self.device, torch.long)
        self.nodes[start:end] = nodes_dev
        assert self.dense is not None
        self.dense[start:end, : rows.shape[1] - self.n_feats] = rows[:, self.n_feats :]

        old_cols_at_new = None
        if start:  # existing rows' edges into the new nodes (features, < n_feats)
            old_cols_at_new = torch.zeros(start, n_new, device=self.device)
            for layer, bucket in self.buckets.items():
                width = self.prefix_width[layer]
                valid = torch.nonzero(nodes_dev < width).squeeze(1)
                if valid.numel():
                    cols = nodes_dev[valid]
                    cols_host = None
                    for offset, chunk in self._chunks(bucket):
                        if isinstance(chunk, _HostChunk):
                            if cols_host is None:
                                cols_host = cols.cpu()
                            vals = chunk.cols(cols_host).to(self.device)
                        else:
                            vals = chunk[:, cols]
                        chunk_ids = bucket["ids"][offset : offset + vals.shape[0]]
                        old_cols_at_new[chunk_ids[:, None], valid[None, :]] = vals

        ids = torch.arange(start, end, device=self.device)
        layers = layers.cpu()
        for layer in layers.unique().tolist():
            sel = torch.nonzero(layers == layer).squeeze(1).to(self.device)
            self._append(int(layer), rows, sel, ids[sel])
        self._update_row_node_matrix(start, rows.abs_(), old_cols_at_new)

    def influences(self, n_rows_done, logit_p, n_logits, n_feats, visited, queue_size):
        """Influence of every feature on the logits, from the rows stored so far.

        Args:
            n_rows_done (int): Number of rows stored so far.
            logit_p (torch.Tensor): Probabilities of the logit targets.
            n_logits (int): Number of logit rows (the first rows).
            n_feats (int): Number of active features.
            visited (torch.Tensor): Unused (the caller masks visited features).
            queue_size (int): Unused.

        Returns:
            torch.Tensor: (n_feats,) influences.
        """
        coeffs = self.weights(n_rows_done, logit_p, n_logits)
        influence = torch.zeros(n_feats, device=self.device)
        for layer, bucket in self.buckets.items():
            n_rows, width = bucket["n_rows"], self.prefix_width[layer]
            if width == 0:
                continue
            bucket_coeffs = coeffs[bucket["ids"][:n_rows]]
            for offset, chunk in self._chunks(bucket):
                if isinstance(chunk, _HostChunk):
                    abs_chunk = chunk.to(self.device).abs_()
                else:
                    abs_chunk = chunk.abs()
                influence[:width] += bucket_coeffs[offset : offset + abs_chunk.shape[0]] @ abs_chunk
                del abs_chunk
        return influence

    def adjacency(
        self, row_to_node_index, selected_features, n_logits, max_feature_nodes, total_nodes
    ) -> torch.Tensor:
        """The graph's (node x node) adjacency, on the host; equals the dense path's.

        Only the feature and logit rows can be non-zero, so just those rows are
        assembled (on the device when ``n_rows x n_final`` fits the budget, else on the
        host block by block). This consumes the ranker: each chunk is freed once
        copied, and the rows that stay zero (error/token nodes, most of the matrix) are
        zeroed last, so the stored rows and the full host matrix are not resident at
        the same time.

        Args:
            row_to_node_index (torch.Tensor): Node index of each stored row's target.
            selected_features (torch.Tensor): Sorted indices of the selected features.
            n_logits (int): Number of logit nodes.
            max_feature_nodes (int): Unused (the selection fixes the feature count).
            total_nodes (int): Number of nodes before feature selection.

        Returns:
            torch.Tensor: (n_final, n_final) adjacency over the selected features, the
                error and token nodes and the logits.
        """
        assert self.dense is not None
        sel = selected_features.to(self.device, torch.long)
        sel_host = selected_features.to("cpu", torch.long)
        n_sel = sel.numel()
        n_other = total_nodes - self.n_feats
        n_final = n_sel + n_other

        nodes = row_to_node_index.to(self.device, torch.long)
        n_rows = nodes.numel()
        colpos = torch.full((self.n_feats,), -1, dtype=torch.long, device=self.device)
        colpos[sel] = torch.arange(n_sel, device=self.device)
        is_feat = nodes < self.n_feats
        # compact row index: selected features first (in column order), then the logits
        row_pos = torch.where(
            is_feat,
            colpos[nodes.clamp(max=self.n_feats - 1)],
            n_sel + (nodes - (total_nodes - n_logits)),
        )

        def feature_blocks():
            """(row ids, edges into the selected features of the prefix); each chunk is
            freed once its block is taken."""
            for layer, bucket in self.buckets.items():
                n_bucket_rows, width = bucket["n_rows"], self.prefix_width[layer]
                n_sel_prefix = int(torch.searchsorted(sel, width))  # selected in the prefix
                if n_bucket_rows == 0 or n_sel_prefix == 0:
                    continue
                for i, (offset, chunk) in enumerate(self._chunks(bucket)):
                    if isinstance(chunk, _HostChunk):
                        vals = chunk.cols(sel_host[:n_sel_prefix])
                    else:
                        vals = chunk[:, sel[:n_sel_prefix]]
                    bucket["chunks"][i] = chunk = None
                    yield bucket["ids"][offset : offset + vals.shape[0]], vals

        if 4 * n_rows * n_final <= self.budget_bytes:
            rows = torch.zeros(n_rows, n_final, device=self.device)
            rows[row_pos, n_sel:] = self.dense[:n_rows].to(self.device)
            for ids, vals in feature_blocks():
                cols = torch.arange(vals.shape[1], device=self.device)[None]
                rows[row_pos[ids][:, None], cols] = vals.to(self.device)
            self.release()
            full = torch.empty(n_final, n_final)
            full[:n_sel].copy_(rows[:n_sel])
            full[n_final - n_logits :].copy_(rows[n_sel:])
        else:
            host_row = torch.where(
                row_pos < n_sel, row_pos, row_pos + (n_final - n_logits - n_sel)
            ).cpu()
            full = torch.empty(n_final, n_final)
            full[:n_sel].zero_()
            full[n_final - n_logits :].zero_()
            step = max(1, self.budget_bytes // (4 * max(n_other, 1)))
            for start in range(0, n_rows, step):
                end = min(n_rows, start + step)
                full[host_row[start:end], n_sel:] = self.dense[start:end].cpu()
            self.dense = None
            for ids, vals in feature_blocks():
                cols = torch.arange(vals.shape[1])[None]
                full[host_row[ids.cpu()][:, None], cols] = vals.cpu()
            self.release()
        full[n_sel : n_final - n_logits].zero_()
        return full
