import atexit
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

import numpy as np
import torch
from torch import nn
from safetensors.torch import load_file, save_file

_offload_files = set()

_TEMP_PREFIX = "safetensors-offload-YqKRr8m3-"


@atexit.register
def cleanup_offload_files():
    for f in _offload_files:
        if os.path.exists(f):
            os.remove(f)


def cleanup_all_offload_files():
    temp_dir = tempfile.gettempdir()
    n_removed = 0
    for f in os.listdir(temp_dir):
        if f.startswith(_TEMP_PREFIX):
            os.remove(os.path.join(temp_dir, f))
            n_removed += 1
    return n_removed


class _HostStager:
    """Device-to-host copies pipelined through a few page-locked staging buffers.

    ``tensor.to("cpu")`` into fresh pageable memory runs at about 3 GiB/s: the driver
    stages the copy through its own pinned buffer and copies out with one thread, which
    also has to fault in every destination page. Here each tensor's bytes are DMA'd into
    one of ``n_buffers`` page-locked buffers while a few threads drain the previous
    buffer into the destination (about 13 GiB/s with PCIe 5 and 8 threads). The drain
    hands out small pieces to a thread pool rather than using one OpenMP region, so a
    thread that the OS deschedules on a busy machine delays only its piece. The
    destination is an ordinary (pageable) CPU tensor laid out exactly as
    ``tensor.to("cpu")`` would lay it out; the bytes are copied unchanged.

    Use as the ``fn`` of ``module._apply`` (it returns the destination at once and fills
    it as the pipeline advances), then call :meth:`finish`. The source tensors are kept
    alive until their bytes have been drained, so their device memory is released
    progressively and never reused while a copy is in flight. :meth:`close` releases the
    staging buffers. Tensors that are not on a CUDA device, not contiguous, or empty, and
    every tensor when the buffers cannot be page-locked, take ``tensor.to("cpu")``.
    """

    def __init__(
        self, n_buffers: int = 2, buffer_bytes: int = 64 * 2**20, piece_bytes: int = 4 * 2**20
    ):
        self.n_buffers = n_buffers
        self.buffer_bytes = buffer_bytes
        self.piece_bytes = piece_bytes
        self._pool: ThreadPoolExecutor | None = None
        self._bufs: list[torch.Tensor] | None = None  # allocated on the first CUDA tensor
        self._usable = torch.cuda.is_available()
        self._events: list = [None] * n_buffers
        self._segments: list[list] = [[] for _ in range(n_buffers)]  # (dst, stage, src)
        self._cur = 0
        self._fill = 0
        self._devices: set = set()

    def _allocate(self) -> bool:
        try:
            # from torch's caching host allocator: reused by later calls
            self._bufs = [
                torch.empty(self.buffer_bytes, dtype=torch.uint8, pin_memory=True)
                for _ in range(self.n_buffers)
            ]
        except RuntimeError:  # page-locked memory unavailable: plain copies
            self._usable = False
            return False
        self._pool = ThreadPoolExecutor(min(8, os.cpu_count() or 1))
        return True

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if (
            not self._usable
            or tensor.device.type != "cuda"
            or tensor.numel() == 0
            or not tensor.is_contiguous()
            or (self._bufs is None and not self._allocate())
        ):
            return tensor.to("cpu")
        dst = torch.empty_like(tensor, device="cpu")
        src_bytes = tensor.detach().reshape(-1).view(torch.uint8)
        dst_np = dst.reshape(-1).view(torch.uint8).numpy()
        done, total = 0, src_bytes.numel()
        while done < total:
            if self._fill == self.buffer_bytes:
                self._submit()
            n_bytes = min(total - done, self.buffer_bytes - self._fill)
            stage = self._bufs[self._cur][self._fill : self._fill + n_bytes]  # type: ignore[index]
            with torch.cuda.device(tensor.device):
                stage.copy_(src_bytes[done : done + n_bytes], non_blocking=True)
            self._devices.add(tensor.device)
            self._segments[self._cur].append(
                (dst_np[done : done + n_bytes], stage.numpy(), src_bytes)
            )
            self._fill += n_bytes
            done += n_bytes
        return dst

    def _submit(self):
        """Close the current buffer (its copies are in flight) and move to the next,
        draining that one first if it still holds copies."""
        if self._fill:
            devices = {seg[2].device for seg in self._segments[self._cur]}
            events = []
            for device in devices:
                event = torch.cuda.Event()
                event.record(torch.cuda.current_stream(device))
                events.append(event)
            self._events[self._cur] = events
        self._cur = (self._cur + 1) % self.n_buffers
        self._fill = 0
        self._drain(self._cur)

    def _drain(self, buf_idx: int):
        for event in self._events[buf_idx] or ():
            event.synchronize()
        self._events[buf_idx] = None
        pieces = [
            (dst[offset : offset + self.piece_bytes], stage[offset : offset + self.piece_bytes])
            for dst, stage, _ in self._segments[buf_idx]
            for offset in range(0, dst.shape[0], self.piece_bytes)
        ]
        # numpy releases the GIL while copying
        list(self._pool.map(lambda piece: np.copyto(piece[0], piece[1]), pieces))  # type: ignore[union-attr]
        self._segments[buf_idx] = []  # drops the references to the sources

    def finish(self):
        """Complete every pending copy."""
        if self._bufs is None:
            return
        self._submit()
        for i in range(1, self.n_buffers):
            self._drain((self._cur + i) % self.n_buffers)
        self._cur, self._fill = 0, 0

    def close(self):
        """Complete pending copies and release the staging buffers."""
        if self._bufs is None:
            return
        try:
            self.finish()
        finally:
            for device in self._devices:  # no copy may still be writing to the buffers
                torch.cuda.synchronize(device)
            self._bufs = None
            self._pool.shutdown()  # type: ignore[union-attr]
            self._pool = None


def disk_offload_module(module):
    org_device = next(module.parameters()).device
    with tempfile.NamedTemporaryFile(prefix=_TEMP_PREFIX, delete=False) as f:
        save_file(module.state_dict(), f.name)
        _offload_files.add(f.name)

    module.to(device="meta")

    def reload_handle(device=None):
        target_device = str(device or org_device)
        module.load_state_dict(load_file(f.name, device=target_device), assign=True)
        os.remove(f.name)
        _offload_files.remove(f.name)

    # the offloaded state dict (read by the linear adjoint)
    setattr(reload_handle, "path", f.name)
    return reload_handle


def cpu_offload_module(module, stager: _HostStager | None = None):
    """Move a module's parameters and buffers to host memory.

    Args:
        module (nn.Module): The module to offload.
        stager (_HostStager, optional): Shared staging buffers for the copies. If given,
            the caller must call ``stager.finish()`` before the module is used; if None,
            the copies complete before this returns.

    Returns:
        Callable: A handle that moves the module back to its original device.
    """
    org_device = next(module.parameters()).device
    if stager is None:
        stager = _HostStager()
        try:
            module._apply(stager)
        finally:
            stager.close()
    else:
        # the caller completes the copies (``stager.finish()``) before the module is used
        module._apply(stager)

    def reload_handle():
        module.to(device=org_device)

    return reload_handle


def offload_modules(
    modules: list | nn.Module | nn.ModuleList | nn.ModuleDict | nn.Sequential,
    offload_type: Literal["cpu", "disk"],
) -> list:
    """Offload one or more modules to CPU or disk.

    Args:
        modules: A single module, list of modules, or PyTorch module container
                 (ModuleList, ModuleDict, Sequential)
        offload_type: Type of offload - "cpu" or "disk"

    Returns:
        List of reload handles, one per module
    """
    if isinstance(modules, nn.ModuleDict):
        mods = modules.values()
    elif isinstance(modules, (list, nn.ModuleList, nn.Sequential)):
        mods = modules
    else:
        mods = [modules]
    if offload_type == "disk":
        return [disk_offload_module(module) for module in mods]
    stager = _HostStager()  # one set of staging buffers for all the modules
    handles = []
    try:
        for module in mods:
            handles.append(cpu_offload_module(module, stager))
        stager.close()
    except BaseException:
        stager.close()
        for reload_handle in handles:  # leave no module half-way
            reload_handle()
        raise
    return handles
