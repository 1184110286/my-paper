from __future__ import annotations

from typing import Any
import torch


def _is_cuda_device(device: torch.device | str | None) -> bool:
    try:
        d = torch.device(device) if device is not None else torch.device('cpu')
    except Exception:
        return False
    return d.type == 'cuda' and torch.cuda.is_available()


def reset_cuda_peak(device: torch.device | str | None) -> None:
    """Reset CUDA peak memory counters when CUDA is active.

    This is analysis-only instrumentation. It does not change model outputs or
    training behavior.  It is intentionally isolated here so model/training code
    does not need to know PyTorch memory API details.
    """
    if not _is_cuda_device(device):
        return
    d = torch.device(device)
    try:
        torch.cuda.reset_peak_memory_stats(d)
    except Exception:
        pass


def cuda_memory_stats(device: torch.device | str | None, *, synchronize: bool = True) -> dict[str, Any]:
    """Return CUDA memory statistics in MiB, or an empty dict on CPU.

    Keys are stable and can be added directly to history/summary rows.
    """
    if not _is_cuda_device(device):
        return {}
    d = torch.device(device)
    try:
        if synchronize:
            torch.cuda.synchronize(d)
    except Exception:
        pass
    to_mb = lambda b: float(b) / (1024.0 ** 2)
    out: dict[str, Any] = {}
    try:
        out['cuda_current_allocated_mb'] = to_mb(torch.cuda.memory_allocated(d))
    except Exception:
        pass
    try:
        out['cuda_current_reserved_mb'] = to_mb(torch.cuda.memory_reserved(d))
    except Exception:
        pass
    try:
        out['cuda_peak_allocated_mb'] = to_mb(torch.cuda.max_memory_allocated(d))
    except Exception:
        pass
    try:
        out['cuda_peak_reserved_mb'] = to_mb(torch.cuda.max_memory_reserved(d))
    except Exception:
        pass
    return out


def empty_cuda_cache(device: torch.device | str | None, *, synchronize: bool = False) -> bool:
    """Release unused cached CUDA memory blocks for visibility/stability.

    This does not change tensors, gradients, model parameters, or the MalSnif
    algorithm.  It only asks PyTorch's caching allocator to return currently
    unoccupied blocks to the driver so nvidia-smi/reserved memory does not keep
    growing across many heterogeneous graph windows.
    """
    if not _is_cuda_device(device):
        return False
    d = torch.device(device)
    try:
        if synchronize:
            torch.cuda.synchronize(d)
    except Exception:
        pass
    try:
        with torch.cuda.device(d):
            torch.cuda.empty_cache()
        return True
    except Exception:
        return False
