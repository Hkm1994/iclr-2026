"""Release cached allocator memory after epochs or when stopping training."""

from __future__ import annotations

import gc

import torch


def release_training_memory() -> None:
    """
    Best-effort cleanup so long runs do not retain peak allocations (CPU/GPU/MPS).

    Safe to call frequently; cheap when nothing is cached.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass
    try:
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass
