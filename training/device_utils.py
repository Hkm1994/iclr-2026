"""Resolve ``train.device`` from YAML with fallbacks when CUDA/MPS are unavailable."""

from __future__ import annotations

import warnings

import torch


def resolve_train_device(train_cfg: dict) -> torch.device:
    """
    If ``device`` is unset, prefer CUDA then CPU.
    If set to CUDA or MPS but that backend is unavailable, warn and use CPU.
    """
    raw = train_cfg.get("device")
    if not raw:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(str(raw))
    if dev.type == "cuda" and not torch.cuda.is_available():
        warnings.warn(
            "train.device requests CUDA but PyTorch has no CUDA; using CPU.",
            UserWarning,
            stacklevel=2,
        )
        return torch.device("cpu")
    if dev.type == "mps":
        mps_ok = getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        if not mps_ok:
            warnings.warn(
                "train.device requests MPS but MPS is not available; using CPU.",
                UserWarning,
                stacklevel=2,
            )
            return torch.device("cpu")
    return dev
