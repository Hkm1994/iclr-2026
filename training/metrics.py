"""Metric implementations referenced by configs/eval_protocol.yaml."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def mse_velocity(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean squared error over all elements (B, T, N, 3)."""
    return F.mse_loss(pred, target)


def l2_per_point_mean(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean L2 norm per velocity vector: mean over batch, time, points (like main.py hint)."""
    return (pred - target).norm(dim=-1).mean()


def subsample_points(
    pred: torch.Tensor,
    target: torch.Tensor,
    n: int | None,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if n is None or n >= pred.shape[2]:
        return pred, target
    num_p = pred.shape[2]
    if generator is None:
        idx = torch.randperm(num_p, device=pred.device)[:n]
    else:
        idx = torch.randperm(num_p, generator=generator)[:n].to(pred.device)
    return pred[:, :, idx, :], target[:, :, idx, :]
