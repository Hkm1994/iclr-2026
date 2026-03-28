"""Shared point-cloud feature helpers."""

from __future__ import annotations

import torch
from torch import Tensor


def surface_mask_from_idcs(
    batch_size: int, num_points: int, idcs_airfoil: list[Tensor], device: torch.device
) -> Tensor:
    """(B, N) float mask, 1.0 at surface indices per batch item."""
    m = torch.zeros(batch_size, num_points, device=device)
    for b, idx in enumerate(idcs_airfoil):
        if idx.numel() == 0:
            continue
        m[b, idx.long()] = 1.0
    return m


def knn_indices_brute_force(pos: Tensor, k: int) -> Tensor:
    """Return (N, k) neighbor indices per point; pos (N, 3). Pure torch, O(N^2)."""
    n = pos.shape[0]
    k = min(k, n)
    d = torch.cdist(pos, pos)
    d.fill_diagonal_(float("inf"))
    _, idx = d.topk(k, dim=-1, largest=False)
    return idx
