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


def knn_indices_brute_force(
    pos: Tensor, k: int, *, row_chunk: int = 1024
) -> Tensor:
    """
    Return ``(N, k_eff)`` neighbor indices per point; pos ``(N, 3)``.

    Uses row blocks of ``row_chunk`` so peak memory is ``O(row_chunk * N)`` instead
    of ``O(N^2)``. Required for full-cloud validation (~100k points) on CPU.
    """
    n = pos.shape[0]
    device = pos.device
    if n <= 1:
        kk = min(max(int(k), 1), 1)
        return torch.zeros(n, kk, dtype=torch.long, device=device)

    k_eff = min(int(k), n - 1)
    k_eff = max(k_eff, 1)

    rc = max(1, int(row_chunk))
    out = torch.empty(n, k_eff, dtype=torch.long, device=device)

    for start in range(0, n, rc):
        end = min(start + rc, n)
        sub = pos[start:end]
        d = torch.cdist(sub, pos, p=2)
        for i in range(end - start):
            d[i, start + i] = float("inf")
        _, idx = d.topk(k_eff, dim=-1, largest=False)
        out[start:end] = idx
        del d

    return out
