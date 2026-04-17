"""Pure-tensor diagnostics: per-point velocity error, surface vs bulk stats."""

from __future__ import annotations

from typing import Any

import torch


def per_point_error_magnitude(
    pred: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """L2 norm of velocity error per point. Shapes ``(B, T, N, 3)`` -> ``(B, T, N)``."""
    return (pred - target).norm(dim=-1)


def bulk_mask(
    n_points: int,
    surface_idcs: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Boolean mask shape ``(N,)`` — True for points not on the airfoil surface.

    ``surface_idcs`` are indices into ``0..N-1``.
    """
    m = torch.ones(n_points, dtype=torch.bool, device=device)
    if surface_idcs.numel() > 0:
        m[surface_idcs.long()] = False
    return m


def error_percentiles(
    err_mag: torch.Tensor, quantiles: tuple[float, ...] = (0.5, 0.9, 0.99)
) -> dict[str, float]:
    """``err_mag`` any shape; flatten and return named quantiles (CPU floats)."""
    flat = err_mag.detach().float().reshape(-1)
    out: dict[str, float] = {
        "min": float(flat.min().cpu()),
        "max": float(flat.max().cpu()),
        "mean": float(flat.mean().cpu()),
    }
    for q in quantiles:
        key = f"p{int(round(100 * q))}"
        out[key] = float(torch.quantile(flat, q).cpu())
    return out


def worst_point_indices(
    err_mag: torch.Tensor, n: int = 20
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    ``err_mag`` shape ``(N,)``. Returns (indices, values) for top-``n`` errors, descending.
    """
    n = min(n, int(err_mag.numel()))
    if n <= 0:
        return (
            torch.tensor([], dtype=torch.long, device=err_mag.device),
            torch.tensor([], dtype=err_mag.dtype, device=err_mag.device),
        )
    vals, idx = torch.topk(err_mag.detach(), k=n, largest=True)
    return idx, vals


def _masked_mean_speed(vel: torch.Tensor, mask: torch.Tensor) -> float:
    """vel (N, 3), mask (N,) bool — mean |v| over True entries."""
    if not mask.any():
        return float("nan")
    s = vel[mask].norm(dim=-1).mean()
    return float(s.cpu())


def _masked_mean_error(err_mag: torch.Tensor, mask: torch.Tensor) -> float:
    if not mask.any():
        return float("nan")
    return float(err_mag[mask].mean().cpu())


def surface_bulk_summary(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    bi: int,
    k: int,
    idcs_airfoil: list[torch.Tensor],
) -> dict[str, Any]:
    """
    One batch index ``bi`` and output timestep ``k``.
    ``pred``/``target`` are ``(B, T, N, 3)``.
    """
    p = pred[bi, k]
    t = target[bi, k]
    err = (p - t).norm(dim=-1)
    idcs = idcs_airfoil[bi]
    n = p.shape[0]
    surf = torch.zeros(n, dtype=torch.bool, device=p.device)
    if idcs.numel() > 0:
        surf[idcs.long()] = True
    bulk = ~surf

    return {
        "surface_mean_speed_pred": _masked_mean_speed(p, surf),
        "surface_mean_speed_actual": _masked_mean_speed(t, surf),
        "surface_mean_error_l2": _masked_mean_error(err, surf),
        "surface_max_error_l2": float(err[surf].max().cpu()) if surf.any() else float("nan"),
        "bulk_mean_speed_pred": _masked_mean_speed(p, bulk),
        "bulk_mean_speed_actual": _masked_mean_speed(t, bulk),
        "bulk_mean_error_l2": _masked_mean_error(err, bulk),
        "n_surface_points": int(surf.sum().cpu()),
        "n_bulk_points": int(bulk.sum().cpu()),
    }


def timestep_l2_table(
    pred: torch.Tensor, target: torch.Tensor
) -> tuple[list[float], list[float]]:
    """Per-output-timestep mean L2 and MSE (scalar per T). ``(B,T,N,3)``."""
    err = (pred - target).norm(dim=-1)
    l2_per_t = err.mean(dim=(0, 2)).detach().cpu().tolist()
    mse_per_t = (pred - target).pow(2).mean(dim=(0, 2, 3)).detach().cpu().tolist()
    return [float(x) for x in l2_per_t], [float(x) for x in mse_per_t]


def global_l2_mse(pred: torch.Tensor, target: torch.Tensor) -> tuple[float, float]:
    """Mean L2 per vector and scalar MSE over all elements."""
    l2 = (pred - target).norm(dim=-1).mean()
    mse = (pred - target).pow(2).mean()
    return float(l2.cpu()), float(mse.cpu())
