"""Metric implementations referenced by configs/eval_protocol.yaml."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def temporal_turbulence_proxy(velocity_in: torch.Tensor) -> torch.Tensor:
    """
    Per-point score from input velocity time series (T, N, 3).

    High values mean stronger temporal fluctuation across the input window — a
    **proxy** for more turbulent-like behaviour (not a CFD turbulence model).
    """
    vi = velocity_in.float()
    mean_t = vi.mean(dim=0, keepdim=True)
    fluct = (vi - mean_t).norm(dim=-1)
    return fluct.mean(dim=0)


def mse_velocity(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean squared error over all elements (B, T, N, 3)."""
    return F.mse_loss(pred, target)


def mse_velocity_train_weighted(
    pred: torch.Tensor,
    target: torch.Tensor,
    velocity_in: torch.Tensor,
    *,
    turb_alpha: float = 0.0,
    timestep_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Training MSE with optional per-output-timestep weights and optional turbulence
    proxy upweighting: per point ``w = 1 + turb_alpha * (proxy / mean(proxy))``.

    If ``turb_alpha == 0`` and ``timestep_weights is None``, equals ``mse_velocity``.
    """
    if turb_alpha == 0.0 and timestep_weights is None:
        return mse_velocity(pred, target)

    sq = (pred - target).pow(2)
    b, t, n, _c = sq.shape
    device, dtype = sq.device, sq.dtype

    tw = (
        timestep_weights.to(device=device, dtype=dtype).view(1, t, 1)
        if timestep_weights is not None
        else torch.ones(t, device=device, dtype=dtype).view(1, t, 1)
    )
    if timestep_weights is not None and timestep_weights.numel() != t:
        raise ValueError(
            f"loss_timestep_weights length {timestep_weights.numel()} != T_out {t}"
        )

    w_pts = torch.ones(b, n, device=device, dtype=dtype)
    if turb_alpha > 0.0:
        rows = []
        for bi in range(b):
            proxy = temporal_turbulence_proxy(velocity_in[bi])
            m = proxy.mean().clamp(min=1e-8)
            rows.append(1.0 + float(turb_alpha) * (proxy / m))
        w_pts = torch.stack(rows, dim=0)

    combined = tw * w_pts.unsqueeze(1)
    w_exp = combined.unsqueeze(-1)
    num = (sq * w_exp).sum()
    den = w_exp.expand_as(sq).sum().clamp(min=1e-8)
    return num / den


def l2_per_point_mean(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean L2 norm per velocity vector: mean over batch, time, points (like main.py hint)."""
    return (pred - target).norm(dim=-1).mean()


def l2_per_timestep_mean(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean L2 per output timestep, averaged over batch and points. ``(B,T,N,3)`` -> ``(T,)``."""
    return (pred - target).norm(dim=-1).mean(dim=(0, 2))


def mse_per_timestep_mean(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-timestep MSE, mean over batch, points, and vector components. ``(B,T,N,3)`` -> ``(T,)``."""
    return F.mse_loss(pred, target, reduction="none").mean(dim=(0, 2, 3))


def l2_per_timestep_mean_masked(
    pred: torch.Tensor, target: torch.Tensor, point_mask: torch.Tensor
) -> torch.Tensor:
    """
    Mean L2 per timestep over points where ``point_mask`` is True.
    ``pred``/``target`` ``(B,T,N,3)``, mask ``(B,N)``. Supports ``B==1`` (eval).
    """
    if pred.shape[0] != 1:
        raise NotImplementedError("l2_per_timestep_mean_masked supports batch_size=1")
    m = point_mask[0].float()
    d = (pred[0] - target[0]).norm(dim=-1)
    den = m.sum().clamp(min=1.0)
    return (d * m.unsqueeze(0)).sum(dim=1) / den


def mse_per_timestep_mean_masked(
    pred: torch.Tensor, target: torch.Tensor, point_mask: torch.Tensor
) -> torch.Tensor:
    """Per-timestep MSE over masked points; ``B==1`` only."""
    if pred.shape[0] != 1:
        raise NotImplementedError("mse_per_timestep_mean_masked supports batch_size=1")
    m = point_mask[0].float()
    err = (pred[0] - target[0]).pow(2).mean(dim=-1)
    den = m.sum().clamp(min=1.0)
    return (err * m.unsqueeze(0)).sum(dim=1) / den


def subsample_points(
    pred: torch.Tensor,
    target: torch.Tensor,
    n: int | None,
    generator: torch.Generator | None = None,
    *,
    return_indices: bool = False,
) -> (
    tuple[torch.Tensor, torch.Tensor]
    | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
):
    if n is None or n >= pred.shape[2]:
        if return_indices:
            idx = torch.arange(pred.shape[2], device=pred.device, dtype=torch.long)
            return pred, target, idx
        return pred, target
    num_p = pred.shape[2]
    if generator is None:
        idx = torch.randperm(num_p, device=pred.device)[:n]
    else:
        idx = torch.randperm(num_p, generator=generator)[:n].to(pred.device)
    if return_indices:
        return pred[:, :, idx, :], target[:, :, idx, :], idx
    return pred[:, :, idx, :], target[:, :, idx, :]


def mse_l2_lam_turb_on_subset(
    pred: torch.Tensor,
    target: torch.Tensor,
    lam_mask: torch.Tensor,
) -> tuple[
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    """
    Split MSE / mean L2 by lam_mask on point dimension (B, T, N, 3), mask (B, N).
    """
    mse_all = mse_velocity(pred, target)
    l2_all = l2_per_point_mean(pred, target)
    m_lams, m_turbs, l_lams, l_turbs = [], [], [], []
    for b in range(pred.shape[0]):
        m = lam_mask[b]
        if m.any():
            m_lams.append(F.mse_loss(pred[b, :, m], target[b, :, m]))
            l_lams.append((pred[b, :, m] - target[b, :, m]).norm(dim=-1).mean())
        inv = ~m
        if inv.any():
            m_turbs.append(F.mse_loss(pred[b, :, inv], target[b, :, inv]))
            l_turbs.append((pred[b, :, inv] - target[b, :, inv]).norm(dim=-1).mean())
    def _mean(xs: list) -> torch.Tensor | None:
        if not xs:
            return None
        return torch.stack(xs).mean()

    return (
        _mean(m_lams),
        _mean(m_turbs),
        mse_all,
        _mean(l_lams),
        _mean(l_turbs),
    )


def lam_turb_mask_for_eval_subset(
    vi_batch: torch.Tensor,
    idx: torch.Tensor,
) -> torch.Tensor:
    """
    vi_batch (B, T_in, N_full, 3), idx (n_sub,) — median split on proxy within subset.
    Returns (B, n_sub) bool (same mask broadcast to all batch items for B=1).
    """
    b, _, n_full, _ = vi_batch.shape
    if b != 1:
        raise NotImplementedError("lam_turb_mask_for_eval_subset supports batch_size=1")
    scores = temporal_turbulence_proxy(vi_batch[0])  # (N_full,)
    sub = scores[idx]
    med = sub.median()
    m = (sub <= med).unsqueeze(0).expand(b, -1)
    return m
