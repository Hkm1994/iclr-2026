"""Matplotlib slice plots and decimation for inspect CLI / Streamlit."""

from __future__ import annotations

import io
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import torch


Axis = Literal["x", "y", "z"]
_AXIS_I = {"x": 0, "y": 1, "z": 2}


def slice_tolerance(pos: torch.Tensor, axis: Axis) -> float:
    """Adaptive thickness from coordinate span."""
    i = _AXIS_I[axis]
    span = float(pos[:, i].max() - pos[:, i].min())
    return max(span / 200.0, 1e-6)


def slice_indices(
    pos: torch.Tensor, axis: Axis, slice_coord: float, tol: float | None = None
) -> torch.Tensor:
    """Boolean mask (N,) of points near the slice plane."""
    i = _AXIS_I[axis]
    if tol is None:
        tol = slice_tolerance(pos, axis)
    return (pos[:, i] - slice_coord).abs() <= tol


def project_slice_xy(
    pos: torch.Tensor, axis: Axis
) -> tuple[np.ndarray, np.ndarray]:
    """2D coordinates for scatter: the two axes orthogonal to ``axis``."""
    i = _AXIS_I[axis]
    others = [j for j in range(3) if j != i]
    u, v = others[0], others[1]
    return pos[:, u].detach().cpu().numpy(), pos[:, v].detach().cpu().numpy()


def speed_magnitude(vel: torch.Tensor) -> torch.Tensor:
    """(N, 3) -> (N,)"""
    return vel.norm(dim=-1)


def decimate_mask(n: int, max_points: int, seed: int) -> torch.Tensor:
    """Boolean mask length n with ~max_points True."""
    if n <= max_points:
        return torch.ones(n, dtype=torch.bool)
    g = torch.Generator()
    g.manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    m = torch.zeros(n, dtype=torch.bool)
    m[perm[:max_points]] = True
    return m


def plot_slice_row(
    pos: torch.Tensor,
    pred_v: torch.Tensor,
    actual_v: torch.Tensor,
    err_mag: torch.Tensor,
    idcs_airfoil: torch.Tensor,
    axis: Axis,
    slice_coord: float,
    *,
    k_label: str = "",
    share_vm: bool = True,
) -> plt.Figure:
    """
    One row: prediction speed |v|, actual speed, error magnitude on the same slice.
    ``pred_v``, ``actual_v`` shape (N, 3); ``err_mag`` (N,).
    """
    mask = slice_indices(pos, axis, slice_coord)
    if mask.sum() == 0:
        mask = torch.ones(pos.shape[0], dtype=torch.bool, device=pos.device)

    xs, ys = project_slice_xy(pos[mask], axis)
    pv = speed_magnitude(pred_v[mask]).detach().cpu().numpy()
    av = speed_magnitude(actual_v[mask]).detach().cpu().numpy()
    ev = err_mag[mask].detach().cpu().numpy()

    if share_vm:
        vm = max(float(pv.max()), float(av.max()), 1e-8)
        vmin_s, vmax_s = 0.0, vm
    else:
        vmin_s, vmax_s = None, None

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    titles = (f"Prediction |v| {k_label}", f"Actual |v| {k_label}", f"||pred−actual|| {k_label}")
    datas = (pv, av, ev)
    cmaps = ("viridis", "viridis", "turbo")
    for ax, data, title, cmap in zip(axes, datas, titles, cmaps):
        if title.startswith("||"):
            sc = ax.scatter(xs, ys, c=data, cmap=cmap, s=2, alpha=0.7)
        else:
            sc = ax.scatter(
                xs,
                ys,
                c=data,
                cmap=cmap,
                s=2,
                alpha=0.7,
                vmin=vmin_s,
                vmax=vmax_s,
            )
        plt.colorbar(sc, ax=ax, fraction=0.046)
        ax.set_title(title)
        ax.set_aspect("equal", adjustable="box")

    # Airfoil points on slice (project)
    if idcs_airfoil.numel() > 0:
        sp = pos[idcs_airfoil.long()]
        sm = slice_indices(sp, axis, slice_coord, tol=slice_tolerance(pos, axis) * 2)
        if sm.any():
            sx, sy = project_slice_xy(sp[sm], axis)
            for ax in axes:
                ax.scatter(sx, sy, s=8, c="none", edgecolors="k", linewidths=0.4, label="airfoil")
    fig.tight_layout()
    return fig


def plot_error_slice_only(
    pos: torch.Tensor,
    err_mag: torch.Tensor,
    idcs_airfoil: torch.Tensor,
    axis: Axis,
    slice_coord: float,
    *,
    p_lo: float = 5.0,
    p_hi: float = 95.0,
    k_label: str = "",
) -> plt.Figure:
    """Error magnitude with robust vmin/vmax from percentiles (highlight highs)."""
    mask = slice_indices(pos, axis, slice_coord)
    if mask.sum() == 0:
        mask = torch.ones(pos.shape[0], dtype=torch.bool, device=pos.device)
    xs, ys = project_slice_xy(pos[mask], axis)
    ev = err_mag[mask].detach().cpu().numpy()
    lo, hi = np.percentile(ev, [p_lo, p_hi])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    sc = ax.scatter(xs, ys, c=ev, cmap="turbo", s=2, alpha=0.75, vmin=lo, vmax=hi)
    plt.colorbar(sc, ax=ax, fraction=0.046, label="||pred−actual||")
    ax.set_title(f"Error (p{p_lo:.0f}–p{p_hi:.0f} scale) {k_label}")
    ax.set_aspect("equal", adjustable="box")
    if idcs_airfoil.numel() > 0:
        sp = pos[idcs_airfoil.long()]
        sm = slice_indices(sp, axis, slice_coord, tol=slice_tolerance(pos, axis) * 2)
        if sm.any():
            sx, sy = project_slice_xy(sp[sm], axis)
            ax.scatter(sx, sy, s=8, c="none", edgecolors="white", linewidths=0.5)
    fig.tight_layout()
    return fig


def figure_to_png_bytes(fig: plt.Figure, dpi: int = 120) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def frames_for_animation(
    pos: torch.Tensor,
    pred: torch.Tensor,
    actual: torch.Tensor,
    err_mag_bt: torch.Tensor,
    idcs_airfoil: torch.Tensor,
    axis: Axis,
    slice_coord: float,
    t_out: int,
    t_in: int,
    t_vec: torch.Tensor | None,
) -> list[bytes]:
    """List of PNG bytes for k=0..T_out-1."""
    frames: list[bytes] = []
    for k in range(t_out):
        label = f"k={k}"
        if t_vec is not None and t_vec.numel() >= t_in + k + 1:
            label += f" t={float(t_vec[t_in + k].cpu()):.5g}"
        fig = plot_slice_row(
            pos,
            pred[k],
            actual[k],
            err_mag_bt[k],
            idcs_airfoil,
            axis,
            slice_coord,
            k_label=label,
        )
        frames.append(figure_to_png_bytes(fig))
    return frames


def png_list_to_gif_bytes(png_frames: list[bytes], duration_ms: int = 300) -> bytes:
    from PIL import Image

    images = [Image.open(io.BytesIO(b)).convert("RGBA") for b in png_frames]
    if not images:
        return b""
    buf = io.BytesIO()
    images[0].save(
        buf,
        format="GIF",
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
    )
    buf.seek(0)
    return buf.getvalue()
