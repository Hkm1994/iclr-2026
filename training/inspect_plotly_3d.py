"""Plotly 3D scatter helpers for prediction inspection (decimated point clouds)."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch

from training.inspect_viz import decimate_mask, speed_magnitude


def _color_values(
    pred_v: torch.Tensor,
    actual_v: torch.Tensor,
    err_mag: torch.Tensor,
    mode: str,
    *,
    log_scale: bool,
    eps: float = 1e-6,
) -> tuple[np.ndarray, str, str, str]:
    """Returns (c_numpy, trace_name, colorbar_title, colorscale_name)."""
    spd_p = speed_magnitude(pred_v)
    spd_a = speed_magnitude(actual_v)
    if mode == "abs_error":
        c = err_mag
        name = "Absolute Δv"
        cbar = "‖pred − actual‖₂"
        colorscale = "Turbo"
    elif mode == "rel_error":
        c = err_mag / (spd_a + eps)
        name = "Relative error"
        cbar = "‖Δv‖₂ / (|vₐ| + ε)"
        colorscale = "Inferno"
    elif mode == "speed_pred":
        c = spd_p
        name = "|v| prediction"
        cbar = "|v| (pred)"
        colorscale = "Viridis"
    elif mode == "speed_actual":
        c = spd_a
        name = "|v| actual"
        cbar = "|v| (actual)"
        colorscale = "Plasma"
    elif mode == "speed_mag_error":
        c = (spd_p - spd_a).abs()
        name = "Speed magnitude error"
        cbar = "||v|ₚ − |v|ₐ|"
        colorscale = "Turbo"
    else:
        raise ValueError(mode)

    c_np = c.detach().float().cpu().numpy()
    if log_scale:
        c_np = np.log10(c_np + eps)
        cbar = f"log₁₀({cbar})" if "log" not in cbar else cbar

    return c_np, name, cbar, colorscale


def _cap_color_range(
    c: np.ndarray, mask: np.ndarray, percentile_hi: float | None
) -> tuple[float | None, float | None]:
    """Return (cmin, cmax) for color scale from bulk subset; None = autoscale."""
    if percentile_hi is None:
        return None, None
    vals = c[mask]
    if vals.size == 0:
        return None, None
    lo = float(np.percentile(vals, 100 - percentile_hi))
    hi = float(np.percentile(vals, percentile_hi))
    if lo >= hi:
        hi = lo + 1e-8
    return lo, hi


def figure_3d_scatter(
    pos: torch.Tensor,
    pred_v: torch.Tensor,
    actual_v: torch.Tensor,
    err_mag: torch.Tensor,
    idcs_airfoil: torch.Tensor,
    *,
    max_points: int,
    decimate_seed: int,
    color_mode: str,
    log_scale: bool = False,
    color_cap_percentile: float | None = 99.0,
    template: str = "plotly_dark",
) -> go.Figure:
    """
    Single 3D scatter: bulk cloud colored by chosen metric; airfoil overlay.

    ``color_mode``: abs_error | rel_error | speed_pred | speed_actual | speed_mag_error
    """
    n = pos.shape[0]
    m = decimate_mask(n, max_points, decimate_seed)
    pos_np = pos[m].detach().cpu().numpy()
    pred_m = pred_v[m]
    act_m = actual_v[m]
    err_m = err_mag[m]

    c_np, _name, cbar, colorscale = _color_values(
        pred_m, act_m, err_m, color_mode, log_scale=log_scale
    )
    cmin, cmax = _cap_color_range(c_np, np.ones_like(c_np, dtype=bool), color_cap_percentile)

    marker_kw: dict = dict(
        size=2,
        color=c_np,
        colorscale=colorscale,
        opacity=0.5,
        showscale=True,
        colorbar=dict(title=cbar, len=0.7, thickness=14),
    )
    if cmin is not None and cmax is not None:
        marker_kw["cmin"] = cmin
        marker_kw["cmax"] = cmax

    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=pos_np[:, 0],
            y=pos_np[:, 1],
            z=pos_np[:, 2],
            mode="markers",
            marker=marker_kw,
            name="field",
            hovertemplate="x=%{x:.4g}<br>y=%{y:.4g}<br>z=%{z:.4g}<br>color=%{marker.color:.4g}<extra></extra>",
        )
    )
    _add_airfoil_trace(fig, pos, idcs_airfoil)
    _layout_3d(fig, title=_title_for_mode(color_mode), template=template)
    return fig


def _title_for_mode(mode: str) -> str:
    titles = {
        "abs_error": "3D: absolute velocity error ‖Δv‖₂",
        "rel_error": "3D: relative error ‖Δv‖₂ / (|v_actual| + ε)",
        "speed_pred": "3D: predicted speed |v|",
        "speed_actual": "3D: actual speed |v|",
        "speed_mag_error": "3D: | |v_pred| − |v_actual| |",
    }
    return titles.get(mode, "3D field")


def _add_airfoil_trace(fig: go.Figure, pos: torch.Tensor, idcs_airfoil: torch.Tensor) -> None:
    if idcs_airfoil.numel() == 0:
        return
    sp = pos[idcs_airfoil.long()].detach().cpu().numpy()
    fig.add_trace(
        go.Scatter3d(
            x=sp[:, 0],
            y=sp[:, 1],
            z=sp[:, 2],
            mode="markers",
            marker=dict(
                size=3,
                color="#ffcc00",
                opacity=0.95,
                line=dict(width=0.4, color="rgba(255,255,255,0.7)"),
            ),
            name="airfoil",
            hovertemplate="airfoil<extra></extra>",
        )
    )


def _layout_3d(fig: go.Figure, *, title: str, template: str) -> None:
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center"),
        template=template,
        margin=dict(l=0, r=0, t=50, b=0),
        height=580,
        scene=dict(
            aspectmode="data",
            xaxis=dict(title="x", backgroundcolor="rgba(0,0,0,0)", gridcolor="#444"),
            yaxis=dict(title="y", backgroundcolor="rgba(0,0,0,0)", gridcolor="#444"),
            zaxis=dict(title="z", backgroundcolor="rgba(0,0,0,0)", gridcolor="#444"),
            bgcolor="rgba(26,26,46,0.95)",
        ),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )


def figure_3d_speed_comparison(
    pos: torch.Tensor,
    pred_v: torch.Tensor,
    actual_v: torch.Tensor,
    idcs_airfoil: torch.Tensor,
    *,
    max_points: int,
    decimate_seed: int,
    log_scale: bool = False,
    template: str = "plotly_dark",
) -> go.Figure:
    """Two scatter3d panels: |v| pred vs |v| actual (same decimation, shared scale)."""
    n = pos.shape[0]
    m = decimate_mask(n, max_points, decimate_seed)
    pos_np = pos[m].detach().cpu().numpy()
    spd_p = speed_magnitude(pred_v[m]).detach().float().cpu().numpy()
    spd_a = speed_magnitude(actual_v[m]).detach().float().cpu().numpy()
    if log_scale:
        spd_p = np.log10(spd_p + 1e-8)
        spd_a = np.log10(spd_a + 1e-8)
    vmax = float(np.percentile(np.maximum(spd_p, spd_a), 99.5))
    vmin = float(np.percentile(np.minimum(spd_p, spd_a), 0.5))

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scatter3d"}, {"type": "scatter3d"}]],
        subplot_titles=(
            "|v| prediction" + (" (log₁₀)" if log_scale else ""),
            "|v| actual" + (" (log₁₀)" if log_scale else ""),
        ),
        horizontal_spacing=0.02,
    )
    for col, vals, lab in ((1, spd_p, "pred"), (2, spd_a, "actual")):
        fig.add_trace(
            go.Scatter3d(
                x=pos_np[:, 0],
                y=pos_np[:, 1],
                z=pos_np[:, 2],
                mode="markers",
                marker=dict(
                    size=2,
                    color=vals,
                    colorscale="Viridis",
                    cmin=vmin,
                    cmax=vmax,
                    opacity=0.55,
                    showscale=(col == 2),
                    colorbar=dict(title="log₁₀|v|" if log_scale else "|v|", len=0.65),
                ),
                name=lab,
                showlegend=False,
            ),
            row=1,
            col=col,
        )
    # Airfoil on both (small)
    if idcs_airfoil.numel() > 0:
        sp = pos[idcs_airfoil.long()].detach().cpu().numpy()
        for col in (1, 2):
            fig.add_trace(
                go.Scatter3d(
                    x=sp[:, 0],
                    y=sp[:, 1],
                    z=sp[:, 2],
                    mode="markers",
                    marker=dict(size=2, color="#ffcc00", opacity=0.9),
                    name="airfoil",
                    showlegend=(col == 2),
                ),
                row=1,
                col=col,
            )

    fig.update_layout(
        title=dict(text="3D: |v| prediction vs actual (same color scale)", x=0.5, xanchor="center"),
        template=template,
        height=520,
        margin=dict(t=80, b=0),
    )
    fig.update_scenes(
        aspectmode="data",
        xaxis_title="x",
        yaxis_title="y",
        zaxis_title="z",
    )
    return fig
