#!/usr/bin/env python3
"""Streamlit UI: inspect model predictions vs ground truth (train/val/test)."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib.pyplot as plt
import streamlit as st
import torch

from training.diagnostics_velocity import (
    error_percentiles,
    global_l2_mse,
    per_point_error_magnitude,
    surface_bulk_summary,
    timestep_l2_table,
    worst_point_indices,
)
from training.inspect_predictions_common import (
    collect_buffered_batches,
    forward_predictions,
    load_model_from_checkpoint,
    load_train_config_only,
    resolve_device,
)
from training.inspect_plotly_3d import figure_3d_scatter, figure_3d_speed_comparison
from training.inspect_viz import (
    frames_for_animation,
    plot_error_slice_only,
    plot_slice_row,
    png_list_to_gif_bytes,
    speed_magnitude,
)


@st.cache_resource
def _cached_model(config_path: str, checkpoint_path: str, device_str: str) -> torch.nn.Module:
    dev = torch.device(device_str)
    return load_model_from_checkpoint(Path(config_path), Path(checkpoint_path), dev)


def _axis_coord_defaults(pos: torch.Tensor, axis: str) -> float:
    i = {"x": 0, "y": 1, "z": 2}[axis]
    return float(pos[:, i].median().cpu())


def main() -> None:
    st.set_page_config(page_title="Inspect predictions", layout="wide")
    st.title("Inspect model predictions vs actual")

    st.caption(
        "Training minimizes MSE to simulation labels — compare **prediction** to **actual** "
        "(velocity_out). On walls, judge the model vs data, then data vs your BC expectations."
    )

    if "batches" not in st.session_state:
        st.session_state.batches = None
    if "preds" not in st.session_state:
        st.session_state.preds = None
    if "batch_idx" not in st.session_state:
        st.session_state.batch_idx = 0

    with st.sidebar:
        st.header("Setup")
        cfg_default = str(_ROOT / "configs/example_mlp.yaml")
        ckpt_default = str(_ROOT / "checkpoints/mlp_last.pt")
        config_path = st.text_input("Training YAML", value=cfg_default)
        checkpoint_path = st.text_input("Checkpoint .pt", value=ckpt_default)
        split = st.selectbox("Split", ("val", "test", "train"), index=0)
        buffer_k = st.number_input("Buffer size K (batches)", min_value=1, max_value=64, value=8)
        device_choice = st.selectbox("Device", ("auto", "cpu", "cuda", "mps"), index=0)
        point_seed = st.number_input("Point subsample seed", value=12345, step=1)

        sample_mode = st.radio("Sample mode", ("sequential", "random"), horizontal=True)
        rng_seed = st.number_input("Random seed (random mode)", value=0, step=1)

        max_3d = st.number_input("Max 3D points", min_value=2000, max_value=100000, value=40000, step=1000)

        if st.button("Load data + run model", type="primary"):
            with st.spinner("Loading batches and running forward…"):
                train_cfg = load_train_config_only(Path(config_path))
                dev_ov = None if device_choice == "auto" else device_choice
                dev = resolve_device(train_cfg, dev_ov)
                device_str = str(dev)
                model = _cached_model(
                    str(Path(config_path).resolve()),
                    str(Path(checkpoint_path).resolve()),
                    device_str,
                )
                batches = collect_buffered_batches(
                    training_config_path=Path(config_path),
                    phase=split,  # type: ignore[arg-type]
                    device=dev,
                    max_batches=int(buffer_k),
                    point_seed=int(point_seed),
                )
                preds: list[torch.Tensor] = []
                for b in batches:
                    preds.append(forward_predictions(model, b))
                st.session_state.batches = batches
                st.session_state.preds = preds
                st.session_state._device_str = device_str
                gen0 = torch.Generator()
                gen0.manual_seed(int(rng_seed))
                if sample_mode == "random":
                    st.session_state.batch_idx = int(
                        torch.randint(0, len(batches), (1,), generator=gen0).item()
                    )
                else:
                    st.session_state.batch_idx = min(
                        int(st.session_state.get("batch_idx", 0)),
                        max(len(batches) - 1, 0),
                    )
            st.success(f"Loaded {len(batches)} batches.")

    batches = st.session_state.batches
    preds = st.session_state.preds
    if not batches or not preds:
        st.info("Configure the sidebar and click **Load data + run model**.")
        return

    nbuf = len(batches)
    st.subheader("Choose batch")
    col_a, col_b = st.columns(2)
    with col_a:
        if sample_mode == "sequential":
            bi = st.slider("Batch index", 0, nbuf - 1, st.session_state.batch_idx)
            st.session_state.batch_idx = bi
        else:
            bi = st.session_state.batch_idx
            st.write(f"Current batch index: **{bi}** (0 … {nbuf - 1})")
            if st.button("Resample random batch"):
                gen = torch.Generator()
                gen.manual_seed(int(rng_seed) + 17)
                st.session_state.batch_idx = int(torch.randint(0, nbuf, (1,), generator=gen).item())
                st.rerun()
            bi = st.session_state.batch_idx
    with col_b:
        t_in = batches[bi].velocity_in.shape[1]
        t_out = batches[bi].velocity_out.shape[1]
        st.write(f"T_in={t_in}, T_out={t_out}, N={batches[bi].pos.shape[1]}")

    batch = batches[bi]
    pred = preds[bi]
    target = batch.velocity_out
    Bitem = batch.t.shape[0]
    if Bitem > 1:
        b_in = st.slider("Batch tensor index (within GPU batch)", 0, Bitem - 1, 0)
    else:
        b_in = 0

    err_bt = per_point_error_magnitude(pred, target)

    k = st.slider("Output timestep k", 0, t_out - 1, 0)

    t_row = batch.t[b_in]
    t_label = ""
    if t_row.numel() >= t_in + k + 1:
        t_label = f" (t[{t_in + k}] = {float(t_row[t_in + k].cpu()):.6g})"

    st.markdown(f"### Metrics for batch **{bi}**, step **k = {k}**{t_label}")

    l2, mse = global_l2_mse(pred, target)
    l2s, mses = timestep_l2_table(pred, target)
    pct = error_percentiles(err_bt[b_in, k])
    surf = surface_bulk_summary(
        pred, target, bi=b_in, k=k, idcs_airfoil=batch.idcs_airfoil
    )

    mcol1, mcol2, mcol3 = st.columns(3)
    mcol1.metric("Global mean L2 (all B,T,N)", f"{l2:.6f}")
    mcol2.metric("Global MSE", f"{mse:.6f}")
    mcol3.metric("Mean L2 @ this k", f"{l2s[k]:.6f}")

    with st.expander("Per-timestep L2 / MSE"):
        st.table(
            {
                "k": list(range(len(l2s))),
                "mean_L2": l2s,
                "MSE": mses,
            }
        )

    with st.expander(f"Error percentiles @ this k (batch item {b_in})"):
        st.json(pct)

    with st.expander("Surface vs bulk @ this k"):
        st.json({k: float(v) if isinstance(v, float) else v for k, v in surf.items()})

    worst_n = st.number_input("Worst points table rows", 1, 50, 15)
    err_1 = err_bt[b_in, k]
    w_idx, w_val = worst_point_indices(err_1, int(worst_n))
    pos_cpu = batch.pos[b_in].detach().cpu()
    rows = []
    for j in range(w_idx.numel()):
        ij = int(w_idx[j].cpu())
        rows.append(
            {
                "idx": ij,
                "err_L2": float(w_val[j].cpu()),
                "x": float(pos_cpu[ij, 0]),
                "y": float(pos_cpu[ij, 1]),
                "z": float(pos_cpu[ij, 2]),
            }
        )
    with st.expander("Largest point errors"):
        st.dataframe(rows)

    st.markdown("### 2D slice (matplotlib)")
    axis = st.selectbox("Slice normal", ("x", "y", "z"), index=2)
    default_c = _axis_coord_defaults(batch.pos[b_in], axis)
    slice_coord = st.number_input("Slice coordinate", value=float(default_c), format="%.6f")

    pos = batch.pos[b_in]
    pv = pred[b_in, k]
    av = target[b_in, k]
    em = err_1

    fig_row = plot_slice_row(
        pos, pv, av, em,         batch.idcs_airfoil[b_in], axis, slice_coord, k_label=f"k={k}"
    )
    st.pyplot(fig_row)
    plt.close(fig_row)

    fig_err = plot_error_slice_only(
        pos, em, batch.idcs_airfoil[b_in], axis, slice_coord, k_label=f"k={k}"
    )
    st.pyplot(fig_err)
    plt.close(fig_err)

    st.markdown("### 3D (Plotly, decimated)")
    view_3d = st.radio(
        "3D layout",
        ("Single cloud (pick color metric)", "Side-by-side |v| pred vs actual"),
        horizontal=True,
    )
    dec_seed = st.number_input("3D decimate seed", 0, 2**31 - 1, 42)
    if view_3d.startswith("Single"):
        _metric_labels = {
            "rel_error": "Relative ‖Δv‖₂ / (|vₐ|+ε) — best default when |v| varies (stagnation vs jet)",
            "abs_error": "Absolute ‖Δv‖₂ — raw vector error",
            "speed_mag_error": "| |vₚ| − |vₐ| | — too fast/slow only (no direction)",
            "speed_pred": "|v| prediction",
            "speed_actual": "|v| actual",
        }
        color_mode = st.selectbox(
            "Color metric",
            list(_metric_labels.keys()),
            index=0,
            format_func=lambda k: _metric_labels[k],
        )
        log_colors = st.checkbox("log₁₀ color scale", value=False)
        cap_outliers = st.checkbox("Cap color scale at p1–p99 (reduce outlier washout)", value=True)
        fig3 = figure_3d_scatter(
            pos,
            pv,
            av,
            em,
            batch.idcs_airfoil[b_in],
            max_points=int(max_3d),
            decimate_seed=int(dec_seed),
            color_mode=color_mode,
            log_scale=log_colors,
            color_cap_percentile=99.0 if cap_outliers else None,
        )
    else:
        log_v = st.checkbox("log₁₀ |v| coloring", value=False, key="side_by_side_log_v")
        fig3 = figure_3d_speed_comparison(
            pos,
            pv,
            av,
            batch.idcs_airfoil[b_in],
            max_points=int(max_3d),
            decimate_seed=int(dec_seed),
            log_scale=log_v,
        )
    st.plotly_chart(fig3, use_container_width=True)
    st.caption(
        "Use **relative error** to see mistakes where the flow is slow; **absolute error** for overall "
        "severity. **Side-by-side |v|** checks spatial pattern of speed without mixing in direction error."
    )

    st.markdown("### Histograms")
    fig_h, axh = plt.subplots(1, 2, figsize=(9, 3))
    idcs = batch.idcs_airfoil[b_in]
    if idcs.numel() > 0:
        sp_pred = speed_magnitude(pv[idcs])
        sp_act = speed_magnitude(av[idcs])
        axh[0].hist(
            sp_pred.detach().cpu().numpy(),
            bins=40,
            alpha=0.6,
            label="pred |v|",
            density=True,
        )
        axh[0].hist(
            sp_act.detach().cpu().numpy(),
            bins=40,
            alpha=0.6,
            label="actual |v|",
            density=True,
        )
        axh[0].legend()
        axh[0].set_title("Surface |v|")
    axh[1].hist(em.detach().cpu().numpy(), bins=60, color="steelblue", alpha=0.8, density=True)
    axh[1].set_title("||pred−actual|| (all points)")
    fig_h.tight_layout()
    st.pyplot(fig_h)
    plt.close(fig_h)

    st.markdown("### Animation over output steps (GIF)")
    dur_ms = st.number_input("Frame duration (ms)", 50, 1000, 350)
    if st.button("Build GIF (pred | actual | error)"):
        with st.spinner("Rendering frames…"):
            frames = frames_for_animation(
                pos,
                pred[b_in],
                target[b_in],
                err_bt[b_in],
                batch.idcs_airfoil[b_in],
                axis,
                slice_coord,
                t_out,
                t_in,
                t_row,
            )
            gif_b = png_list_to_gif_bytes(frames, duration_ms=int(dur_ms))
        st.image(gif_b, caption="Prediction / actual / error through k")


if __name__ == "__main__":
    main()
