#!/usr/bin/env python3
"""CLI: print diagnostics and optional PNG figures (pred | actual | error)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch

from training.diagnostics_velocity import (
    error_percentiles,
    global_l2_mse,
    per_point_error_magnitude,
    surface_bulk_summary,
    timestep_l2_table,
)
from training.inspect_predictions_common import (
    collect_buffered_batches,
    forward_predictions,
    load_model_from_checkpoint,
    load_train_config_only,
    resolve_device,
)
from training.inspect_viz import figure_to_png_bytes, plot_error_slice_only, plot_slice_row


def main() -> int:
    ap = argparse.ArgumentParser(description="Inspect predictions vs velocity_out.")
    ap.add_argument("--config", type=str, required=True, help="Training YAML")
    ap.add_argument("--checkpoint", type=str, required=True, help="State dict .pt")
    ap.add_argument(
        "--phase",
        type=str,
        default="val",
        choices=("train", "val", "test"),
        help="Data split",
    )
    ap.add_argument("--max-batches", type=int, default=4, help="Batches to buffer")
    ap.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="If set, pick random buffer index in [0, max-batches) with this seed",
    )
    ap.add_argument("--point-seed", type=int, default=None, help="Subsample seed")
    ap.add_argument("--device", type=str, default=None, help="cpu | cuda | mps (default: auto)")
    ap.add_argument("--save-dir", type=str, default=None, help="Directory for PNGs")
    ap.add_argument("--slice-axis", type=str, default="z", choices=("x", "y", "z"))
    ap.add_argument("--batch-idx", type=int, default=0, help="Index into buffered batches")
    ap.add_argument("--k", type=int, default=0, help="Output timestep index")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    ckpt_path = Path(args.checkpoint)
    train_cfg = load_train_config_only(cfg_path)
    dev = resolve_device(train_cfg, args.device)
    model = load_model_from_checkpoint(cfg_path, ckpt_path, dev)

    batches = collect_buffered_batches(
        training_config_path=cfg_path,
        phase=args.phase,  # type: ignore[arg-type]
        device=dev,
        max_batches=int(args.max_batches),
        point_seed=args.point_seed,
    )
    if not batches:
        print("No batches loaded.", file=sys.stderr)
        return 1

    bi = int(args.batch_idx)
    if args.random_seed is not None:
        g = torch.Generator()
        g.manual_seed(int(args.random_seed))
        bi = int(torch.randint(0, len(batches), (1,), generator=g).item())

    bi = max(0, min(bi, len(batches) - 1))
    batch = batches[bi]
    pred = forward_predictions(model, batch)
    tgt = batch.velocity_out
    err = per_point_error_magnitude(pred, tgt)

    l2, mse = global_l2_mse(pred, tgt)
    l2s, mses = timestep_l2_table(pred, tgt)
    print(f"Buffered batches: {len(batches)}, using index {bi}")
    print(f"Global mean L2: {l2:.6f}  MSE: {mse:.6f}")
    print("Per-timestep mean L2:", ", ".join(f"{x:.5f}" for x in l2s))

    t_in = batch.velocity_in.shape[1]
    k = int(args.k)
    if k < 0 or k >= batch.velocity_out.shape[1]:
        k = 0
    b_in = 0
    pct = error_percentiles(err[b_in, k])
    print(f"Error percentiles @ k={k}: {pct}")
    surf = surface_bulk_summary(
        pred, tgt, bi=b_in, k=k, idcs_airfoil=batch.idcs_airfoil
    )
    print("Surface/bulk summary:", surf)

    if args.save_dir:
        out = Path(args.save_dir)
        out.mkdir(parents=True, exist_ok=True)
        pos = batch.pos[b_in]
        axis = args.slice_axis
        i = {"x": 0, "y": 1, "z": 2}[axis]
        coord = float(pos[:, i].median().cpu())
        fig1 = plot_slice_row(
            pos,
            pred[b_in, k],
            tgt[b_in, k],
            err[b_in, k],
            batch.idcs_airfoil[b_in],
            axis,
            coord,
            k_label=f"k={k}",
        )
        (out / f"slice_pred_actual_err_{args.phase}_b{bi}_k{k}.png").write_bytes(
            figure_to_png_bytes(fig1)
        )
        fig2 = plot_error_slice_only(
            pos,
            err[b_in, k],
            batch.idcs_airfoil[b_in],
            axis,
            coord,
            k_label=f"k={k}",
        )
        (out / f"slice_error_robust_{args.phase}_b{bi}_k{k}.png").write_bytes(
            figure_to_png_bytes(fig2)
        )
        print(f"Wrote PNGs to {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
