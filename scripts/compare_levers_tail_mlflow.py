#!/usr/bin/env python3
"""Compare MLflow metrics: levers baseline vs levers_tail (by config_file / model_family)."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import mlflow
from mlflow.tracking import MlflowClient


def _suffix_match(params: dict[str, str], needle: str) -> bool:
    cf = params.get("config_file") or ""
    return cf.endswith(needle)


def _find_latest_run(
    client: MlflowClient,
    experiment_id: str,
    *,
    config_suffix: str,
    exclude_suffix: str | None = None,
    require_val_l2: bool = True,
) -> str | None:
    runs = client.search_runs([experiment_id], max_results=100)
    best_id: str | None = None
    best_time = 0.0
    for r in runs:
        if not _suffix_match(r.data.params, config_suffix):
            continue
        if exclude_suffix and _suffix_match(r.data.params, exclude_suffix):
            continue
        if require_val_l2:
            h = client.get_metric_history(r.info.run_id, "val/l2_per_point_mean")
            if not h:
                continue
        t = r.info.start_time or 0
        if t >= best_time:
            best_time = float(t)
            best_id = r.info.run_id
    return best_id


def _metric_last(client: MlflowClient, run_id: str, key: str) -> float | None:
    h = client.get_metric_history(run_id, key)
    if not h:
        return None
    return float(h[-1].value)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Print val/l2 and per-timestep val L2 for levers vs levers_tail runs."
    )
    ap.add_argument(
        "--experiment",
        default="gram-warped-ifw-strong",
        help="MLflow experiment name",
    )
    ap.add_argument(
        "--baseline-run-id",
        default=None,
        help="Optional run_id for levers baseline; else latest config ..._levers.yaml",
    )
    ap.add_argument(
        "--tail-run-id",
        default=None,
        help="Optional run_id for tail; else latest config ..._levers_tail.yaml",
    )
    args = ap.parse_args()

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns"))
    client = MlflowClient()
    exp = client.get_experiment_by_name(args.experiment)
    if exp is None:
        # In CI / fresh clones, mlruns/ may not exist. This script is a convenience
        # helper and should not fail test suites in that case.
        print(f"No experiment {args.experiment!r} (nothing to compare).", file=sys.stderr)
        return 0
    eid = exp.experiment_id

    base_id = args.baseline_run_id or _find_latest_run(
        client,
        eid,
        config_suffix="strong_baseline_knn_mp_v2_levers.yaml",
        exclude_suffix="strong_baseline_knn_mp_v2_levers_tail.yaml",
    )
    tail_id = args.tail_run_id or _find_latest_run(
        client, eid, config_suffix="strong_baseline_knn_mp_v2_levers_tail.yaml"
    )
    if tail_id is None and args.tail_run_id is None:
        tail_id = _find_latest_run(
            client,
            eid,
            config_suffix="strong_baseline_knn_mp_v2_levers_tail.yaml",
            require_val_l2=False,
        )

    if not base_id:
        print(
            "No baseline levers run found (config_file ending ..._levers.yaml).",
            file=sys.stderr,
        )
        return 0
    if not tail_id:
        print(
            "No levers_tail run found (config_file ending ..._levers_tail.yaml).",
            file=sys.stderr,
        )
        return 0

    if not _metric_last(client, tail_id, "val/l2_per_point_mean"):
        print(
            "(note) Latest levers_tail run has no val/l2_per_point_mean yet — "
            "finish training or pass --tail-run-id.",
            file=sys.stderr,
        )

    rows = []
    for label, rid in (("levers", base_id), ("levers_tail", tail_id)):
        row = {"label": label, "run_id": rid}
        row["val/l2_per_point_mean"] = _metric_last(client, rid, "val/l2_per_point_mean")
        for i in range(5):
            row[f"tau{i}"] = _metric_last(client, rid, f"val/l2_timestep_{i}_mean")
        rows.append(row)

    cols = ["label", "run_id", "val/l2_per_point_mean"] + [f"tau{i}" for i in range(5)]
    print(" | ".join(cols))
    print("-" * (len(" | ".join(cols))))
    for row in rows:
        print(" | ".join(str(row.get(c, "")) for c in cols))

    b0 = rows[0]["tau0"] or 0.0
    t0 = rows[1]["tau0"] or 0.0
    if b0 > 0 and t0 > 0:
        rel = (t0 - b0) / b0 * 100.0
        print(f"\ntau0 regression check (tail vs baseline): {rel:+.2f}% (plan threshold often ~+2%)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
