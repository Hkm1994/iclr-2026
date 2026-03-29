#!/usr/bin/env python3
"""Print MLflow runs sorted by primary KPI (from eval_protocol.yaml)."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import mlflow
import pandas as pd

from training.yaml_config import load_yaml


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="gram-warped-ifw")
    ap.add_argument("--eval-protocol", default="configs/eval_protocol.yaml")
    ap.add_argument("--data-split-version", default=None, help="Filter runs by param")
    ap.add_argument("--model-family", default=None)
    ap.add_argument("--limit", type=int, default=30)
    args = ap.parse_args()

    ev = load_yaml(args.eval_protocol)
    primary = ev["primary_kpi"]
    lower_better = bool(ev.get("lower_is_better", True))

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns"))
    exp = mlflow.get_experiment_by_name(args.experiment)
    if exp is None:
        print(f"No experiment {args.experiment!r}")
        return 1

    runs = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        max_results=args.limit * 3,
    )
    if runs.empty:
        print("No runs.")
        return 0

    if args.data_split_version:
        runs = runs[runs["params.data_split_version"] == args.data_split_version]
    if args.model_family:
        runs = runs[runs["params.model_family"] == args.model_family]

    col = f"metrics.{primary}"
    if col not in runs.columns:
        print(f"Metric column {col!r} missing; available metric columns:", [c for c in runs.columns if c.startswith("metrics.")])
        return 1

    runs = runs.dropna(subset=[col])
    runs = runs.sort_values(col, ascending=lower_better).head(args.limit)

    display_cols = [
        "run_id",
        "params.model_family",
        "params.data_split_version",
        "params.eval_protocol_version",
        col,
    ]
    display_cols = [c for c in display_cols if c in runs.columns]
    print(runs[display_cols].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
