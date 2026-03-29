#!/usr/bin/env python3
"""Evaluate multiple best checkpoints on the test split and rank by a KPI."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from training.hf_progress import silence_hf_download_progress

silence_hf_download_progress()

import mlflow

from training.eval_runner import evaluate_checkpoint_on_test
from training.leaderboard_rank import sort_leaderboard_rows
from training.yaml_config import load_yaml


def _resolve_path(p: str) -> Path:
    pp = Path(p)
    return pp.resolve() if pp.is_absolute() else (_REPO_ROOT / pp).resolve()


def _lower_is_better_from_manifest(manifest: dict[str, Any]) -> bool:
    if manifest.get("lower_is_better") is not None:
        return bool(manifest["lower_is_better"])
    entries = manifest.get("entries") or []
    if not entries:
        return True
    tc = _resolve_path(str(entries[0]["training_config"]))
    cfg = load_yaml(tc)
    ev_path = _resolve_path(str(cfg["paths"]["eval_protocol"]))
    ev = load_yaml(ev_path)
    return bool(ev.get("lower_is_better", True))


def _rank_metric_from_manifest(manifest: dict[str, Any]) -> str:
    m = manifest.get("rank_metric")
    if m:
        return str(m)
    return "test/l2_per_point_mean"


def _print_table(rows: list[dict[str, Any]], columns: list[str]) -> None:
    widths = [max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in columns]
    header = " | ".join(c.ljust(widths[i]) for i, c in enumerate(columns))
    sep = "-+-".join("-" * w for w in widths)
    print(header, flush=True)
    print(sep, flush=True)
    for r in rows:
        line = " | ".join(
            str(r.get(c, "")).ljust(widths[i]) for i, c in enumerate(columns)
        )
        print(line, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Run held-out test evaluation for each manifest entry and print a ranked table."
        )
    )
    ap.add_argument(
        "--manifest",
        type=str,
        required=True,
        help="YAML manifest with `entries` (label, training_config, checkpoint).",
    )
    ap.add_argument(
        "--output-dir",
        type=str,
        default="leaderboard_outputs",
        help="Directory for leaderboard.json and leaderboard.csv (created if missing).",
    )
    ap.add_argument(
        "--quiet",
        action="store_true",
        help="Less per-checkpoint progress output.",
    )
    ap.add_argument(
        "--mlflow",
        action="store_true",
        help="Log one MLflow run with leaderboard artifact and rank_* metrics.",
    )
    ap.add_argument(
        "--mlflow-run-name",
        type=str,
        default=None,
        help="MLflow run name when --mlflow is set.",
    )
    args = ap.parse_args()

    manifest_path = _resolve_path(args.manifest)
    if not manifest_path.is_file():
        print(f"Manifest not found: {manifest_path}", file=sys.stderr)
        return 1

    manifest = load_yaml(manifest_path)
    entries = manifest.get("entries")
    if not entries:
        print("Manifest has no entries.", file=sys.stderr)
        return 1

    rank_metric = _rank_metric_from_manifest(manifest)
    lower_is_better = _lower_is_better_from_manifest(manifest)

    results: list[dict[str, Any]] = []
    verbose = not args.quiet

    for ent in entries:
        label = str(ent.get("label") or Path(str(ent["checkpoint"])).stem)
        tc = _resolve_path(str(ent["training_config"]))
        ck = _resolve_path(str(ent["checkpoint"]))
        row: dict[str, Any] = {
            "label": label,
            "training_config": str(tc),
            "checkpoint": str(ck),
            "ok": False,
        }
        try:
            if not ck.is_file():
                raise FileNotFoundError(f"missing checkpoint {ck}")
            if not tc.is_file():
                raise FileNotFoundError(f"missing training config {tc}")
            metrics = evaluate_checkpoint_on_test(
                training_config_path=tc,
                checkpoint_path=ck,
                verbose=verbose,
            )
            n_te = int(metrics["n_test_batches"])
            if n_te == 0:
                raise RuntimeError("test split produced zero batches")
            row.update(metrics)
            row["ok"] = True
        except Exception as e:
            row["error"] = str(e)
            row[rank_metric] = float("nan")
        results.append(row)

    ok_rows = sort_leaderboard_rows(results, rank_metric, lower_is_better)

    ranked_out: list[dict[str, Any]] = []
    rank = 1
    for r in ok_rows:
        copy = {
            "rank": rank,
            "label": r["label"],
            "model": r.get("model", ""),
            rank_metric: r.get(rank_metric),
            "test/mse_velocity": r.get("test/mse_velocity"),
            "checkpoint": r["checkpoint"],
        }
        ranked_out.append(copy)
        rank += 1

    for r in results:
        if not r.get("ok"):
            ranked_out.append(
                {
                    "rank": None,
                    "label": r["label"],
                    "model": "",
                    rank_metric: None,
                    "test/mse_velocity": None,
                    "checkpoint": r.get("checkpoint", ""),
                    "error": r.get("error", ""),
                }
            )

    out_dir = _resolve_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = out_dir / f"leaderboard_{stamp}.json"
    csv_path = out_dir / f"leaderboard_{stamp}.csv"
    latest_json = out_dir / "leaderboard_latest.json"
    latest_csv = out_dir / "leaderboard_latest.csv"

    payload = {
        "generated_at_utc": stamp,
        "manifest": str(manifest_path),
        "rank_metric": rank_metric,
        "lower_is_better": lower_is_better,
        "results_full": results,
        "ranked": ranked_out,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    latest_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    csv_fields = [
        "rank",
        "label",
        "model",
        rank_metric,
        "test/mse_velocity",
        "checkpoint",
        "error",
    ]
    for path in (csv_path, latest_csv):
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
            w.writeheader()
            for block in ranked_out:
                w.writerow({fn: block.get(fn, "") for fn in csv_fields})

    print(
        f"\nLeaderboard (rank_metric={rank_metric!r}, lower_is_better={lower_is_better})\n",
        flush=True,
    )
    display_cols = ["rank", "label", "model", rank_metric, "test/mse_velocity"]
    _print_table([r for r in ranked_out if r.get("rank") is not None], display_cols)
    failed = [r for r in ranked_out if r.get("rank") is None]
    if failed:
        print("\nFailed / skipped:", flush=True)
        for r in failed:
            print(f"  {r.get('label')}: {r.get('error', '')}", flush=True)

    print(
        f"\nWrote {json_path} and {csv_path}",
        flush=True,
    )

    if args.mlflow:
        first_cfg = load_yaml(_resolve_path(str(entries[0]["training_config"])))
        exp_cfg = first_cfg.get("experiment", {})
        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns"))
        mlflow.set_experiment(exp_cfg.get("mlflow_experiment_name", "gram-warped-ifw"))
        run_name = args.mlflow_run_name or f"leaderboard-{stamp}"
        with mlflow.start_run(run_name=run_name):
            mlflow.log_param("leaderboard_manifest", str(manifest_path))
            mlflow.log_param("rank_metric", rank_metric)
            mlflow.log_param("lower_is_better", str(lower_is_better))
            mlflow.log_artifact(str(json_path), artifact_path="leaderboard")
            for r in ranked_out:
                rk = r.get("rank")
                if rk is None:
                    continue
                prefix = f"leaderboard/rank_{rk}"
                mlflow.log_param(f"{prefix}_label", r["label"])
                if r.get("model"):
                    mlflow.log_param(f"{prefix}_model", r["model"])
                v = r.get(rank_metric)
                if v is not None and not (isinstance(v, float) and math.isnan(v)):
                    safe_k = rank_metric.replace("/", "_")
                    mlflow.log_metric(f"{prefix}_{safe_k}", float(v))
                mse = r.get("test/mse_velocity")
                if mse is not None:
                    mlflow.log_metric(f"{prefix}_test_mse_velocity", float(mse))
        print(f"Logged MLflow run {run_name!r}", flush=True)

    return 0 if ok_rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
