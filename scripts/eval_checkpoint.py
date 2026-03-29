#!/usr/bin/env python3
"""Load a trained checkpoint and evaluate the held-out test split; log test/* to MLflow."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from training.hf_progress import silence_hf_download_progress

silence_hf_download_progress()

import mlflow

from training.eval_runner import evaluate_checkpoint_on_test
from training.memory_utils import release_training_memory
from training.seeds import seed_all
from training.yaml_config import load_yaml


def _positive_int_or_none(x) -> int | None:
    if x is None:
        return None
    n = int(x)
    if n <= 0:
        return None
    return n


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Run test-split evaluation only and log metrics to MLflow."
    )
    ap.add_argument(
        "--config",
        type=str,
        required=True,
        help="Training YAML (paths, train.*, experiment).",
    )
    ap.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to state_dict .pt (e.g. checkpoints/strong_mlp_best.pt).",
    )
    ap.add_argument(
        "--quiet",
        action="store_true",
        help="Less terminal output (MLflow logging still runs).",
    )
    ap.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional MLflow run name; default derives from checkpoint + timestamp.",
    )
    args = ap.parse_args()

    cfg_path = Path(args.config).resolve()
    ckpt_path = Path(args.checkpoint).resolve()
    cfg = load_yaml(cfg_path)
    paths = cfg["paths"]
    train_cfg = cfg["train"]
    exp_cfg = cfg.get("experiment", {})

    if not ckpt_path.is_file():
        print(f"Checkpoint not found: {ckpt_path}", file=sys.stderr)
        return 1

    data_split_path = Path(paths["data_split"])
    if not data_split_path.is_absolute():
        data_split_path = (_REPO_ROOT / data_split_path).resolve()
    ds_cfg = load_yaml(data_split_path)
    master_seed = int(ds_cfg["seed"])
    seed_all(master_seed)

    eval_path = Path(paths["eval_protocol"])
    if not eval_path.is_absolute():
        eval_path = (_REPO_ROOT / eval_path).resolve()
    ev_cfg = load_yaml(eval_path)

    verbose = bool(train_cfg.get("verbose", True)) and not args.quiet
    log_every_val = _positive_int_or_none(
        train_cfg.get(
            "log_every_n_val_batches",
            train_cfg.get("log_every_n_train_batches", 5),
        )
    )
    hb_raw = train_cfg.get("heartbeat_seconds")
    heartbeat_seconds: float | None = None
    if hb_raw is not None:
        heartbeat_seconds = float(hb_raw)
        if heartbeat_seconds <= 0:
            heartbeat_seconds = None

    eval_sub = ev_cfg.get("eval_subsample_N")
    eval_preforward_subsample_N = train_cfg.get("eval_preforward_subsample_N")
    model_name = train_cfg["model"]

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns"))
    mlflow.set_experiment(exp_cfg.get("mlflow_experiment_name", "gram-warped-ifw"))

    run_name = args.run_name
    if not run_name:
        run_name = f"eval-{ckpt_path.stem}-{model_name}"

    params = {
        "eval_only": True,
        "checkpoint_path": str(ckpt_path.resolve()),
        "config_file": str(cfg_path.resolve()),
        "model": model_name,
        "data_split_version": str(ds_cfg.get("version", "unknown")),
        "eval_protocol_version": str(ev_cfg.get("version", "unknown")),
        "eval_subsample_N": eval_sub if eval_sub is not None else "full",
        "eval_preforward_subsample_N": (
            _positive_int_or_none(eval_preforward_subsample_N)
            if eval_preforward_subsample_N is not None
            else "off"
        ),
        "seed": master_seed,
    }

    existing = os.environ.get("MLFLOW_RUN_ID")
    if existing:
        print(f"Logging metrics to existing MLflow run MLFLOW_RUN_ID={existing}", flush=True)

    ctx = (
        mlflow.start_run(run_id=existing)
        if existing
        else mlflow.start_run(run_name=run_name)
    )

    with ctx:
        if not existing:
            mlflow.log_params(params)
            mlflow.log_artifact(str(cfg_path), artifact_path="config")
        if verbose:
            print(
                f"Eval-only | model={model_name} | "
                f"checkpoint={ckpt_path} | data_split={data_split_path}",
                flush=True,
            )

        try:
            results = evaluate_checkpoint_on_test(
                training_config_path=cfg_path,
                checkpoint_path=ckpt_path,
                verbose=verbose,
                log_every_n_batches=log_every_val,
                heartbeat_seconds=heartbeat_seconds,
            )
        finally:
            release_training_memory()

        n_te = int(results["n_test_batches"])
        if n_te == 0:
            print(
                "No test batches (set split.hash_ids.test_fraction > 0 in data_split.yaml).",
                file=sys.stderr,
            )
            return 1

        test_mse = results["test/mse_velocity"]
        test_l2 = results["test/l2_per_point_mean"]
        ts = int(results["mlflow_step"])
        mlflow.log_metric("test/mse_velocity", test_mse, step=ts)
        mlflow.log_metric("test/l2_per_point_mean", test_l2, step=ts)
        mlflow.log_metric("test/batches", float(n_te), step=ts)
        for k, v in results.items():
            if k.startswith("test/") and k not in (
                "test/mse_velocity",
                "test/l2_per_point_mean",
            ):
                if isinstance(v, (int, float)):
                    mlflow.log_metric(k, float(v), step=ts)

        if verbose:
            print(
                f"Test summary: mean_mse={test_mse:.6f} mean_l2={test_l2:.6f} "
                f"batches={n_te} | mlflow step={ts}",
                flush=True,
            )

    print("Done. MLflow UI: mlflow ui --backend-store-uri ./mlruns", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
