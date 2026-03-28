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
import torch

from models.registry import get_model_class
from training.epoch_loop import evaluate_split_full
from training.memory_utils import release_training_memory
from training.mlflow_steps import new_stream_counters
from training.seeds import seed_all
from training.yaml_config import load_yaml


def _device_from_cfg(train_cfg: dict) -> torch.device:
    d = train_cfg.get("device")
    if d:
        return torch.device(d)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

    cfg_path = Path(args.config)
    ckpt_path = Path(args.checkpoint)
    cfg = load_yaml(cfg_path)
    paths = cfg["paths"]
    train_cfg = cfg["train"]
    exp_cfg = cfg.get("experiment", {})

    if not ckpt_path.is_file():
        print(f"Checkpoint not found: {ckpt_path}", file=sys.stderr)
        return 1

    data_split_path = Path(paths["data_split"])
    eval_path = Path(paths["eval_protocol"])
    ds_cfg = load_yaml(data_split_path)
    ev_cfg = load_yaml(eval_path)
    master_seed = int(ds_cfg["seed"])
    seed_all(master_seed)

    device = _device_from_cfg(train_cfg)
    model_name = train_cfg["model"]
    model_cls = get_model_class(model_name)
    model_cfg: dict = {"skip_weights": True}
    extra_mc = train_cfg.get("model_config")
    if isinstance(extra_mc, dict):
        model_cfg.update(extra_mc)
    model_cfg["skip_weights"] = True
    model = model_cls(config=model_cfg)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()

    batch_size = int(train_cfg.get("batch_size", 1))
    eval_sub = ev_cfg.get("eval_subsample_N")
    eval_seed = int(ev_cfg.get("eval_point_subsample_seed", 0))
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
        "seed": master_seed,
    }

    stream_steps = new_stream_counters()

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
                f"Eval-only | model={model_name} device={device} | "
                f"checkpoint={ckpt_path} | data_split={data_split_path}",
                flush=True,
            )

        test_mse, test_l2, n_te, test_lt = evaluate_split_full(
            model=model,
            data_split_path=data_split_path,
            device=device,
            batch_size=batch_size,
            eval_subsample_N=eval_sub,
            eval_seed=eval_seed,
            phase="test",
            epoch_idx=0,
            log_every_n_batches=log_every_val,
            verbose=verbose,
            heartbeat_seconds=heartbeat_seconds,
            eval_stream_step_counter=stream_steps.test_batch,
            run_label="held-out test (eval_checkpoint)",
        )

        if n_te == 0:
            print(
                "No test batches (set split.hash_ids.test_fraction > 0 in data_split.yaml).",
                file=sys.stderr,
            )
            return 1

        ts = stream_steps.test_batch[0]
        mlflow.log_metric("test/mse_velocity", test_mse, step=ts)
        mlflow.log_metric("test/l2_per_point_mean", test_l2, step=ts)
        mlflow.log_metric("test/batches", float(n_te), step=ts)
        for k, v in test_lt.items():
            mlflow.log_metric(k, v, step=ts)

        if verbose:
            print(
                f"Test summary: mean_mse={test_mse:.6f} mean_l2={test_l2:.6f} "
                f"batches={n_te} | mlflow step={ts}",
                flush=True,
            )

    release_training_memory()
    print("Done. MLflow UI: mlflow ui --backend-store-uri ./mlruns", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
