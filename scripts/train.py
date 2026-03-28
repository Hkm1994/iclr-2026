#!/usr/bin/env python3
"""Train with HF streaming, MLflow logging, central data_split + eval_protocol."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import mlflow
import torch
from torch.optim import AdamW

from models.registry import get_model_class
from training.epoch_loop import is_better, train_one_epoch, validate_full
from training.hf_dataset import streaming_batches
from training.metrics import l2_per_point_mean, mse_velocity, subsample_points
from training.seeds import seed_all
from training.yaml_config import load_yaml


def _device_from_cfg(train_cfg: dict) -> torch.device:
    d = train_cfg.get("device")
    if d:
        return torch.device(d)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _run_legacy_step_training(
    *,
    model,
    opt,
    data_split_path: Path,
    device: torch.device,
    train_cfg: dict,
    ev_cfg: dict,
    max_train: int,
    max_val: int,
    grad_accum: int,
    batch_size: int,
    train_sub,
    point_seed_train: int,
    eval_seed: int,
    eval_sub,
) -> None:
    step = 0
    accum = 0
    opt.zero_grad(set_to_none=True)

    train_it = streaming_batches(
        data_split_path,
        "train",
        device=device,
        batch_size=batch_size,
        train_subsample_N=train_sub,
        point_seed=point_seed_train,
        max_batches=None,
    )

    for batch in train_it:
        pred = model(batch.t, batch.pos, batch.idcs_airfoil, batch.velocity_in)
        loss = mse_velocity(pred, batch.velocity_out)
        (loss / grad_accum).backward()
        accum += 1
        if accum >= grad_accum:
            opt.step()
            opt.zero_grad(set_to_none=True)
            accum = 0
        mlflow.log_metric("train/mse_velocity", float(loss.detach().cpu()), step=step)
        step += 1
        if step >= max_train:
            break

    if accum > 0:
        opt.step()
        opt.zero_grad(set_to_none=True)

    g = torch.Generator()
    g.manual_seed(eval_seed)
    val_step = 0
    val_mse_acc = 0.0
    val_l2_acc = 0.0
    val_count = 0
    val_it = streaming_batches(
        data_split_path,
        "val",
        device=device,
        batch_size=batch_size,
        train_subsample_N=None,
        point_seed=eval_seed,
        max_batches=max_val,
    )
    with torch.no_grad():
        for batch in val_it:
            pred = model(batch.t, batch.pos, batch.idcs_airfoil, batch.velocity_in)
            p, tgt = subsample_points(pred, batch.velocity_out, eval_sub, generator=g)
            vm = mse_velocity(p, tgt)
            vl = l2_per_point_mean(p, tgt)
            val_mse_acc += float(vm.cpu())
            val_l2_acc += float(vl.cpu())
            val_count += 1
            val_step += 1
            if val_step >= max_val:
                break

    if val_count > 0:
        mlflow.log_metric("val/mse_velocity", val_mse_acc / val_count)
        mlflow.log_metric("val/l2_per_point_mean", val_l2_acc / val_count)


def _run_epoch_training(
    *,
    model,
    opt,
    data_split_path: Path,
    device: torch.device,
    train_cfg: dict,
    ev_cfg: dict,
    grad_accum: int,
    batch_size: int,
    train_sub,
    point_seed_train: int,
    eval_seed: int,
    eval_sub,
    max_epochs: int,
    min_epochs: int,
    patience: int,
    min_delta: float,
    monitor: str,
    lower_is_better: bool,
    best_ckpt_path: Path,
) -> None:
    metric_keys = {
        "val/mse_velocity": "mse",
        "val/l2_per_point_mean": "l2",
    }
    if monitor not in metric_keys:
        raise ValueError(
            f"early_stopping_monitor must be one of {list(metric_keys)}, got {monitor!r}"
        )

    best = float("inf") if lower_is_better else float("-inf")
    epochs_without_improve = 0

    for epoch in range(max_epochs):
        seed_ep = int(point_seed_train) + epoch * 1_000_003
        train_loss, n_tr = train_one_epoch(
            model=model,
            opt=opt,
            data_split_path=data_split_path,
            device=device,
            batch_size=batch_size,
            grad_accum=grad_accum,
            train_subsample_N=train_sub,
            point_seed=seed_ep,
        )
        val_mse, val_l2, n_val = validate_full(
            model=model,
            data_split_path=data_split_path,
            device=device,
            batch_size=batch_size,
            eval_subsample_N=eval_sub,
            eval_seed=eval_seed,
        )
        if n_val == 0:
            print("Validation produced zero batches; check data_split and HF access.")
            break

        mlflow.log_metric("train/mse_velocity_epoch_mean", train_loss, step=epoch)
        mlflow.log_metric("train/batches_per_epoch", float(n_tr), step=epoch)
        mlflow.log_metric("val/mse_velocity", val_mse, step=epoch)
        mlflow.log_metric("val/l2_per_point_mean", val_l2, step=epoch)
        mlflow.log_metric("val/batches_per_epoch", float(n_val), step=epoch)

        current = val_mse if metric_keys[monitor] == "mse" else val_l2

        if is_better(current, best, min_delta=min_delta, lower_is_better=lower_is_better):
            best = current
            epochs_without_improve = 0
            best_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), best_ckpt_path)
            mlflow.log_metric("val/best_" + monitor.replace("/", "_"), best, step=epoch)
        else:
            epochs_without_improve += 1

        if epoch + 1 >= min_epochs and epochs_without_improve >= patience:
            mlflow.log_param("stopped_epoch", epoch + 1)
            mlflow.log_param("stop_reason", "early_stopping")
            break
    else:
        mlflow.log_param("stopped_epoch", max_epochs)
        mlflow.log_param("stop_reason", "max_epochs")

    mlflow.log_param("best_" + monitor.replace("/", "_"), best)
    if best_ckpt_path.is_file():
        mlflow.log_artifact(str(best_ckpt_path), artifact_path="checkpoints")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/example_mlp.yaml")
    ap.add_argument("--max-train-steps", type=int, default=None)
    args = ap.parse_args()

    cfg_path = Path(args.config)
    cfg = load_yaml(cfg_path)
    paths = cfg["paths"]
    train_cfg = cfg["train"]
    exp_cfg = cfg.get("experiment", {})

    data_split_path = Path(paths["data_split"])
    eval_path = Path(paths["eval_protocol"])
    ds_cfg = load_yaml(data_split_path)
    ev_cfg = load_yaml(eval_path)

    master_seed = int(ds_cfg["seed"])
    seed_all(master_seed)

    device = _device_from_cfg(train_cfg)
    model_name = train_cfg["model"]
    model_cls = get_model_class(model_name)

    model_cfg = {"skip_weights": True}
    if train_cfg.get("checkpoint_path"):
        model = model_cls(config=model_cfg)
        ck = torch.load(train_cfg["checkpoint_path"], map_location="cpu", weights_only=True)
        model.load_state_dict(ck)
    else:
        model = model_cls(config=model_cfg)
    model = model.to(device)

    lr = float(train_cfg.get("lr", 1e-4))
    wd = float(train_cfg.get("weight_decay", 0.0))
    opt = AdamW(model.parameters(), lr=lr, weight_decay=wd)

    batch_size = int(train_cfg.get("batch_size", 1))
    grad_accum = int(train_cfg.get("grad_accum_steps", 1))
    train_sub = train_cfg.get("train_subsample_N")
    point_seed_train = int(train_cfg.get("train_point_seed", 0))
    eval_sub = ev_cfg.get("eval_subsample_N")
    eval_seed = int(ev_cfg.get("eval_point_subsample_seed", 0))

    max_epochs = train_cfg.get("max_epochs")
    use_epochs = max_epochs is not None

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns"))
    mlflow.set_experiment(exp_cfg.get("mlflow_experiment_name", "gram-warped-ifw"))

    data_split_version = str(ds_cfg.get("version", "unknown"))
    eval_version = str(ev_cfg.get("version", "unknown"))

    params = {
        "model_family": train_cfg.get("model_family", model_name),
        "model": model_name,
        "data_split_version": data_split_version,
        "eval_protocol_version": eval_version,
        "seed": master_seed,
        "lr": lr,
        "weight_decay": wd,
        "batch_size": batch_size,
        "train_subsample_N": train_sub if train_sub is not None else "full",
        "eval_subsample_N": eval_sub if eval_sub is not None else "full",
        "config_file": str(cfg_path),
        "training_mode": "epoch" if use_epochs else "steps",
    }
    if use_epochs:
        monitor = train_cfg.get("early_stopping_monitor") or ev_cfg.get(
            "primary_kpi", "val/l2_per_point_mean"
        )
        lower = train_cfg.get("early_stopping_lower_is_better")
        if lower is None:
            lower = bool(ev_cfg.get("lower_is_better", True))
        params.update(
            {
                "max_epochs": int(max_epochs),
                "min_epochs": int(train_cfg.get("min_epochs", 1)),
                "early_stopping_patience": int(train_cfg.get("early_stopping_patience", 10)),
                "early_stopping_min_delta": float(
                    train_cfg.get("early_stopping_min_delta", 0.0)
                ),
                "early_stopping_monitor": monitor,
                "early_stopping_lower_is_better": lower,
            }
        )

    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.log_artifact(str(cfg_path), artifact_path="config")

        try:
            if use_epochs:
                ck = train_cfg.get("best_checkpoint_path")
                if ck is None:
                    ck = f"checkpoints/{model_name}_best.pt"
                _run_epoch_training(
                    model=model,
                    opt=opt,
                    data_split_path=data_split_path,
                    device=device,
                    train_cfg=train_cfg,
                    ev_cfg=ev_cfg,
                    grad_accum=grad_accum,
                    batch_size=batch_size,
                    train_sub=train_sub,
                    point_seed_train=point_seed_train,
                    eval_seed=eval_seed,
                    eval_sub=eval_sub,
                    max_epochs=int(max_epochs),
                    min_epochs=int(train_cfg.get("min_epochs", 1)),
                    patience=int(train_cfg.get("early_stopping_patience", 10)),
                    min_delta=float(train_cfg.get("early_stopping_min_delta", 0.0)),
                    monitor=monitor,
                    lower_is_better=bool(lower),
                    best_ckpt_path=Path(ck),
                )
            else:
                max_train = args.max_train_steps or int(
                    train_cfg.get("max_train_steps", 100)
                )
                max_val = int(train_cfg.get("max_val_steps", 10))
                _run_legacy_step_training(
                    model=model,
                    opt=opt,
                    data_split_path=data_split_path,
                    device=device,
                    train_cfg=train_cfg,
                    ev_cfg=ev_cfg,
                    max_train=max_train,
                    max_val=max_val,
                    grad_accum=grad_accum,
                    batch_size=batch_size,
                    train_sub=train_sub,
                    point_seed_train=point_seed_train,
                    eval_seed=eval_seed,
                    eval_sub=eval_sub,
                )
        except Exception as e:
            print("Training failed:", e)
            print("Check HF_TOKEN / huggingface-cli login and configs/data_split.yaml id_key.")
            return 1

    print("Done. MLflow UI: mlflow ui --backend-store-uri ./mlruns")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
