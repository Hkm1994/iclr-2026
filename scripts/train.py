#!/usr/bin/env python3
"""Train with HF streaming, MLflow logging, central data_split + eval_protocol."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from training.hf_progress import silence_hf_download_progress

silence_hf_download_progress()

import mlflow
import torch
from torch.optim import AdamW

from models.registry import get_model_class
from training.epoch_loop import is_better, train_one_epoch, validate_full
from training.hf_dataset import streaming_batches
from training.metrics import l2_per_point_mean, mse_velocity, subsample_points
from training.seeds import seed_all
from training.train_checkpointing import (
    register_interrupt_checkpoint,
    save_state_dict_atomic,
)
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
    verbose: bool,
    log_every_n_train_batches: int | None,
    log_mlflow_train_every_n: int | None,
    last_ckpt_path: Path,
    heartbeat_seconds: float | None,
) -> None:
    step = 0
    accum = 0
    opt.zero_grad(set_to_none=True)
    mlflow_train = log_mlflow_train_every_n
    if mlflow_train is None:
        mlflow_train = 1
    hb_last = time.monotonic()

    if verbose:
        print(
            f"[legacy train] max_train_steps={max_train} batch_size={batch_size} "
            f"train_subsample_N={train_sub!s}",
            flush=True,
        )

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
        loss_f = float(loss.detach().cpu())
        if step == 0 or (step + 1) % mlflow_train == 0:
            mlflow.log_metric("train/mse_velocity", loss_f, step=step)
        if verbose and log_every_n_train_batches is not None:
            if step == 0 or (step + 1) % log_every_n_train_batches == 0:
                print(
                    f"  [train] step {step + 1}/{max_train} mse={loss_f:.6f} "
                    f"points={batch.pos.shape[1]}",
                    flush=True,
                )
        step += 1
        if heartbeat_seconds is not None and heartbeat_seconds > 0:
            now = time.monotonic()
            if now - hb_last >= heartbeat_seconds:
                if verbose:
                    print(
                        f"  [heartbeat] legacy train step {step} mse={loss_f:.6f} (still running…)",
                        flush=True,
                    )
                hb_last = now
        if step >= max_train:
            break

    if accum > 0:
        opt.step()
        opt.zero_grad(set_to_none=True)

    if verbose:
        print(f"[legacy train] finished {step} optimizer steps", flush=True)

    g = torch.Generator()
    g.manual_seed(eval_seed)
    val_step = 0
    val_mse_acc = 0.0
    val_l2_acc = 0.0
    val_count = 0
    hb_last = time.monotonic()
    val_it = streaming_batches(
        data_split_path,
        "val",
        device=device,
        batch_size=batch_size,
        train_subsample_N=None,
        point_seed=eval_seed,
        max_batches=max_val,
    )
    if verbose:
        print(f"[legacy val] max_val_steps={max_val}", flush=True)
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
            if verbose and log_every_n_train_batches and (
                val_count == 1 or val_count % log_every_n_train_batches == 0
            ):
                print(
                    f"  [val] batch {val_count}/{max_val} mse={float(vm.cpu()):.6f} "
                    f"l2={float(vl.cpu()):.6f}",
                    flush=True,
                )
            if heartbeat_seconds is not None and heartbeat_seconds > 0:
                now = time.monotonic()
                if now - hb_last >= heartbeat_seconds:
                    if verbose:
                        print(
                            f"  [heartbeat] legacy val batch {val_count} (still running…)",
                            flush=True,
                        )
                    hb_last = now
            if val_step >= max_val:
                break

    if val_count > 0:
        mlflow.log_metric("val/mse_velocity", val_mse_acc / val_count)
        mlflow.log_metric("val/l2_per_point_mean", val_l2_acc / val_count)
        if verbose:
            print(
                f"[legacy val] mean mse={val_mse_acc / val_count:.6f} "
                f"mean l2={val_l2_acc / val_count:.6f}",
                flush=True,
            )
    save_state_dict_atomic(last_ckpt_path, model)


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
    verbose: bool,
    log_every_n_train_batches: int | None,
    log_every_n_val_batches: int | None,
    last_ckpt_path: Path,
    heartbeat_seconds: float | None,
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

    if verbose:
        print(
            f"\n=== Epoch training: max_epochs={max_epochs} min_epochs={min_epochs} "
            f"patience={patience} monitor={monitor} ===\n",
            flush=True,
        )

    for epoch in range(max_epochs):
        seed_ep = int(point_seed_train) + epoch * 1_000_003
        if verbose:
            print(
                f"\n--- Epoch {epoch + 1}/{max_epochs} | train (seed_ep={seed_ep}) ---",
                flush=True,
            )
        train_loss, n_tr = train_one_epoch(
            model=model,
            opt=opt,
            data_split_path=data_split_path,
            device=device,
            batch_size=batch_size,
            grad_accum=grad_accum,
            train_subsample_N=train_sub,
            point_seed=seed_ep,
            epoch_idx=epoch,
            log_every_n_batches=log_every_n_train_batches,
            verbose=verbose,
            heartbeat_seconds=heartbeat_seconds,
        )
        if verbose:
            print(
                f"--- Epoch {epoch + 1}/{max_epochs} | validation ---",
                flush=True,
            )
        val_mse, val_l2, n_val = validate_full(
            model=model,
            data_split_path=data_split_path,
            device=device,
            batch_size=batch_size,
            eval_subsample_N=eval_sub,
            eval_seed=eval_seed,
            epoch_idx=epoch,
            log_every_n_batches=log_every_n_val_batches,
            verbose=verbose,
            heartbeat_seconds=heartbeat_seconds,
        )
        if n_val == 0:
            print("Validation produced zero batches; check data_split and HF access.")
            save_state_dict_atomic(last_ckpt_path, model)
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
            save_state_dict_atomic(best_ckpt_path, model)
            mlflow.log_metric("val/best_" + monitor.replace("/", "_"), best, step=epoch)
        else:
            epochs_without_improve += 1

        if verbose:
            imp = "improved" if epochs_without_improve == 0 else f"no_improve_{epochs_without_improve}"
            print(
                f"--- Epoch {epoch + 1} summary | train_mean_mse={train_loss:.6f} "
                f"val_mse={val_mse:.6f} val_l2={val_l2:.6f} | "
                f"best_{monitor}={best:.6f} ({imp}) | batches train/val={n_tr}/{n_val} ---\n",
                flush=True,
            )

        save_state_dict_atomic(last_ckpt_path, model)

        if epoch + 1 >= min_epochs and epochs_without_improve >= patience:
            mlflow.log_param("stopped_epoch", epoch + 1)
            mlflow.log_param("stop_reason", "early_stopping")
            if verbose:
                print(
                    f"Early stopping: no improvement for {patience} epochs "
                    f"(after min_epochs={min_epochs}).",
                    flush=True,
                )
            break
    else:
        mlflow.log_param("stopped_epoch", max_epochs)
        mlflow.log_param("stop_reason", "max_epochs")
        if verbose:
            print(f"Stopped: reached max_epochs={max_epochs}.", flush=True)

    mlflow.log_param("best_" + monitor.replace("/", "_"), best)
    if best_ckpt_path.is_file():
        mlflow.log_artifact(str(best_ckpt_path), artifact_path="checkpoints")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/example_mlp.yaml")
    ap.add_argument("--max-train-steps", type=int, default=None)
    ap.add_argument(
        "--quiet",
        action="store_true",
        help="Disable verbose terminal progress (MLflow batch metrics still logged unless disabled in YAML).",
    )
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

    verbose = bool(train_cfg.get("verbose", True)) and not args.quiet
    log_every_train = _positive_int_or_none(
        train_cfg.get("log_every_n_train_batches", 5)
    )
    log_every_val = _positive_int_or_none(
        train_cfg.get(
            "log_every_n_val_batches",
            train_cfg.get("log_every_n_train_batches", 5),
        )
    )
    log_mlflow_train_every = _positive_int_or_none(
        train_cfg.get("log_mlflow_train_every_n_steps", 1)
    )

    last_ckpt = Path(
        train_cfg.get("last_checkpoint_path") or f"checkpoints/{model_name}_last.pt"
    )
    hb_raw = train_cfg.get("heartbeat_seconds")
    if hb_raw is None:
        heartbeat_seconds: float | None = None
    else:
        heartbeat_seconds = float(hb_raw)
        if heartbeat_seconds <= 0:
            heartbeat_seconds = None

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
    params["verbose_terminal"] = verbose
    params["log_every_n_train_batches"] = log_every_train or "off"
    params["log_every_n_val_batches"] = log_every_val or "off"
    params["last_checkpoint_path"] = str(last_ckpt)
    params["heartbeat_seconds"] = (
        heartbeat_seconds if heartbeat_seconds is not None else "off"
    )
    if not use_epochs:
        params["log_mlflow_train_every_n_steps"] = log_mlflow_train_every or "off"

    restore_sig = register_interrupt_checkpoint(model, last_ckpt, verbose=verbose)
    try:
        with mlflow.start_run():
            mlflow.log_params(params)
            mlflow.log_artifact(str(cfg_path), artifact_path="config")

            if verbose:
                print(
                    f"MLflow experiment={exp_cfg.get('mlflow_experiment_name', 'gram-warped-ifw')} "
                    f"tracking_uri={os.environ.get('MLFLOW_TRACKING_URI', 'file:./mlruns')}",
                    flush=True,
                )
                print(
                    f"Model={model_name} device={device} | data_split={data_split_path} | "
                    f"eval_protocol={eval_path}",
                    flush=True,
                )
                if use_epochs:
                    print(
                        f"Logging: train batch metrics every {log_every_train or 'never'} batches, "
                        f"val every {log_every_val or 'never'} batches.",
                        flush=True,
                    )
                else:
                    print(
                        f"Logging: terminal every {log_every_train or 'never'} steps; "
                        f"MLflow train metric every {log_mlflow_train_every or 'never'} steps.",
                        flush=True,
                    )
                print(
                    f"Checkpoints: last={last_ckpt} (every epoch end + on interrupt/error); "
                    f"best=see config best_checkpoint_path (epoch mode). "
                    f"Heartbeat: {heartbeat_seconds or 'off'} s.",
                    flush=True,
                )

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
                        verbose=verbose,
                        log_every_n_train_batches=log_every_train,
                        log_every_n_val_batches=log_every_val,
                        last_ckpt_path=last_ckpt,
                        heartbeat_seconds=heartbeat_seconds,
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
                        verbose=verbose,
                        log_every_n_train_batches=log_every_train,
                        log_mlflow_train_every_n=log_mlflow_train_every,
                        last_ckpt_path=last_ckpt,
                        heartbeat_seconds=heartbeat_seconds,
                    )
            except KeyboardInterrupt:
                save_state_dict_atomic(last_ckpt, model)
                if verbose:
                    print(
                        f"[checkpoint] Saved to {last_ckpt} (KeyboardInterrupt)\n",
                        flush=True,
                    )
                raise
            except Exception as e:
                save_state_dict_atomic(last_ckpt, model)
                if verbose:
                    print(
                        f"[checkpoint] Saved to {last_ckpt} after error.\n",
                        flush=True,
                    )
                print("Training failed:", e)
                print(
                    "Check HF_TOKEN / huggingface-cli login and configs/data_split.yaml id_key."
                )
                return 1

            print("Done. MLflow UI: mlflow ui --backend-store-uri ./mlruns")
            return 0
    finally:
        restore_sig()


if __name__ == "__main__":
    raise SystemExit(main())
