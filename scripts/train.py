#!/usr/bin/env python3
"""Train with HF streaming, MLflow logging, central data_split + eval_protocol."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import mlflow
import torch
from torch.optim import AdamW

from models.registry import get_model_class
from training.hf_dataset import streaming_batches
from training.metrics import l2_per_point_mean, mse_velocity, subsample_points
from training.seeds import seed_all
from training.yaml_config import load_yaml


def _device_from_cfg(train_cfg: dict) -> torch.device:
    d = train_cfg.get("device")
    if d:
        return torch.device(d)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    max_train = args.max_train_steps or int(train_cfg.get("max_train_steps", 100))
    max_val = int(train_cfg.get("max_val_steps", 10))
    point_seed_train = int(train_cfg.get("train_point_seed", 0))

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns"))
    mlflow.set_experiment(exp_cfg.get("mlflow_experiment_name", "gram-warped-ifw"))

    data_split_version = str(ds_cfg.get("version", "unknown"))
    eval_version = str(ev_cfg.get("version", "unknown"))

    with mlflow.start_run():
        mlflow.log_params(
            {
                "model_family": train_cfg.get("model_family", model_name),
                "model": model_name,
                "data_split_version": data_split_version,
                "eval_protocol_version": eval_version,
                "seed": master_seed,
                "lr": lr,
                "weight_decay": wd,
                "batch_size": batch_size,
                "train_subsample_N": train_sub if train_sub is not None else "full",
                "eval_subsample_N": ev_cfg.get("eval_subsample_N") or "full",
                "config_file": str(cfg_path),
            }
        )
        mlflow.log_artifact(str(cfg_path), artifact_path="config")

        step = 0
        accum = 0
        opt.zero_grad(set_to_none=True)

        try:
            train_it = streaming_batches(
                data_split_path,
                "train",
                device=device,
                batch_size=batch_size,
                train_subsample_N=train_sub,
                point_seed=point_seed_train,
                max_batches=None,
            )
        except Exception as e:
            print("Failed to build training stream:", e)
            print("Check HF_TOKEN / huggingface-cli login and configs/data_split.yaml id_key.")
            return 1

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

        eval_sub = ev_cfg.get("eval_subsample_N")
        eval_seed = int(ev_cfg.get("eval_point_subsample_seed", 0))
        g = torch.Generator()
        g.manual_seed(eval_seed)

        val_step = 0
        val_mse_acc = 0.0
        val_l2_acc = 0.0
        val_count = 0

        try:
            val_it = streaming_batches(
                data_split_path,
                "val",
                device=device,
                batch_size=batch_size,
                train_subsample_N=None,
                point_seed=eval_seed,
                max_batches=max_val,
            )
        except Exception as e:
            print("Val stream failed:", e)
            val_it = iter(())

        with torch.no_grad():
            for batch in val_it:
                pred = model(batch.t, batch.pos, batch.idcs_airfoil, batch.velocity_in)
                p, tgt = subsample_points(
                    pred, batch.velocity_out, eval_sub, generator=g
                )
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

    print("Done. MLflow UI: mlflow ui --backend-store-uri ./mlruns")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
