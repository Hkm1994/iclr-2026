"""Full pass train/val over streaming splits (one epoch per call)."""

from __future__ import annotations

import time
from pathlib import Path

import mlflow
import torch
from torch.optim import Optimizer

from training.hf_dataset import streaming_batches
from training.metrics import l2_per_point_mean, mse_velocity, subsample_points


def _mlflow_step_epoch_batch(epoch_idx: int, batch_in_epoch: int, *, val: bool = False) -> int:
    """Monotonic-ish step id: train uses [0,1M), val uses [1M,2M) per epoch."""
    base = epoch_idx * 2_000_000
    if val:
        return base + 1_000_000 + min(batch_in_epoch, 999_999)
    return base + min(batch_in_epoch, 999_999)


def train_one_epoch(
    *,
    model: torch.nn.Module,
    opt: Optimizer,
    data_split_path: Path,
    device: torch.device,
    batch_size: int,
    grad_accum: int,
    train_subsample_N: int | None,
    point_seed: int,
    epoch_idx: int = 0,
    log_every_n_batches: int | None = 5,
    verbose: bool = True,
    heartbeat_seconds: float | None = None,
) -> tuple[float, int]:
    """One full pass over the train split. Returns (mean batch MSE, num_batches)."""
    model.train()
    opt.zero_grad(set_to_none=True)
    accum = 0
    loss_sum = 0.0
    n_batches = 0
    hb_last = time.monotonic()
    it = streaming_batches(
        data_split_path,
        "train",
        device=device,
        batch_size=batch_size,
        train_subsample_N=train_subsample_N,
        point_seed=point_seed,
        max_batches=None,
    )
    for batch in it:
        pred = model(batch.t, batch.pos, batch.idcs_airfoil, batch.velocity_in)
        loss = mse_velocity(pred, batch.velocity_out)
        (loss / grad_accum).backward()
        accum += 1
        loss_sum += float(loss.detach().cpu())
        n_batches += 1
        if accum >= grad_accum:
            opt.step()
            opt.zero_grad(set_to_none=True)
            accum = 0

        if log_every_n_batches and (
            n_batches == 1 or n_batches % log_every_n_batches == 0
        ):
            last_mse = float(loss.detach().cpu())
            partial_mean = loss_sum / n_batches
            step = _mlflow_step_epoch_batch(epoch_idx, n_batches, val=False)
            mlflow.log_metric("train/mse_velocity_batch_last", last_mse, step=step)
            mlflow.log_metric(
                "train/mse_velocity_epoch_mean_partial", partial_mean, step=step
            )
            if verbose:
                n_pts = batch.pos.shape[1]
                print(
                    f"  [train] epoch {epoch_idx + 1} batch {n_batches} | "
                    f"last_mse={last_mse:.6f} running_mean_mse={partial_mean:.6f} | "
                    f"batch_points={n_pts} device={device}",
                    flush=True,
                )

        if heartbeat_seconds is not None and heartbeat_seconds > 0:
            now = time.monotonic()
            if now - hb_last >= heartbeat_seconds:
                if verbose:
                    print(
                        f"  [heartbeat] train epoch {epoch_idx + 1} | batch {n_batches} | "
                        f"last_mse={float(loss.detach().cpu()):.6f} (still running…)",
                        flush=True,
                    )
                hb_last = now

    if accum > 0:
        opt.step()
        opt.zero_grad(set_to_none=True)

    mean_loss = loss_sum / max(n_batches, 1)
    if verbose:
        print(
            f"  [train] epoch {epoch_idx + 1} finished | batches={n_batches} "
            f"mean_mse={mean_loss:.6f}",
            flush=True,
        )
    return mean_loss, n_batches


@torch.no_grad()
def validate_full(
    *,
    model: torch.nn.Module,
    data_split_path: Path,
    device: torch.device,
    batch_size: int,
    eval_subsample_N: int | None,
    eval_seed: int,
    epoch_idx: int = 0,
    log_every_n_batches: int | None = 5,
    verbose: bool = True,
    heartbeat_seconds: float | None = None,
) -> tuple[float, float, int]:
    """Full pass over val split. Returns (mean MSE, mean L2, num_batches)."""
    model.eval()
    g = torch.Generator()
    g.manual_seed(eval_seed)
    mse_acc = 0.0
    l2_acc = 0.0
    n = 0
    hb_last = time.monotonic()
    it = streaming_batches(
        data_split_path,
        "val",
        device=device,
        batch_size=batch_size,
        train_subsample_N=None,
        point_seed=eval_seed,
        max_batches=None,
    )
    for batch in it:
        pred = model(batch.t, batch.pos, batch.idcs_airfoil, batch.velocity_in)
        p, tgt = subsample_points(
            pred, batch.velocity_out, eval_subsample_N, generator=g
        )
        mse_b = float(mse_velocity(p, tgt).cpu())
        l2_b = float(l2_per_point_mean(p, tgt).cpu())
        mse_acc += mse_b
        l2_acc += l2_b
        n += 1

        if log_every_n_batches and (n == 1 or n % log_every_n_batches == 0):
            step = _mlflow_step_epoch_batch(epoch_idx, n, val=True)
            mlflow.log_metric(
                "val/mse_velocity_partial_mean",
                mse_acc / n,
                step=step,
            )
            mlflow.log_metric(
                "val/l2_per_point_mean_partial_mean",
                l2_acc / n,
                step=step,
            )
            if verbose:
                print(
                    f"  [val]   epoch {epoch_idx + 1} batch {n} | "
                    f"batch_mse={mse_b:.6f} batch_l2={l2_b:.6f} | "
                    f"running_mean_mse={mse_acc / n:.6f} running_mean_l2={l2_acc / n:.6f}",
                    flush=True,
                )

        if heartbeat_seconds is not None and heartbeat_seconds > 0:
            now = time.monotonic()
            if now - hb_last >= heartbeat_seconds:
                if verbose:
                    print(
                        f"  [heartbeat] val epoch {epoch_idx + 1} | batch {n} | "
                        f"running_mean_l2={l2_acc / n:.6f} (still running…)",
                        flush=True,
                    )
                hb_last = now

    mean_mse = mse_acc / max(n, 1)
    mean_l2 = l2_acc / max(n, 1)
    if verbose and n > 0:
        print(
            f"  [val]   epoch {epoch_idx + 1} finished | batches={n} "
            f"mean_mse={mean_mse:.6f} mean_l2={mean_l2:.6f}",
            flush=True,
        )
    return mean_mse, mean_l2, n


def is_better(
    value: float,
    best: float,
    *,
    min_delta: float,
    lower_is_better: bool,
) -> bool:
    if lower_is_better:
        return value < best - min_delta
    return value > best + min_delta
