"""Full pass train/val over streaming splits (one epoch per call)."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.optim import Optimizer

from training.hf_dataset import streaming_batches
from training.metrics import l2_per_point_mean, mse_velocity, subsample_points


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
) -> tuple[float, int]:
    """One full pass over the train split. Returns (mean batch MSE, num_batches)."""
    model.train()
    opt.zero_grad(set_to_none=True)
    accum = 0
    loss_sum = 0.0
    n_batches = 0
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
    if accum > 0:
        opt.step()
        opt.zero_grad(set_to_none=True)
    return loss_sum / max(n_batches, 1), n_batches


@torch.no_grad()
def validate_full(
    *,
    model: torch.nn.Module,
    data_split_path: Path,
    device: torch.device,
    batch_size: int,
    eval_subsample_N: int | None,
    eval_seed: int,
) -> tuple[float, float, int]:
    """Full pass over val split. Returns (mean MSE, mean L2, num_batches)."""
    model.eval()
    g = torch.Generator()
    g.manual_seed(eval_seed)
    mse_acc = 0.0
    l2_acc = 0.0
    n = 0
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
        mse_acc += float(mse_velocity(p, tgt).cpu())
        l2_acc += float(l2_per_point_mean(p, tgt).cpu())
        n += 1
    return mse_acc / max(n, 1), l2_acc / max(n, 1), n


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
