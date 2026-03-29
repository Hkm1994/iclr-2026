"""Full pass train/val over streaming splits (one epoch per call)."""

from __future__ import annotations

import time
from pathlib import Path

import mlflow
import torch
from torch.optim import Optimizer

from training.hf_dataset import SplitPhase, streaming_batches, subsample_batch_preforward
from training.memory_utils import release_training_memory
from training.ema import ModelEMA
from training.metrics import (
    l2_per_point_mean,
    l2_per_timestep_mean,
    l2_per_timestep_mean_masked,
    lam_turb_mask_for_eval_subset,
    mse_l2_lam_turb_on_subset,
    mse_per_timestep_mean,
    mse_per_timestep_mean_masked,
    mse_velocity,
    mse_velocity_train_weighted,
    subsample_points,
)


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
    train_stream_step_counter: list[int],
    epoch_idx: int = 0,
    log_every_n_batches: int | None = 5,
    verbose: bool = True,
    heartbeat_seconds: float | None = None,
    subsample_options: dict | None = None,
    loss_turb_weight_alpha: float = 0.0,
    loss_timestep_weights: torch.Tensor | None = None,
    ema: ModelEMA | None = None,
) -> tuple[float, int]:
    """
    One full pass over the train split.

    ``train_stream_step_counter`` is a length-1 list incremented once per batch;
    it is the MLflow ``step`` for ``stream/train_*`` metrics only.
    """
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
        subsample_options=subsample_options,
    )
    try:
        for batch in it:
            pred = model(batch.t, batch.pos, batch.idcs_airfoil, batch.velocity_in)
            loss = mse_velocity_train_weighted(
                pred,
                batch.velocity_out,
                batch.velocity_in,
                turb_alpha=loss_turb_weight_alpha,
                timestep_weights=loss_timestep_weights,
            )
            (loss / grad_accum).backward()
            accum += 1
            loss_sum += float(loss.detach().cpu())
            n_batches += 1
            if accum >= grad_accum:
                opt.step()
                opt.zero_grad(set_to_none=True)
                if ema is not None:
                    ema.update(model)
                accum = 0

            train_stream_step_counter[0] += 1
            step = train_stream_step_counter[0]

            if log_every_n_batches and (
                n_batches == 1 or n_batches % log_every_n_batches == 0
            ):
                last_mse = float(loss.detach().cpu())
                partial_mean = loss_sum / n_batches
                mlflow.log_metric("stream/train_last_mse", last_mse, step=step)
                mlflow.log_metric(
                    "stream/train_running_mean_mse", partial_mean, step=step
                )
                if batch.lam_point_mask is not None:
                    m_lam, m_turb, _, l_lam, l_turb = mse_l2_lam_turb_on_subset(
                        pred.detach(),
                        batch.velocity_out.detach(),
                        batch.lam_point_mask,
                    )
                    if m_lam is not None:
                        mlflow.log_metric(
                            "stream/train_mse_lam_proxy",
                            float(m_lam.cpu()),
                            step=step,
                        )
                    if m_turb is not None:
                        mlflow.log_metric(
                            "stream/train_mse_turb_proxy",
                            float(m_turb.cpu()),
                            step=step,
                        )
                    if l_lam is not None:
                        mlflow.log_metric(
                            "stream/train_l2_lam_proxy",
                            float(l_lam.cpu()),
                            step=step,
                        )
                    if l_turb is not None:
                        mlflow.log_metric(
                            "stream/train_l2_turb_proxy",
                            float(l_turb.cpu()),
                            step=step,
                        )
                if verbose:
                    n_pts = batch.pos.shape[1]
                    print(
                        f"  [train] epoch {epoch_idx + 1} batch {n_batches} | "
                        f"last_mse={last_mse:.6f} running_mean_mse={partial_mean:.6f} | "
                        f"batch_points={n_pts} device={device} | train_stream_step={step}",
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
            if ema is not None:
                ema.update(model)

        mean_loss = loss_sum / max(n_batches, 1)
        if verbose:
            print(
                f"  [train] epoch {epoch_idx + 1} finished | batches={n_batches} "
                f"mean_mse={mean_loss:.6f}",
                flush=True,
            )
        return mean_loss, n_batches
    finally:
        release_training_memory()


@torch.no_grad()
def evaluate_split_full(
    *,
    model: torch.nn.Module,
    data_split_path: Path,
    device: torch.device,
    batch_size: int,
    eval_subsample_N: int | None,
    eval_seed: int,
    phase: SplitPhase,
    epoch_idx: int = 0,
    log_every_n_batches: int | None = 5,
    verbose: bool = True,
    heartbeat_seconds: float | None = None,
    eval_stream_step_counter: list[int],
    run_label: str | None = None,
    eval_preforward_subsample_N: int | None = None,
) -> tuple[float, float, int, dict[str, float]]:
    """
    Full pass over ``phase`` (``val`` or ``test``). MLflow partial metrics use
    ``stream/val_*`` or ``stream/test_*`` with ``eval_stream_step_counter`` as ``step``
    (independent from training and from epoch-level metrics).

    When ``eval_subsample_N`` is set and ``batch_size == 1``, the fourth return value
    contains epoch-mean proxy KPIs ``{tag}/mse_lam_proxy``, ``{tag}/mse_turb_proxy``,
    and L2 counterparts (median split on temporal fluctuation within the subsample).

    Always (when at least one batch): per-output-timestep means
    ``{tag}/l2_timestep_{i}_mean`` and ``{tag}/mse_timestep_{i}_mean``.
    With lam/turb KPIs, also ``{tag}/l2_lam_timestep_*``, ``{tag}/mse_lam_timestep_*``,
    and turb counterparts averaged over batches where the respective mask is non-empty.

    If ``eval_preforward_subsample_N`` is set, points are subsampled (same RNG stream as
    metric subsampling) before ``forward``, so kNN-style models see the same scale as
    training-time ``train_subsample_N``.
    """
    tag = "val" if phase == "val" else "test"
    label = run_label or f"epoch {epoch_idx + 1}"
    model.eval()
    g = torch.Generator()
    g.manual_seed(eval_seed)
    mse_acc = 0.0
    l2_acc = 0.0
    n = 0
    lam_turb_kpis = eval_subsample_N is not None and batch_size == 1
    lt_mse_lam_sum = lt_mse_turb_sum = 0.0
    lt_l2_lam_sum = lt_l2_turb_sum = 0.0
    n_lt_mse_lam = n_lt_mse_turb = n_lt_l2_lam = n_lt_l2_turb = 0
    ts_l2_sum: torch.Tensor | None = None
    ts_mse_sum: torch.Tensor | None = None
    ts_l2_lam_sum: torch.Tensor | None = None
    ts_l2_turb_sum: torch.Tensor | None = None
    ts_mse_lam_sum: torch.Tensor | None = None
    ts_mse_turb_sum: torch.Tensor | None = None
    n_batch_lam_ts = 0
    n_batch_turb_ts = 0
    hb_last = time.monotonic()
    it = streaming_batches(
        data_split_path,
        phase,
        device=device,
        batch_size=batch_size,
        train_subsample_N=None,
        point_seed=eval_seed,
        max_batches=None,
    )
    mse_key = f"stream/{tag}_running_mean_mse"
    l2_key = f"stream/{tag}_running_mean_l2"

    try:
        for batch in it:
            if eval_preforward_subsample_N is not None:
                batch = subsample_batch_preforward(
                    batch, eval_preforward_subsample_N, g
                )
            pred = model(batch.t, batch.pos, batch.idcs_airfoil, batch.velocity_in)
            if lam_turb_kpis:
                p, tgt, idx = subsample_points(
                    pred,
                    batch.velocity_out,
                    eval_subsample_N,
                    generator=g,
                    return_indices=True,
                )
                lam_m = lam_turb_mask_for_eval_subset(batch.velocity_in, idx)
                m_lam, m_turb, _, l_lam, l_turb = mse_l2_lam_turb_on_subset(
                    p, tgt, lam_m
                )
                if m_lam is not None:
                    lt_mse_lam_sum += float(m_lam.cpu())
                    n_lt_mse_lam += 1
                if m_turb is not None:
                    lt_mse_turb_sum += float(m_turb.cpu())
                    n_lt_mse_turb += 1
                if l_lam is not None:
                    lt_l2_lam_sum += float(l_lam.cpu())
                    n_lt_l2_lam += 1
                if l_turb is not None:
                    lt_l2_turb_sum += float(l_turb.cpu())
                    n_lt_l2_turb += 1
            else:
                p, tgt = subsample_points(
                    pred, batch.velocity_out, eval_subsample_N, generator=g
                )
                lam_m = None
            mse_b = float(mse_velocity(p, tgt).cpu())
            l2_b = float(l2_per_point_mean(p, tgt).cpu())
            mse_acc += mse_b
            l2_acc += l2_b
            n += 1

            lt2 = l2_per_timestep_mean(p, tgt).detach().cpu()
            mt2 = mse_per_timestep_mean(p, tgt).detach().cpu()
            if ts_l2_sum is None:
                ts_l2_sum = lt2.clone()
                ts_mse_sum = mt2.clone()
            else:
                ts_l2_sum += lt2
                ts_mse_sum += mt2
            if lam_turb_kpis and lam_m is not None:
                if lam_m.any():
                    if ts_l2_lam_sum is None:
                        ts_l2_lam_sum = torch.zeros_like(lt2)
                        ts_mse_lam_sum = torch.zeros_like(mt2)
                    ts_l2_lam_sum += l2_per_timestep_mean_masked(
                        p, tgt, lam_m
                    ).detach().cpu()
                    ts_mse_lam_sum += mse_per_timestep_mean_masked(
                        p, tgt, lam_m
                    ).detach().cpu()
                    n_batch_lam_ts += 1
                turb_m = ~lam_m
                if turb_m.any():
                    if ts_l2_turb_sum is None:
                        ts_l2_turb_sum = torch.zeros_like(lt2)
                        ts_mse_turb_sum = torch.zeros_like(mt2)
                    ts_l2_turb_sum += l2_per_timestep_mean_masked(
                        p, tgt, turb_m
                    ).detach().cpu()
                    ts_mse_turb_sum += mse_per_timestep_mean_masked(
                        p, tgt, turb_m
                    ).detach().cpu()
                    n_batch_turb_ts += 1

            eval_stream_step_counter[0] += 1
            step = eval_stream_step_counter[0]

            if log_every_n_batches and (n == 1 or n % log_every_n_batches == 0):
                mlflow.log_metric(mse_key, mse_acc / n, step=step)
                mlflow.log_metric(l2_key, l2_acc / n, step=step)
                if lam_turb_kpis:
                    if n_lt_mse_lam:
                        mlflow.log_metric(
                            f"stream/{tag}_running_mean_mse_lam_proxy",
                            lt_mse_lam_sum / n_lt_mse_lam,
                            step=step,
                        )
                    if n_lt_mse_turb:
                        mlflow.log_metric(
                            f"stream/{tag}_running_mean_mse_turb_proxy",
                            lt_mse_turb_sum / n_lt_mse_turb,
                            step=step,
                        )
                    if n_lt_l2_lam:
                        mlflow.log_metric(
                            f"stream/{tag}_running_mean_l2_lam_proxy",
                            lt_l2_lam_sum / n_lt_l2_lam,
                            step=step,
                        )
                    if n_lt_l2_turb:
                        mlflow.log_metric(
                            f"stream/{tag}_running_mean_l2_turb_proxy",
                            lt_l2_turb_sum / n_lt_l2_turb,
                            step=step,
                        )
                if verbose:
                    print(
                        f"  [{tag}] {label} batch {n} | "
                        f"batch_mse={mse_b:.6f} batch_l2={l2_b:.6f} | "
                        f"running_mean_mse={mse_acc / n:.6f} running_mean_l2={l2_acc / n:.6f} | "
                        f"{tag}_stream_step={step}",
                        flush=True,
                    )

            if heartbeat_seconds is not None and heartbeat_seconds > 0:
                now = time.monotonic()
                if now - hb_last >= heartbeat_seconds:
                    if verbose:
                        print(
                            f"  [heartbeat] {tag} {label} | batch {n} | "
                            f"running_mean_l2={l2_acc / max(n, 1):.6f} (still running…)",
                            flush=True,
                        )
                    hb_last = now

        mean_mse = mse_acc / max(n, 1)
        mean_l2 = l2_acc / max(n, 1)
        if verbose and n > 0:
            print(
                f"  [{tag}] {label} finished | batches={n} "
                f"mean_mse={mean_mse:.6f} mean_l2={mean_l2:.6f}",
                flush=True,
            )
        extra: dict[str, float] = {}
        if lam_turb_kpis and n > 0:
            if n_lt_mse_lam:
                extra[f"{tag}/mse_lam_proxy"] = lt_mse_lam_sum / n_lt_mse_lam
            if n_lt_mse_turb:
                extra[f"{tag}/mse_turb_proxy"] = lt_mse_turb_sum / n_lt_mse_turb
            if n_lt_l2_lam:
                extra[f"{tag}/l2_lam_proxy"] = lt_l2_lam_sum / n_lt_l2_lam
            if n_lt_l2_turb:
                extra[f"{tag}/l2_turb_proxy"] = lt_l2_turb_sum / n_lt_l2_turb
        if ts_l2_sum is not None and n > 0:
            tn = int(ts_l2_sum.shape[0])
            for ti in range(tn):
                extra[f"{tag}/l2_timestep_{ti}_mean"] = float(ts_l2_sum[ti] / n)
                extra[f"{tag}/mse_timestep_{ti}_mean"] = float(ts_mse_sum[ti] / n)
        if ts_l2_lam_sum is not None and n_batch_lam_ts > 0:
            tn = int(ts_l2_lam_sum.shape[0])
            for ti in range(tn):
                extra[f"{tag}/l2_lam_timestep_{ti}_mean"] = float(
                    ts_l2_lam_sum[ti] / n_batch_lam_ts
                )
                extra[f"{tag}/mse_lam_timestep_{ti}_mean"] = float(
                    ts_mse_lam_sum[ti] / n_batch_lam_ts
                )
        if ts_l2_turb_sum is not None and n_batch_turb_ts > 0:
            tn = int(ts_l2_turb_sum.shape[0])
            for ti in range(tn):
                extra[f"{tag}/l2_turb_timestep_{ti}_mean"] = float(
                    ts_l2_turb_sum[ti] / n_batch_turb_ts
                )
                extra[f"{tag}/mse_turb_timestep_{ti}_mean"] = float(
                    ts_mse_turb_sum[ti] / n_batch_turb_ts
                )
        return mean_mse, mean_l2, n, extra
    finally:
        release_training_memory()


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
    eval_stream_step_counter: list[int],
    run_label: str | None = None,
    eval_preforward_subsample_N: int | None = None,
) -> tuple[float, float, int, dict[str, float]]:
    """Backward-compatible alias: evaluate validation split."""
    return evaluate_split_full(
        model=model,
        data_split_path=data_split_path,
        device=device,
        batch_size=batch_size,
        eval_subsample_N=eval_subsample_N,
        eval_seed=eval_seed,
        phase="val",
        epoch_idx=epoch_idx,
        log_every_n_batches=log_every_n_batches,
        verbose=verbose,
        heartbeat_seconds=heartbeat_seconds,
        eval_stream_step_counter=eval_stream_step_counter,
        run_label=run_label,
        eval_preforward_subsample_N=eval_preforward_subsample_N,
    )


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
