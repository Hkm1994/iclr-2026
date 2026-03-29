"""Shared test-split evaluation for checkpoints (used by eval_checkpoint and leaderboard)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from models.registry import get_model_class
from training.epoch_loop import evaluate_split_full
from training.memory_utils import release_training_memory
from training.mlflow_steps import new_stream_counters
from training.seeds import seed_all
from training.device_utils import resolve_train_device
from training.yaml_config import load_yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _resolve_repo_path(p: str | Path) -> Path:
    pp = Path(p)
    return pp.resolve() if pp.is_absolute() else (_REPO_ROOT / pp).resolve()


def _positive_int_or_none(x: Any) -> int | None:
    if x is None:
        return None
    n = int(x)
    if n <= 0:
        return None
    return n


def evaluate_checkpoint_on_test(
    *,
    training_config_path: Path,
    checkpoint_path: Path,
    device: torch.device | None = None,
    verbose: bool = True,
    log_every_n_batches: int | None = 5,
    heartbeat_seconds: float | None = None,
) -> dict[str, Any]:
    """
    Load ``checkpoint_path`` using architecture from training YAML, run full test split.

    Returns a flat dict including ``test/mse_velocity``, ``test/l2_per_point_mean``,
    ``n_test_batches``, ``model``, plus keys from ``evaluate_split_full``'s ``extra``
    (e.g. ``test/l2_timestep_*``).
    """
    cfg = load_yaml(training_config_path)
    paths = cfg["paths"]
    train_cfg = cfg["train"]

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    data_split_path = _resolve_repo_path(paths["data_split"])
    eval_path = _resolve_repo_path(paths["eval_protocol"])
    ds_cfg = load_yaml(data_split_path)
    ev_cfg = load_yaml(eval_path)
    seed_all(int(ds_cfg["seed"]))

    dev = device if device is not None else resolve_train_device(train_cfg)
    model_name = train_cfg["model"]
    model_cls = get_model_class(model_name)
    model_cfg: dict = {"skip_weights": True}
    extra_mc = train_cfg.get("model_config")
    if isinstance(extra_mc, dict):
        model_cfg.update(extra_mc)
    model_cfg["skip_weights"] = True
    model = model_cls(config=model_cfg)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model = model.to(dev)
    model.eval()

    batch_size = int(train_cfg.get("batch_size", 1))
    eval_sub = ev_cfg.get("eval_subsample_N")
    eval_seed = int(ev_cfg.get("eval_point_subsample_seed", 0))
    eval_preforward = _positive_int_or_none(train_cfg.get("eval_preforward_subsample_N"))
    le = log_every_n_batches
    if le is None:
        le = _positive_int_or_none(
            train_cfg.get(
                "log_every_n_val_batches",
                train_cfg.get("log_every_n_train_batches", 5),
            )
        )

    hb = heartbeat_seconds
    if hb is None:
        hb_raw = train_cfg.get("heartbeat_seconds")
        if hb_raw is not None:
            hb = float(hb_raw)
            if hb <= 0:
                hb = None

    stream_steps = new_stream_counters()
    try:
        test_mse, test_l2, n_te, test_lt = evaluate_split_full(
            model=model,
            data_split_path=data_split_path,
            device=dev,
            batch_size=batch_size,
            eval_subsample_N=eval_sub,
            eval_seed=eval_seed,
            phase="test",
            epoch_idx=0,
            log_every_n_batches=le,
            verbose=verbose,
            heartbeat_seconds=hb,
            eval_stream_step_counter=stream_steps.test_batch,
            run_label="leaderboard test eval",
            eval_preforward_subsample_N=eval_preforward,
        )
    finally:
        release_training_memory()

    out: dict[str, Any] = {
        "model": model_name,
        "training_config": str(training_config_path.resolve()),
        "checkpoint": str(checkpoint_path.resolve()),
        "n_test_batches": n_te,
        "test/mse_velocity": float(test_mse),
        "test/l2_per_point_mean": float(test_l2),
        "mlflow_step": int(stream_steps.test_batch[0]),
    }
    out.update({k: float(v) for k, v in test_lt.items()})
    return out
