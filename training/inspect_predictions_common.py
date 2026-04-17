"""Shared helpers for inspect CLI and Streamlit: load model, buffer batches, forward."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from models.registry import get_model_class
from training.device_utils import resolve_train_device
from training.hf_dataset import (
    Batch,
    SplitPhase,
    streaming_batches,
    subsample_batch_preforward,
)
from training.seeds import seed_all
from training.yaml_config import load_yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]


def resolve_repo_path(p: str | Path) -> Path:
    pp = Path(p)
    return pp.resolve() if pp.is_absolute() else (_REPO_ROOT / pp).resolve()


def positive_int_or_none(x: Any) -> int | None:
    if x is None:
        return None
    n = int(x)
    if n <= 0:
        return None
    return n


def load_model_from_checkpoint(
    training_config_path: Path,
    checkpoint_path: Path,
    device: torch.device,
) -> torch.nn.Module:
    cfg = load_yaml(training_config_path)
    train_cfg = cfg["train"]
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
    model = model.to(device)
    model.eval()
    return model


def collect_buffered_batches(
    *,
    training_config_path: Path,
    phase: SplitPhase,
    device: torch.device,
    max_batches: int,
    point_seed: int | None = None,
) -> list[Batch]:
    """
    Load up to ``max_batches`` batches with the same subsampling as training/eval
    (``train_subsample_N``, ``eval_preforward_subsample_N``).
    """
    cfg = load_yaml(training_config_path)
    paths = cfg["paths"]
    train_cfg = cfg["train"]
    data_split_path = resolve_repo_path(paths["data_split"])
    eval_path = resolve_repo_path(paths["eval_protocol"])
    ds_cfg = load_yaml(data_split_path)
    ev_cfg = load_yaml(eval_path)
    seed_all(int(ds_cfg["seed"]))

    batch_size = int(train_cfg.get("batch_size", 1))
    train_sub = positive_int_or_none(train_cfg.get("train_subsample_N"))
    eval_preforward = positive_int_or_none(train_cfg.get("eval_preforward_subsample_N"))
    if point_seed is None:
        point_seed = int(ev_cfg.get("eval_point_subsample_seed", 0))

    subsample_options: dict[str, Any] | None = None
    ps = train_cfg.get("point_subsample")
    if isinstance(ps, dict) and str(ps.get("mode", "")) == "stratified_lam_turb":
        subsample_options = {
            "mode": "stratified_lam_turb",
            "lam_fraction": float(ps.get("lam_fraction", 0.5)),
            "pool_quantile": float(ps.get("pool_quantile", 0.5)),
        }

    g: torch.Generator | None = None
    if eval_preforward is not None:
        g = torch.Generator(device="cpu")
        g.manual_seed(point_seed)

    out: list[Batch] = []
    it = streaming_batches(
        data_split_path,
        phase,
        device=device,
        batch_size=batch_size,
        train_subsample_N=train_sub,
        point_seed=point_seed,
        max_batches=max_batches,
        subsample_options=subsample_options,
    )
    for batch in it:
        if eval_preforward is not None:
            batch = subsample_batch_preforward(batch, eval_preforward, g)
        out.append(batch)
    return out


@torch.inference_mode()
def forward_predictions(model: torch.nn.Module, batch: Batch) -> torch.Tensor:
    return model(batch.t, batch.pos, batch.idcs_airfoil, batch.velocity_in)


def resolve_device(
    train_cfg: dict[str, Any],
    override: str | None,
) -> torch.device:
    if override is None or override == "auto":
        return resolve_train_device(train_cfg)
    return torch.device(override)


def load_train_config_only(path: Path) -> dict[str, Any]:
    return load_yaml(path)["train"]
