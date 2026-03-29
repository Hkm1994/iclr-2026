"""Per-epoch learning-rate schedulers for scripts/train.py."""

from __future__ import annotations

from typing import Any

from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    ReduceLROnPlateau,
    SequentialLR,
)


def build_epoch_lr_scheduler(
    optimizer: Optimizer, train_cfg: dict[str, Any], max_epochs: int
) -> tuple[Any | None, str | None]:
    """
    Build a scheduler stepped once per epoch (after validation).

    Returns ``(scheduler, kind)``. ``kind`` is ``\"plateau\"``, ``\"cosine\"``,
    ``\"sequential\"`` (warmup + cosine), ``\"warmup_only\"``, or ``None``.
    """
    block = train_cfg.get("lr_schedule")
    warmup = int(train_cfg.get("lr_warmup_epochs", 0) or 0)

    if block is None or block is False:
        if warmup > 0:
            w_epochs = min(warmup, max_epochs)
            warm = LinearLR(
                optimizer,
                start_factor=1.0 / max(w_epochs, 1),
                end_factor=1.0,
                total_iters=max(1, w_epochs),
            )
            return warm, "warmup_only"
        return None, None

    typ = str(block.get("type", "cosine")).lower()
    if typ == "plateau":
        mode = str(block.get("mode", "min"))
        sch = ReduceLROnPlateau(
            optimizer,
            mode=mode,
            factor=float(block.get("factor", 0.5)),
            patience=int(block.get("patience", 3)),
            min_lr=float(block.get("min_lr", 0.0)),
            threshold=float(block.get("threshold", 1e-4)),
        )
        return sch, "plateau"

    if typ == "cosine":
        eta_min = float(block.get("eta_min", 1e-6))
        t_max = block.get("t_max_epochs")
        T = int(t_max) if t_max is not None else max_epochs
        if warmup > 0:
            w_epochs = min(warmup, max_epochs)
            warm = LinearLR(
                optimizer,
                start_factor=1.0 / max(w_epochs, 1),
                end_factor=1.0,
                total_iters=max(1, w_epochs),
            )
            rest = max(1, max_epochs - w_epochs)
            cos = CosineAnnealingLR(optimizer, T_max=rest, eta_min=eta_min)
            seq = SequentialLR(optimizer, [warm, cos], milestones=[w_epochs])
            return seq, "sequential"
        return CosineAnnealingLR(optimizer, T_max=max(1, T), eta_min=eta_min), "cosine"

    raise ValueError(f"Unknown lr_schedule.type {typ!r} (use cosine or plateau)")


def step_lr_scheduler(
    scheduler: Any,
    kind: str | None,
    *,
    monitor_value: float,
) -> None:
    if scheduler is None or kind is None:
        return
    if kind == "plateau":
        scheduler.step(monitor_value)
    else:
        scheduler.step()
