"""Atomic CPU state_dict saves + SIGINT/SIGTERM so interrupts still leave a resume file."""

from __future__ import annotations

import os
import signal
import sys
from pathlib import Path
from typing import Any, Callable

import torch
from torch import nn

from training.memory_utils import release_training_memory


def state_dict_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu() for k, v in model.state_dict().items()}


def save_state_dict_atomic(path: Path, model: nn.Module) -> None:
    """Write `model` weights on CPU; replace atomically so crashes mid-write do not corrupt."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state_dict_cpu(model), tmp)
    os.replace(tmp, path)


def _signal_name(signum: int) -> str:
    try:
        return signal.strsignal(signum) or str(signum)
    except Exception:
        return str(signum)


def register_interrupt_checkpoint(
    model: nn.Module,
    path: Path,
    *,
    verbose: bool = True,
) -> Callable[[], None]:
    """
    On SIGINT/SIGTERM, save weights to ``path`` then exit with 128+signum.

    Returns ``restore`` to reinstall previous handlers (call from ``finally``).
    """
    path = Path(path)
    prev: dict[str, Any] = {}

    def handler(signum: int, frame: Any) -> None:
        try:
            save_state_dict_atomic(path, model)
            if verbose:
                print(
                    f"\n[checkpoint] Interrupt ({_signal_name(signum)}): saved {path.resolve()}",
                    flush=True,
                )
        except Exception as e:
            print(
                f"\n[checkpoint] Could not save on interrupt: {e}",
                flush=True,
                file=sys.stderr,
            )
        finally:
            release_training_memory()
            signal.signal(signal.SIGINT, prev["int"])
            if prev.get("term") is not None:
                try:
                    signal.signal(signal.SIGTERM, prev["term"])
                except (AttributeError, ValueError):
                    pass
        raise SystemExit(128 + signum)

    prev["int"] = signal.signal(signal.SIGINT, handler)
    try:
        prev["term"] = signal.signal(signal.SIGTERM, handler)
    except (AttributeError, ValueError):
        prev["term"] = None

    def restore() -> None:
        signal.signal(signal.SIGINT, prev["int"])
        if prev.get("term") is not None:
            try:
                signal.signal(signal.SIGTERM, prev["term"])
            except (AttributeError, ValueError):
                pass

    return restore
