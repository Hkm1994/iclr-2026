"""Trainable / submission model registry (name -> class)."""

from __future__ import annotations

from typing import Type

from torch.nn import Module

from models.mlp.model import MLP

MODELS: dict[str, Type[Module]] = {
    "mlp": MLP,
}


def get_model_class(name: str) -> Type[Module]:
    if name not in MODELS:
        raise KeyError(f"Unknown model {name!r}. Available: {sorted(MODELS)}")
    return MODELS[name]
