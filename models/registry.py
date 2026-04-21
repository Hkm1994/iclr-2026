"""Trainable / submission model registry (name -> class)."""

from __future__ import annotations

from typing import Type

from torch.nn import Module

from models.mlp.model import MLP
from models.strong_mlp.model import StrongMLP
from models.strong_mlp_knn.model import StrongMLPKnn
from models.strong_mlp_knn_mp.model import StrongMLPKnnMP
from models.strong_mlp_knn_mp_v2.model import StrongMLPKnnMPv2
from models.strong_mlp_knn_mp_v2_temporal.model import StrongMLPKnnMPv2Temporal
from models.tiny_linear.model import TinyLinearBaseline
from models.levers_tail_submission.model import LeversTailV2Submission

MODELS: dict[str, Type[Module]] = {
    "tiny_linear": TinyLinearBaseline,
    "mlp": MLP,
    "strong_mlp": StrongMLP,
    "strong_mlp_knn": StrongMLPKnn,
    "strong_mlp_knn_mp": StrongMLPKnnMP,
    "strong_mlp_knn_mp_v2": StrongMLPKnnMPv2,
    "strong_mlp_knn_mp_v2_temporal": StrongMLPKnnMPv2Temporal,
    "levers_tail_v2_submission": LeversTailV2Submission,
}


def get_model_class(name: str) -> Type[Module]:
    if name not in MODELS:
        raise KeyError(f"Unknown model {name!r}. Available: {sorted(MODELS)}")
    return MODELS[name]
