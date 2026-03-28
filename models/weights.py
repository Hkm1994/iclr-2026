"""Load state dict from paths relative to project root (competition layout)."""

from __future__ import annotations

import os
from pathlib import Path

import torch


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_state_dict_from_package(*parts: str) -> dict[str, torch.Tensor]:
    path = project_root().joinpath(*parts)
    if not path.is_file():
        raise FileNotFoundError(f"Missing weights: {path}")
    return torch.load(path, map_location="cpu", weights_only=True)
