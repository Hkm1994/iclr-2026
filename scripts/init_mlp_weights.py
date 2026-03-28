#!/usr/bin/env python3
"""Write models/mlp/state_dict.pt for submission (random init matching architecture)."""

from __future__ import annotations

from pathlib import Path

import torch

from models.mlp.model import MLP


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out = root / "models" / "mlp" / "state_dict.pt"
    m = MLP(config={"skip_weights": True})
    torch.save(m.state_dict(), out)
    print("Wrote", out)


if __name__ == "__main__":
    main()
