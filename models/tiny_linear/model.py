import os
from typing import Any, Optional

import torch
from torch.nn import Linear

from models.base import GramForecastModel


class TinyLinearBaseline(GramForecastModel):
    """One `Linear(5·3 → 5·3)` per point on flattened `velocity_in`. Minimal trainable baseline for E2E tests."""

    def __init__(self, config: Optional[dict[str, Any]] = None):
        super().__init__()
        cfg = dict(config or {})
        t_in, c = 5, 3
        d = t_in * c
        self.fc = Linear(d, d, bias=True)

        if cfg.get("skip_weights"):
            weight_path = None
        elif config is None:
            weight_path = os.path.join("models", "tiny_linear", "state_dict.pt")
        else:
            weight_path = cfg.get("weight_path")

        if weight_path and os.path.isfile(weight_path):
            state = torch.load(weight_path, map_location="cpu", weights_only=True)
            self.load_state_dict(state)

    def forward(
        self,
        t: torch.Tensor,
        pos: torch.Tensor,
        idcs_airfoil: list[torch.Tensor],
        velocity_in: torch.Tensor,
    ) -> torch.Tensor:
        del t, pos, idcs_airfoil
        b, t_in, n, _ = velocity_in.shape
        x = self.flatten_velocity_in(velocity_in)
        y = self.fc(x)
        return y.view(b, n, t_in, 3).transpose(1, 2)
