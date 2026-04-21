from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import torch
from torch import Tensor, nn

from models.strong_mlp_knn_mp_v2.model import StrongMLPKnnMPv2


class LeversTailV2Submission(nn.Module):
    """
    Official submission wrapper for the v2 kNN+MP backbone trained with the levers_tail
    recipe. Loads committed weights at construction time.
    """

    def __init__(self, config: Optional[dict[str, Any]] = None):
        super().__init__()
        # Ignore external config for submission robustness; allow a small escape hatch
        # for local debugging (e.g., {"skip_load": True}).
        cfg = dict(config or {})
        self._skip_load = bool(cfg.get("skip_load", False))

        self.backbone = StrongMLPKnnMPv2(config={"skip_weights": True})

        if not self._skip_load:
            weight_path = Path(__file__).resolve().parent / "state_dict.pt"
            state = torch.load(weight_path, map_location="cpu", weights_only=True)
            self.backbone.load_state_dict(state)

    def forward(
        self,
        t: Tensor,
        pos: Tensor,
        idcs_airfoil: list[Tensor],
        velocity_in: Tensor,
    ) -> Tensor:
        return self.backbone(t, pos, idcs_airfoil, velocity_in)

