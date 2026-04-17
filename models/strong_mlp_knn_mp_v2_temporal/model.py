"""
strong_mlp_knn_mp_v2_temporal: residual temporal Conv1d over T_in per point, then
StrongMLPKnnMPv2 (kNN, two MP blocks, trunk). velocity_effective = velocity_in + conv(velocity_in).
"""

from __future__ import annotations

from typing import Any, Optional

import torch
from torch import nn
from torch.nn import ReLU

from models.strong_mlp_knn_mp_v2.model import StrongMLPKnnMPv2


class StrongMLPKnnMPv2Temporal(StrongMLPKnnMPv2):
    """
    Temporal mixing along input timesteps + StrongMLPKnnMPv2.

    If ``skip_weights`` is omitted, defaults to True: plain v2 checkpoints omit
    ``temporal_conv.*`` and do not load with strict=True.
    """

    def __init__(self, config: Optional[dict[str, Any]] = None):
        cfg = dict(config or {})
        if "skip_weights" not in cfg:
            cfg["skip_weights"] = True

        self.temporal_hidden = int(cfg.get("temporal_hidden", 32))
        super().__init__(cfg)

        h = self.temporal_hidden
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(3, h, kernel_size=3, padding=1),
            ReLU(),
            nn.Conv1d(h, h, kernel_size=3, padding=1),
            ReLU(),
            nn.Conv1d(h, 3, kernel_size=1),
        )

    def _temporal_residual(self, velocity_in: torch.Tensor) -> torch.Tensor:
        """(B, T_in, N, 3) -> delta, same shape."""
        b, t, n, _ = velocity_in.shape
        x = velocity_in.permute(0, 2, 3, 1).reshape(b * n, 3, t)
        d = self.temporal_conv(x)
        d = d.reshape(b, n, 3, t).permute(0, 3, 1, 2).contiguous()
        return d

    def forward(
        self,
        t: torch.Tensor,
        pos: torch.Tensor,
        idcs_airfoil: list[torch.Tensor],
        velocity_in: torch.Tensor,
    ) -> torch.Tensor:
        v_eff = velocity_in + self._temporal_residual(velocity_in)
        return super().forward(t, pos, idcs_airfoil, v_eff)
