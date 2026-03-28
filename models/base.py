"""Base class for GRaM forecast models (competition forward contract)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import torch
from torch import Tensor, nn


class GramForecastModel(nn.Module, ABC):
    """Geometry-conditioned velocity forecast: first half of window -> second half."""

    @abstractmethod
    def forward(
        self,
        t: Tensor,
        pos: Tensor,
        idcs_airfoil: list[Tensor],
        velocity_in: Tensor,
    ) -> Tensor:
        """
        Args:
            t: (B, 10)
            pos: (B, N, 3)
            idcs_airfoil: length B, each 1D long indices into pos
            velocity_in: (B, 5, N, 3)
        Returns:
            velocity_out: (B, 5, N, 3)
        """
        raise NotImplementedError

    @staticmethod
    def flatten_velocity_in(velocity_in: Tensor) -> Tensor:
        """(B, T_in, N, 3) -> (B, N, T_in * 3)."""
        b, t_in, n, _ = velocity_in.shape
        return velocity_in.transpose(1, 2).reshape(b, n, t_in * 3)
