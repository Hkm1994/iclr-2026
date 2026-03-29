"""Exponential moving average of model weights for optional eval/checkpointing."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch.nn import Module


class ModelEMA:
    """Maintains a shadow ``state_dict`` blended with ``decay`` after each optimizer step."""

    def __init__(self, model: Module, decay: float):
        self.decay = float(decay)
        self.shadow: dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for k, v in model.state_dict().items():
                self.shadow[k] = v.detach().clone()

    @torch.no_grad()
    def update(self, model: Module) -> None:
        for k, v in model.state_dict().items():
            if k not in self.shadow:
                self.shadow[k] = v.detach().clone()
                continue
            sv = self.shadow[k]
            if torch.is_floating_point(v):
                sv.mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)
            else:
                sv.copy_(v.detach())

    @torch.no_grad()
    def swap_in_shadow(self, model: Module) -> dict[str, torch.Tensor]:
        """Copy shadow into ``model``; return backup of previous weights for ``restore``."""
        sd = model.state_dict()
        backup = {k: v.detach().clone() for k, v in sd.items()}
        for k in sd:
            if k in self.shadow:
                sd[k].copy_(self.shadow[k])
        return backup

    @torch.no_grad()
    def restore(self, model: Module, backup: dict[str, torch.Tensor]) -> None:
        sd = model.state_dict()
        for k, v in backup.items():
            sd[k].copy_(v)
