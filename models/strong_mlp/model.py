import os
from typing import Any, Optional

import torch
from torch.nn import Dropout, Identity, LayerNorm, Linear, ReLU

from models.base import GramForecastModel


def _surface_mask(
    pos: torch.Tensor,
    idcs_airfoil: list[torch.Tensor],
) -> torch.Tensor:
    """(B, N, 1) float indicator: 1 on airfoil surface points."""
    b, n, _ = pos.shape
    m = torch.zeros(b, n, 1, device=pos.device, dtype=pos.dtype)
    for i, idcs in enumerate(idcs_airfoil):
        if idcs.numel() > 0:
            m[i, idcs.long(), 0] = 1.0
    return m


class StrongMLP(GramForecastModel):
    """
    Per-point MLP on pos, flattened velocity_in, full time series t, and a surface indicator.
    Wider/deeper than `MLP` for a stronger streaming baseline.
    """

    num_channels_default = (29, 512, 512, 256, 15)
    dropout_probability = 0.1

    def __init__(self, config: Optional[dict[str, Any]] = None):
        super().__init__()
        cfg = dict(config or {})
        num_channels = tuple(cfg.get("num_channels", self.num_channels_default))
        self.num_channels = num_channels
        dropout_p = float(cfg.get("dropout_probability", self.dropout_probability))

        self.linears = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        self.activations = torch.nn.ModuleList()

        for num_channels_in, num_channels_out in zip(
            num_channels[:-2], num_channels[1:-1]
        ):
            self.linears.append(Linear(num_channels_in, num_channels_out))
            self.norms.append(LayerNorm(num_channels_out))
            self.activations.append(ReLU())

        self.linears.append(Linear(*num_channels[-2:]))
        self.norms.append(Identity())
        self.activations.append(Identity())

        self.dropout = Dropout(dropout_p)

        if cfg.get("skip_weights"):
            weight_path = None
        elif config is None:
            weight_path = os.path.join("models", "strong_mlp", "state_dict.pt")
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
        batch_size, num_t_in, num_pos, _ = velocity_in.shape
        x_vel = self.flatten_velocity_in(velocity_in)
        t_exp = t.unsqueeze(1).expand(batch_size, num_pos, t.shape[-1])
        surf = _surface_mask(pos, idcs_airfoil)
        x = torch.cat((pos, x_vel, t_exp, surf), dim=2)

        for linear, norm, activation in zip(self.linears, self.norms, self.activations):
            x = activation(norm(linear(self.dropout(x))))

        x = x.view(batch_size, num_pos, num_t_in, 3).transpose(1, 2)
        return x
