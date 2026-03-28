import os
from typing import Any, Optional

import torch
from torch.nn import Dropout, Identity, LayerNorm, Linear, ReLU

from models.base import GramForecastModel


class MLP(GramForecastModel):
    """Per-point MLP baseline; submission: Model() loads models/mlp/state_dict.pt when present."""

    num_channels_default = (18, 256, 15)
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
            weight_path = os.path.join("models", "mlp", "state_dict.pt")
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
        del t, idcs_airfoil
        batch_size, num_t_in, num_pos, _ = velocity_in.shape
        x = self.flatten_velocity_in(velocity_in)

        x = torch.cat((pos, x), dim=2)

        for linear, norm, activation in zip(self.linears, self.norms, self.activations):
            x = activation(norm(linear(self.dropout(x))))

        x = x.view(batch_size, num_pos, num_t_in, 3).transpose(1, 2)
        return x
