import os
from typing import Any, Optional

import torch
from torch.nn import Dropout, Identity, LayerNorm, Linear, ReLU

from models.base import GramForecastModel
from models.features import knn_indices_brute_force


def _surface_mask(
    pos: torch.Tensor,
    idcs_airfoil: list[torch.Tensor],
) -> torch.Tensor:
    b, n, _ = pos.shape
    m = torch.zeros(b, n, 1, device=pos.device, dtype=pos.dtype)
    for i, idcs in enumerate(idcs_airfoil):
        if idcs.numel() > 0:
            m[i, idcs.long(), 0] = 1.0
    return m


def _neighbor_vel_and_rel_pos_pools(
    pos: torch.Tensor,
    vel_mean: torch.Tensor,
    k: int,
    *,
    row_chunk: int,
) -> torch.Tensor:
    """
    pos (N, 3), vel_mean (N, 3) — mean velocity over input time at each point.
    Single kNN index pass; returns (N, 12): mean/max neighbor vel_mean, then mean/max
    of (pos_j - pos_i) over neighbors (self excluded via inf diagonal in kNN).
    """
    idx = knn_indices_brute_force(pos, k, row_chunk=row_chunk)
    nbr_v = vel_mean[idx]
    nbv_mean = nbr_v.mean(dim=1)
    nbv_max = nbr_v.max(dim=1).values
    nbr_p = pos[idx]
    delta = nbr_p - pos.unsqueeze(1)
    nbp_mean = delta.mean(dim=1)
    nbp_max = delta.max(dim=1).values
    return torch.cat([nbv_mean, nbv_max, nbp_mean, nbp_max], dim=-1)


class StrongMLPKnn(GramForecastModel):
    """
    strong_mlp input plus kNN neighbor statistics: time-mean input velocity pools and
    relative-position pools (mean/max over neighbors), then the same MLP trunk.
    """

    # 3 + 5*3 + 10 + 1 + 12 = 41  (12 = vel mean/max + rel-pos delta mean/max)
    num_channels_default = (41, 512, 512, 256, 15)
    dropout_probability = 0.1
    knn_k_default = 16
    knn_row_chunk_default = 1024

    def __init__(self, config: Optional[dict[str, Any]] = None):
        super().__init__()
        cfg = dict(config or {})
        num_channels = tuple(cfg.get("num_channels", self.num_channels_default))
        self.num_channels = num_channels
        self.knn_k = int(cfg.get("knn_k", self.knn_k_default))
        self.knn_row_chunk = int(cfg.get("knn_row_chunk", self.knn_row_chunk_default))
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
            weight_path = os.path.join("models", "strong_mlp_knn", "state_dict.pt")
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
        vel_mean = velocity_in.mean(dim=1)
        k = min(self.knn_k, num_pos)
        nbr_parts = []
        for b in range(batch_size):
            nbr_parts.append(
                _neighbor_vel_and_rel_pos_pools(
                    pos[b], vel_mean[b], k, row_chunk=self.knn_row_chunk
                ).unsqueeze(0)
            )
        nbr_feat = torch.cat(nbr_parts, dim=0)
        x = torch.cat((pos, x_vel, t_exp, surf, nbr_feat), dim=2)

        for linear, norm, activation in zip(self.linears, self.norms, self.activations):
            x = activation(norm(linear(self.dropout(x))))

        x = x.view(batch_size, num_pos, num_t_in, 3).transpose(1, 2)
        return x
