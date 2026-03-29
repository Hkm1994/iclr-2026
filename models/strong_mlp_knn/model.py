import os
from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch.nn import Dropout, LayerNorm, Linear, ModuleList, ReLU

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


def _knn_neighbor_tensors(
    pos: torch.Tensor,
    velocity_in: torch.Tensor,
    vel_mean: torch.Tensor,
    k: int,
    *,
    row_chunk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Single kNN pass on ``pos`` (N, 3).

    Returns:
        nbr6: (N, k, 6) neighbor [vel_mean_j, pos_j - pos_i] per edge.
        per_tau_flat: (N, T_in * 3) mean neighbor input velocity at each input timestep.
    """
    idx = knn_indices_brute_force(pos, k, row_chunk=row_chunk)
    nbr_v = vel_mean[idx]
    nbr_p = pos[idx]
    delta = nbr_p - pos.unsqueeze(1)
    nbr6 = torch.cat([nbr_v, delta], dim=-1)

    t_in = velocity_in.shape[0]
    per_t_chunks: list[torch.Tensor] = []
    for t in range(t_in):
        vt = velocity_in[t]
        per_t_chunks.append(vt[idx].mean(dim=1))
    per_tau_flat = torch.cat(per_t_chunks, dim=-1)
    return nbr6, per_tau_flat


class StrongMLPKnn(GramForecastModel):
    """
    Per-point MLP with:

    - **Neighbor attention**: query from ``[pos, vel_mean, surface]``, keys/values from
      neighbor ``[vel_mean_j, delta_pos]`` (single kNN index pass).
    - **Per-input-timestep neighbor means**: mean neighbor ``velocity_in`` at each τ.
    - **Output-time heads**: shared trunk then one ``Linear`` head per output timestep.
    """

    # 3 + 5*3 + 10 + 1 + d_v_attn + (T_in*3) = 29 + 12 + 15 = 56 (T_in=5, d_v=12)
    trunk_channels_default = (56, 512, 512, 256)
    dropout_probability = 0.1
    knn_k_default = 16
    knn_row_chunk_default = 1024
    nbr_attn_d_qk_default = 32
    nbr_attn_d_v_default = 12

    def __init__(self, config: Optional[dict[str, Any]] = None):
        super().__init__()
        cfg = dict(config or {})
        trunk = tuple(cfg.get("num_channels", self.trunk_channels_default))
        if len(trunk) != 4:
            raise ValueError(
                "strong_mlp_knn expects num_channels as a 4-tuple "
                "(d_in, hidden1, hidden2, trunk_out); e.g. (56, 512, 512, 256)."
            )
        d_in, h1, h2, h3 = trunk
        self.trunk_channels = trunk
        self.knn_k = int(cfg.get("knn_k", self.knn_k_default))
        self.knn_row_chunk = int(cfg.get("knn_row_chunk", self.knn_row_chunk_default))
        dropout_p = float(cfg.get("dropout_probability", self.dropout_probability))
        d_qk = int(cfg.get("nbr_attn_d_qk", self.nbr_attn_d_qk_default))
        d_v = int(cfg.get("nbr_attn_d_v", self.nbr_attn_d_v_default))

        self.nbr_attn_d_qk = d_qk
        self.nbr_attn_d_v = d_v
        self.center_q = Linear(7, d_qk)
        self.nbr_k = Linear(6, d_qk)
        self.nbr_v = Linear(6, d_v)

        self.linears = ModuleList()
        self.norms = ModuleList()
        self.activations = ModuleList()
        dims = [d_in, h1, h2, h3]
        for a, b in zip(dims[:-1], dims[1:]):
            self.linears.append(Linear(a, b))
            self.norms.append(LayerNorm(b))
            self.activations.append(ReLU())

        num_t_out = int(cfg.get("num_output_timesteps", 5))
        self.num_output_timesteps = num_t_out
        self.out_heads = ModuleList([Linear(h3, 3) for _ in range(num_t_out)])

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

    def _neighbor_attention(
        self,
        pos: torch.Tensor,
        vel_mean: torch.Tensor,
        surf: torch.Tensor,
        nbr6: torch.Tensor,
    ) -> torch.Tensor:
        """(N, d_v) aggregated neighbor context."""
        center = torch.cat([pos, vel_mean, surf], dim=-1)
        q = self.center_q(center)
        k = self.nbr_k(nbr6)
        v = self.nbr_v(nbr6)
        scale = self.nbr_attn_d_qk**0.5
        scores = (q.unsqueeze(1) * k).sum(dim=-1) / scale
        attn = F.softmax(scores, dim=-1)
        return (attn.unsqueeze(-1) * v).sum(dim=1)

    def forward(
        self,
        t: torch.Tensor,
        pos: torch.Tensor,
        idcs_airfoil: list[torch.Tensor],
        velocity_in: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_t_in, num_pos, _ = velocity_in.shape
        if num_t_in != self.num_output_timesteps:
            raise ValueError(
                f"strong_mlp_knn expects num_output_timesteps={self.num_output_timesteps} "
                f"input times, got {num_t_in}"
            )
        x_vel = self.flatten_velocity_in(velocity_in)
        t_exp = t.unsqueeze(1).expand(batch_size, num_pos, t.shape[-1])
        surf = _surface_mask(pos, idcs_airfoil)
        vel_mean = velocity_in.mean(dim=1)
        k_eff = min(self.knn_k, num_pos)

        attn_parts: list[torch.Tensor] = []
        per_tau_parts: list[torch.Tensor] = []
        for b in range(batch_size):
            nbr6, per_tau = _knn_neighbor_tensors(
                pos[b],
                velocity_in[b],
                vel_mean[b],
                k_eff,
                row_chunk=self.knn_row_chunk,
            )
            attn_parts.append(
                self._neighbor_attention(
                    pos[b], vel_mean[b], surf[b], nbr6
                ).unsqueeze(0)
            )
            per_tau_parts.append(per_tau.unsqueeze(0))
        nbr_attn = torch.cat(attn_parts, dim=0)
        per_tau_nbr = torch.cat(per_tau_parts, dim=0)

        x = torch.cat((pos, x_vel, t_exp, surf, nbr_attn, per_tau_nbr), dim=2)

        for linear, norm, activation in zip(self.linears, self.norms, self.activations):
            x = activation(norm(linear(self.dropout(x))))

        outs = [head(x) for head in self.out_heads]
        return torch.stack(outs, dim=1)
