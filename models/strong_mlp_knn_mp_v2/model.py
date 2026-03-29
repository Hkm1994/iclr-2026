"""
strong_mlp_knn_mp_v2: two residual MP blocks, attention-weighted neighbor aggregation,
and richer edges (relative velocity + inverse distance). Original StrongMLPKnnMP unchanged.

Merge uses Option A: concat(x_raw, h_mp1, h_mp2) -> first trunk dim = 58 + 2 * mp_hidden_dim.
"""

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


def _distance_to_surface_features(
    pos: torch.Tensor,
    idcs_airfoil: list[torch.Tensor],
) -> torch.Tensor:
    n = pos.shape[0]
    device, dtype = pos.device, pos.dtype
    out = torch.zeros(n, 2, device=device, dtype=dtype)
    idcs = idcs_airfoil[0] if idcs_airfoil else torch.tensor([], device=device, dtype=torch.long)
    if idcs.numel() == 0:
        return out
    surf = pos[idcs.long()]
    d = torch.cdist(pos, surf, p=2).min(dim=1).values
    out[:, 0] = d
    out[:, 1] = torch.log1p(d)
    return out


def _knn_neighbor_tensors(
    pos: torch.Tensor,
    velocity_in: torch.Tensor,
    vel_mean: torch.Tensor,
    k: int,
    *,
    row_chunk: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
        idx: (N, k)
        nbr6: (N, k, 6)
        per_tau_flat: (N, T_in * 3)
        rel_vel: (N, k, 3) neighbor mean velocity minus center
        inv_dist: (N, k, 1) 1 / (eps + ||delta||)
    """
    idx = knn_indices_brute_force(pos, k, row_chunk=row_chunk)
    nbr_v = vel_mean[idx]
    nbr_p = pos[idx]
    delta = nbr_p - pos.unsqueeze(1)
    nbr6 = torch.cat([nbr_v, delta], dim=-1)

    vm_i = vel_mean.unsqueeze(1)
    rel_vel = nbr_v - vm_i

    dist = delta.norm(dim=-1, keepdim=True)
    inv_dist = 1.0 / (dist + 1e-6)

    t_in = velocity_in.shape[0]
    per_t_chunks: list[torch.Tensor] = []
    for t in range(t_in):
        vt = velocity_in[t]
        per_t_chunks.append(vt[idx].mean(dim=1))
    per_tau_flat = torch.cat(per_t_chunks, dim=-1)
    return idx, nbr6, per_tau_flat, rel_vel, inv_dist


class StrongMLPKnnMPv2(GramForecastModel):
    """
    kNN neighbor attention (pooled + softmax weights), two stacked MP blocks with
    attention-weighted edge aggregation and rich edges, merge(raw, h1, h2), MLP trunk.
    """

    geom_channels = 2
    raw_feature_dim = 58
    mp_hidden_default = 128
    dropout_probability = 0.15
    knn_k_default = 16
    knn_row_chunk_default = 1024
    nbr_attn_d_qk_default = 32
    nbr_attn_d_v_default = 12
    edge_geom_extra = 4  # rel_vel (3) + inv_dist (1)

    def __init__(self, config: Optional[dict[str, Any]] = None):
        super().__init__()
        cfg = dict(config or {})
        d_mp = int(cfg.get("mp_hidden_dim", self.mp_hidden_default))
        self.mp_hidden_dim = d_mp

        trunk = tuple(cfg.get("num_channels", self._default_trunk(d_mp)))
        if len(trunk) != 4:
            raise ValueError(
                "strong_mlp_knn_mp_v2 expects num_channels as 4-tuple "
                "(d_merge, h1, h2, trunk_out)."
            )
        d_merge, h1, h2, h3 = trunk
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

        raw_d = self.raw_feature_dim
        edge_in = 2 * d_mp + 3 + self.edge_geom_extra  # delta + rel_vel + inv_dist

        self.embed1 = Linear(raw_d, d_mp)
        self.edge_mlp1 = torch.nn.Sequential(
            Linear(edge_in, d_mp),
            ReLU(),
            Linear(d_mp, d_mp),
        )
        self.node_mlp1 = torch.nn.Sequential(
            Linear(2 * d_mp, d_mp),
            ReLU(),
            Linear(d_mp, d_mp),
        )
        self.norm_mp1 = LayerNorm(d_mp)

        self.edge_mlp2 = torch.nn.Sequential(
            Linear(edge_in, d_mp),
            ReLU(),
            Linear(d_mp, d_mp),
        )
        self.node_mlp2 = torch.nn.Sequential(
            Linear(2 * d_mp, d_mp),
            ReLU(),
            Linear(d_mp, d_mp),
        )
        self.norm_mp2 = LayerNorm(d_mp)

        merge_in = raw_d + 2 * d_mp
        self.merge = Linear(merge_in, d_merge)

        self.linears = ModuleList()
        self.norms = ModuleList()
        self.activations = ModuleList()
        dims = [d_merge, h1, h2, h3]
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
            weight_path = os.path.join("models", "strong_mlp_knn_mp_v2", "state_dict.pt")
        else:
            weight_path = cfg.get("weight_path")

        if weight_path and os.path.isfile(weight_path):
            state = torch.load(weight_path, map_location="cpu", weights_only=True)
            self.load_state_dict(state)

    @staticmethod
    def _default_trunk(d_mp: int) -> tuple[int, int, int, int]:
        d_merge = StrongMLPKnnMPv2.raw_feature_dim + 2 * d_mp
        return (d_merge, 512, 512, 256)

    def _neighbor_attention(
        self,
        pos: torch.Tensor,
        vel_mean: torch.Tensor,
        surf: torch.Tensor,
        nbr6: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        center = torch.cat([pos, vel_mean, surf], dim=-1)
        q = self.center_q(center)
        k = self.nbr_k(nbr6)
        v = self.nbr_v(nbr6)
        scale = self.nbr_attn_d_qk**0.5
        scores = (q.unsqueeze(1) * k).sum(dim=-1) / scale
        attn = F.softmax(scores, dim=-1)
        pooled = (attn.unsqueeze(-1) * v).sum(dim=1)
        return pooled, attn

    def _rich_edges(
        self,
        h: torch.Tensor,
        idx: torch.Tensor,
        delta: torch.Tensor,
        rel_vel: torch.Tensor,
        inv_dist: torch.Tensor,
    ) -> torch.Tensor:
        """h (N, d_mp), idx (N,k), deltas -> (N, k, edge_in)."""
        h_i = h.unsqueeze(1).expand(-1, idx.shape[1], -1)
        h_j = h[idx]
        return torch.cat([h_i, h_j, delta, rel_vel, inv_dist], dim=-1)

    def _mp_block(
        self,
        h: torch.Tensor,
        idx: torch.Tensor,
        delta: torch.Tensor,
        rel_vel: torch.Tensor,
        inv_dist: torch.Tensor,
        attn: torch.Tensor,
        edge_mlp: torch.nn.Sequential,
        node_mlp: torch.nn.Sequential,
        norm: LayerNorm,
    ) -> torch.Tensor:
        edge = self._rich_edges(h, idx, delta, rel_vel, inv_dist)
        msg = edge_mlp(edge)
        w = attn.unsqueeze(-1)
        agg = (w * msg).sum(dim=1)
        upd = node_mlp(torch.cat([h, agg], dim=-1))
        return norm(h + upd)

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
                f"strong_mlp_knn_mp_v2 expects num_output_timesteps={self.num_output_timesteps} "
                f"input times, got {num_t_in}"
            )
        x_vel = self.flatten_velocity_in(velocity_in)
        t_exp = t.unsqueeze(1).expand(batch_size, num_pos, t.shape[-1])
        surf = _surface_mask(pos, idcs_airfoil)
        vel_mean = velocity_in.mean(dim=1)
        k_eff = min(self.knn_k, num_pos)

        outs: list[torch.Tensor] = []
        for b in range(batch_size):
            idcs_b = [idcs_airfoil[b]]
            geom = _distance_to_surface_features(pos[b], idcs_b)
            idx, nbr6, per_tau, rel_vel, inv_dist = _knn_neighbor_tensors(
                pos[b],
                velocity_in[b],
                vel_mean[b],
                k_eff,
                row_chunk=self.knn_row_chunk,
            )
            nbr_attn, attn_w = self._neighbor_attention(
                pos[b], vel_mean[b], surf[b], nbr6
            )
            x_raw = torch.cat(
                [
                    pos[b],
                    x_vel[b],
                    t_exp[b],
                    surf[b],
                    nbr_attn,
                    per_tau,
                    geom,
                ],
                dim=-1,
            )
            delta = nbr6[:, :, 3:6]

            h0 = self.embed1(x_raw)
            h1 = self._mp_block(
                h0, idx, delta, rel_vel, inv_dist, attn_w,
                self.edge_mlp1, self.node_mlp1, self.norm_mp1,
            )
            h2 = self._mp_block(
                h1, idx, delta, rel_vel, inv_dist, attn_w,
                self.edge_mlp2, self.node_mlp2, self.norm_mp2,
            )
            x = self.merge(torch.cat([x_raw, h1, h2], dim=-1)).unsqueeze(0)

            for linear, norm, activation in zip(
                self.linears, self.norms, self.activations
            ):
                x = activation(norm(linear(self.dropout(x))))

            outs.append(torch.stack([head(x[0]) for head in self.out_heads], dim=0))

        return torch.stack(outs, dim=0)
