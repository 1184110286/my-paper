from __future__ import annotations

import math
import torch
from torch import nn
import torch.nn.functional as F


class ElasticTemporalSoftmax(nn.Module):
    """Node-adaptive temporal/attention temperature (EA-THGN ETS).

    For every destination node we predict a temperature tau_i in [tau_min,
    tau_max].  Attention logits for messages arriving at node i are divided by
    tau_i before softmax.  tau<1 sharpens attention, tau>1 smooths it.  This is
    a local attention calibration; it is not the old AGF late-fusion gate.
    """

    def __init__(self, dim: int, tau_min: float = 0.1, tau_max: float = 5.0, hidden: int | None = None):
        super().__init__()
        hidden = int(hidden or dim)
        self.tau_min = float(tau_min)
        self.tau_max = float(tau_max)
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, node_query: torch.Tensor) -> tuple[torch.Tensor, dict]:
        raw = self.net(node_query).view(-1)
        tau = self.tau_min + (self.tau_max - self.tau_min) * torch.sigmoid(raw)
        td = tau.detach().float()
        return tau, {
            "ets_tau_mean": float(td.mean().cpu()) if td.numel() else None,
            "ets_tau_std": float(td.std(unbiased=False).cpu()) if td.numel() else None,
            "ets_tau_min": float(td.min().cpu()) if td.numel() else None,
            "ets_tau_max": float(td.max().cpu()) if td.numel() else None,
        }


class ElasticAttentionWidth(nn.Module):
    """Node-adaptive per-head capacity modulation (EA-THGN EAW).

    Given a [num_nodes, num_heads, head_dim] tensor, predict node-specific head
    coefficients in [0,1]^H and multiply the heads.  Although this is a gating
    operation in EA-THGN terminology, it is *not* the removed AGF semantic vs.
    structure fusion gate; it only controls multi-head bandwidth inside the GNN.
    """

    def __init__(self, head_dim: int, num_heads: int, hidden: int | None = None, dropout: float = 0.0):
        super().__init__()
        hidden = int(hidden or head_dim)
        self.num_heads = int(num_heads)
        self.net = nn.Sequential(
            nn.Linear(head_dim, hidden),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden, self.num_heads),
        )

    def forward(self, heads: torch.Tensor) -> tuple[torch.Tensor, dict]:
        # heads: [N, H, D]
        if heads.numel() == 0:
            return heads, {}
        pooled = heads.mean(dim=1)
        coeff = torch.sigmoid(self.net(pooled)).to(dtype=heads.dtype)  # [N, H]
        out = heads * coeff.unsqueeze(-1)
        cd = coeff.detach().float()
        return out, {
            "eaw_head_mean": float(cd.mean().cpu()),
            "eaw_head_std": float(cd.std(unbiased=False).cpu()),
            "eaw_head_min": float(cd.min().cpu()),
            "eaw_head_max": float(cd.max().cpu()),
        }


class ElasticHopAggregation(nn.Module):
    """Node-adaptive hop-depth aggregation (EA-THGN EHA).

    Aggregates hop-level node embeddings h_i^(0), ..., h_i^(L) by predicting a
    node-specific distribution over hops from the final hop embedding.
    """

    def __init__(self, dim: int, max_hops: int, hidden: int | None = None, dropout: float = 0.0):
        super().__init__()
        self.max_hops = int(max_hops)
        hidden = int(hidden or dim)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden, self.max_hops),
        )

    def forward(self, hop_outputs: list[torch.Tensor]) -> tuple[torch.Tensor, dict]:
        if not hop_outputs:
            raise ValueError("ElasticHopAggregation requires at least one hop tensor")
        if len(hop_outputs) == 1:
            return hop_outputs[0], {"eha_hops": 1, "eha_entropy_mean": 0.0}
        x = torch.stack(hop_outputs, dim=1)  # [N, L, D]
        logits = self.net(hop_outputs[-1])[:, : x.size(1)]
        weights = torch.softmax(logits, dim=-1).to(dtype=x.dtype)
        out = torch.sum(x * weights.unsqueeze(-1), dim=1)
        wd = weights.detach().float()
        entropy = -(wd * torch.log(wd.clamp_min(1e-12))).sum(dim=-1) / math.log(max(wd.size(1), 2))
        stats = {
            "eha_hops": int(wd.size(1)),
            "eha_weight_mean": float(wd.mean().cpu()),
            "eha_weight_std": float(wd.std(unbiased=False).cpu()),
            "eha_entropy_mean": float(entropy.mean().cpu()),
        }
        # Store average weight per hop for lightweight interpretability.
        mean_by_hop = wd.mean(dim=0).cpu().tolist()
        for i, v in enumerate(mean_by_hop):
            stats[f"eha_hop_{i}_mean"] = float(v)
        return out, stats
