from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from malsnif.config import Config
from malsnif.constants import NODE_TYPE_TO_ID
from malsnif.models.hgan import TypeSpecificProjection
from malsnif.models.elastic_adaptivity import ElasticTemporalSoftmax, ElasticAttentionWidth, ElasticHopAggregation


def _largest_valid_heads(dim: int, requested: int) -> int:
    requested = max(1, int(requested or 1))
    for h in range(requested, 0, -1):
        if dim % h == 0:
            return h
    return 1


class ElasticRelationTemporalLayer(nn.Module):
    """Heterogeneous relation/time attention layer with EA-THGN mechanisms.

    The layer deliberately removes the old semantic-structure AGF / edge-fusion
    gate.  It supports three orthogonal node-adaptive mechanisms:
      * ETS: node-specific attention temperature;
      * EAW: node-specific per-head bandwidth modulation;
      * EHA is applied by the stacked encoder after all layers.
    """

    def __init__(self, in_dim: int, edge_dim: int, out_dim: int, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.out_dim = int(out_dim)
        self.num_types = len(NODE_TYPE_TO_ID)
        self.num_relations = int(getattr(cfg, "hgan_num_relations", 128) or 128)
        self.num_time_buckets = int(getattr(cfg, "hgan_num_time_buckets", 16) or 16)
        self.heads = _largest_valid_heads(out_dim, int(getattr(cfg, "ea_num_heads", getattr(cfg, "mcbg_attention_heads", 4)) or 4))
        self.head_dim = out_dim // self.heads
        inner = self.heads * self.head_dim

        self.q_proj = TypeSpecificProjection(in_dim, inner, self.num_types)
        self.k_proj = TypeSpecificProjection(in_dim, inner, self.num_types)
        self.v_proj = TypeSpecificProjection(in_dim, inner, self.num_types)
        self.root_proj = nn.Linear(in_dim, out_dim)
        self.edge_proj = nn.Linear(edge_dim, inner)
        self.rel_emb = nn.Embedding(self.num_relations, inner)
        self.time_emb = nn.Embedding(self.num_time_buckets, inner)
        self.att_vec = nn.Parameter(torch.empty(self.heads, self.head_dim * 5))
        nn.init.xavier_uniform_(self.att_vec)
        self.out_proj = nn.Linear(inner, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(float(getattr(cfg, "hgan_attention_dropout", getattr(cfg, "dropout", 0.2))))
        self.negative_slope = float(getattr(cfg, "hgan_leaky_relu_negative_slope", 0.2) or 0.2)

        hidden = int(getattr(cfg, "ea_hidden_dim", 0) or out_dim)
        self.use_ets = bool(getattr(cfg, "ea_use_ets", False))
        self.use_eaw = bool(getattr(cfg, "ea_use_eaw", False))
        self.ets = ElasticTemporalSoftmax(out_dim, float(getattr(cfg, "ea_tau_min", 0.1)), float(getattr(cfg, "ea_tau_max", 5.0)), hidden) if self.use_ets else None
        self.eaw = ElasticAttentionWidth(self.head_dim, self.heads, hidden=max(1, hidden // self.heads), dropout=float(getattr(cfg, "ea_dropout", 0.0))) if self.use_eaw else None

    def _attention_alpha(self, logits: torch.Tensor, dst: torch.Tensor, tau: torch.Tensor | None = None) -> tuple[torch.Tensor, dict]:
        # logits: [E, H]
        alpha = torch.zeros_like(logits)
        if logits.numel() == 0:
            return alpha, {"edges": 0, "kept_edges": 0, "kept_ratio": None}
        topk = int(getattr(self.cfg, "hgan_topk", 0) or 0)
        mode = str(getattr(self.cfg, "hgan_pruning_mode", "none") or "none").lower()
        soft_floor = float(getattr(self.cfg, "hgan_soft_pruning_floor", 0.05) or 0.05)
        kept_total = 0
        # Looping per destination is slower than torch_scatter but keeps the
        # project dependency-free and readable for research iteration.
        for d in torch.unique(dst).tolist():
            idx = torch.where(dst == int(d))[0]
            local = logits[idx]
            if tau is not None and tau.numel() > int(d):
                local = local / tau[int(d)].to(dtype=local.dtype).clamp_min(1e-6)
            if topk > 0 and idx.numel() > topk:
                score = local.mean(dim=1)
                keep_ids = torch.topk(score, k=min(topk, int(idx.numel()))).indices
                keep = torch.zeros(idx.numel(), dtype=torch.bool, device=logits.device)
                keep[keep_ids] = True
                kept_total += int(keep.sum().item())
                if mode in {"hard", "hard_pruning", "drop"}:
                    active = idx[keep]
                    alpha[active] = torch.softmax(local[keep], dim=0)
                    continue
                if mode in {"soft", "soft_pruning", "topk"}:
                    penalty_value = torch.log(torch.tensor(max(soft_floor, 1e-8), device=logits.device, dtype=logits.dtype))
                    penalty = torch.full((local.size(0), 1), penalty_value, dtype=local.dtype, device=local.device)
                    penalty[keep] = 0.0
                    local = local + penalty
            else:
                kept_total += int(idx.numel())
            alpha[idx] = torch.softmax(local, dim=0)
        return alpha, {
            "edges": int(logits.size(0)),
            "kept_edges": int(kept_total),
            "kept_ratio": float(kept_total / max(int(logits.size(0)), 1)),
            "topk": int(topk),
            "pruning_mode": mode,
        }

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, node_type_ids: torch.Tensor, edge_attr: torch.Tensor, edge_type_ids: torch.Tensor, edge_time_buckets: torch.Tensor, edge_weight: torch.Tensor | None = None) -> tuple[torch.Tensor, dict | None]:
        n = x.size(0)
        device = x.device
        H, D = self.heads, self.head_dim
        q_all = self.q_proj(x, node_type_ids, use_node_types=bool(getattr(self.cfg, "hgan_use_node_types", True))).view(n, H, D)
        k_all = self.k_proj(x, node_type_ids, use_node_types=bool(getattr(self.cfg, "hgan_use_node_types", True))).view(n, H, D)
        v_all = self.v_proj(x, node_type_ids, use_node_types=bool(getattr(self.cfg, "hgan_use_node_types", True))).view(n, H, D)
        root = self.root_proj(x)
        if edge_index.numel() == 0:
            return self.norm(root), None
        src, dst = edge_index[0], edge_index[1]
        h_dst = q_all[dst]
        h_src = k_all[src]
        v_src = v_all[src]
        edge_sem = self.edge_proj(edge_attr).to(dtype=q_all.dtype).view(-1, H, D)
        rel = self.rel_emb(edge_type_ids.clamp_min(0) % self.num_relations).to(dtype=q_all.dtype).view(-1, H, D) if bool(getattr(self.cfg, "hgan_use_relation_types", True)) else torch.zeros_like(edge_sem)
        tvec = self.time_emb(edge_time_buckets.clamp_min(0) % self.num_time_buckets).to(dtype=q_all.dtype).view(-1, H, D) if bool(getattr(self.cfg, "hgan_use_time_bias", True)) else torch.zeros_like(edge_sem)
        att_in = torch.cat([h_dst, h_src, edge_sem, rel, tvec], dim=-1)
        logits = F.leaky_relu((att_in * self.att_vec.unsqueeze(0)).sum(dim=-1), negative_slope=self.negative_slope)
        if edge_weight is not None and edge_weight.numel() == logits.size(0):
            logits = logits + torch.log(edge_weight.to(device=device, dtype=logits.dtype).clamp_min(1e-6)).unsqueeze(-1)
        tau_stats: dict = {}
        tau = None
        if self.ets is not None:
            tau, tau_stats = self.ets(root.to(dtype=x.dtype))
        alpha, stats = self._attention_alpha(logits, dst, tau=tau)
        msg = (v_src + edge_sem + rel + tvec) * alpha.unsqueeze(-1)
        heads = torch.zeros((n, H, D), dtype=msg.dtype, device=device)
        heads.index_add_(0, dst, msg)
        eaw_stats: dict = {}
        if self.eaw is not None:
            heads, eaw_stats = self.eaw(heads)
        out = self.out_proj(heads.reshape(n, H * D)).to(dtype=root.dtype)
        if bool(getattr(self.cfg, "hgan_use_residual", True)):
            out = out + root
        out = self.norm(out)
        out = F.gelu(out)
        out = self.dropout(out)
        if bool(getattr(self.cfg, "return_attention_stats", True)):
            ad = alpha.detach().float()
            stats.update({
                "alpha_min": float(ad.min().cpu()) if ad.numel() else None,
                "alpha_max": float(ad.max().cpu()) if ad.numel() else None,
                "alpha_mean": float(ad.mean().cpu()) if ad.numel() else None,
                "ea_use_ets": bool(self.use_ets),
                "ea_use_eaw": bool(self.use_eaw),
            })
            stats.update(tau_stats)
            stats.update(eaw_stats)
        return out, stats


class ElasticSTHGANEncoder(nn.Module):
    """Stacked ST-HGAN with optional EHA/ETS/EAW mechanisms."""

    def __init__(self, in_dim: int, edge_dim: int, hidden_dim: int, out_dim: int, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.hidden_dim = int(hidden_dim)
        self.out_dim = int(out_dim)
        layers = int(getattr(cfg, "gcn_layers", 2) or 2)
        self.input_proj = nn.Linear(in_dim, hidden_dim) if in_dim != hidden_dim else nn.Identity()
        self.layers = nn.ModuleList([
            ElasticRelationTemporalLayer(hidden_dim, edge_dim, hidden_dim, cfg)
            for _ in range(max(layers, 1))
        ])
        self.final_proj = nn.Linear(hidden_dim, out_dim) if hidden_dim != out_dim else nn.Identity()
        self.use_eha = bool(getattr(cfg, "ea_use_eha", False))
        hidden = int(getattr(cfg, "ea_hidden_dim", 0) or hidden_dim)
        self.eha = ElasticHopAggregation(hidden_dim, max(layers + 1, 1), hidden, dropout=float(getattr(cfg, "ea_dropout", 0.0))) if self.use_eha else None

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, node_type_ids: torch.Tensor, edge_attr: torch.Tensor, edge_type_ids: torch.Tensor, edge_time_buckets: torch.Tensor, edge_weight: torch.Tensor | None = None) -> tuple[torch.Tensor, dict | None]:
        h = self.input_proj(x)
        hop_outputs = [h]
        stats_all: list[dict] = []
        for layer in self.layers:
            h, stats = layer(h, edge_index, node_type_ids, edge_attr, edge_type_ids, edge_time_buckets, edge_weight=edge_weight)
            hop_outputs.append(h)
            if stats is not None:
                stats_all.append(stats)
        eha_stats = None
        if self.eha is not None:
            h, eha_stats = self.eha(hop_outputs)
        h = self.final_proj(h)
        if not bool(getattr(self.cfg, "return_attention_stats", True)):
            return h, None
        out_stats: dict = {"layers": stats_all, "ea_use_eha": bool(self.use_eha), "ea_use_ets": bool(getattr(self.cfg, "ea_use_ets", False)), "ea_use_eaw": bool(getattr(self.cfg, "ea_use_eaw", False))}
        if eha_stats:
            out_stats.update(eha_stats)
        return h, out_stats
