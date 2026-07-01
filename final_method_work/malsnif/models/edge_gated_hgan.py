from __future__ import annotations

from collections import defaultdict

import torch
from torch import nn
import torch.nn.functional as F

from malsnif.config import Config
from malsnif.constants import NODE_TYPE_TO_ID
from malsnif.models.hgan import TypeSpecificProjection


class EdgeGateNetwork(nn.Module):
    """Semantic-aware edge gate used inside message passing.

    The gate is deliberately placed *inside* graph propagation, not after a
    separate semantic/structure late fusion.  It consumes source/target node
    states, MCBG edge semantics, relation embeddings and time embeddings, then
    returns either a vector gate or a scalar gate per provenance edge.
    """

    def __init__(self, dim: int, cfg: Config):
        super().__init__()
        self.dim = int(dim)
        self.mode = str(getattr(cfg, "edge_gate_mode", "vector") or "vector").lower()
        hidden = int(getattr(cfg, "edge_gate_hidden_dim", 0) or dim)
        dropout = float(getattr(cfg, "edge_gate_dropout", getattr(cfg, "gate_dropout", 0.1)) or 0.0)
        out_dim = 1 if self.mode in {"scalar", "scalar_gate"} else dim
        # [h_src, h_dst, edge_sem, relation, time, |src-dst|, src*dst]
        self.net = nn.Sequential(
            nn.Linear(dim * 7, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )
        self.temperature = float(getattr(cfg, "edge_gate_temperature", getattr(cfg, "gate_temperature", 1.0)) or 1.0)
        self.use_edge_semantics = bool(getattr(cfg, "edge_gate_use_edge_semantics", True))

    def forward(
        self,
        h_src: torch.Tensor,
        h_dst: torch.Tensor,
        edge_sem: torch.Tensor,
        rel: torch.Tensor,
        tvec: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        mode = self.mode
        if mode in {"none", "off", "disabled", "no_gate"}:
            z = torch.ones_like(h_src)
            return z, self._stats(z, "edge_none")
        if mode in {"fixed", "fixed_one", "ones"}:
            z = torch.ones_like(h_src)
            return z, self._stats(z, "edge_fixed_one")
        if mode in {"fixed_half", "half"}:
            z = torch.full_like(h_src, 0.5)
            return z, self._stats(z, "edge_fixed_half")
        if not self.use_edge_semantics:
            edge_sem = torch.zeros_like(edge_sem)
        gate_in = torch.cat([
            h_src,
            h_dst,
            edge_sem,
            rel,
            tvec,
            torch.abs(h_src - h_dst),
            h_src * h_dst,
        ], dim=-1)
        z = torch.sigmoid(self.net(gate_in) / max(self.temperature, 1e-6))
        if z.size(-1) == 1:
            z = z.expand_as(h_src)
            mode_name = "edge_scalar"
        else:
            mode_name = "edge_vector"
        return z, self._stats(z, mode_name)

    @staticmethod
    def _stats(z: torch.Tensor, mode: str) -> dict:
        zd = z.detach().float()
        return {
            "gate_mode": mode,
            "edge_gate_mean": float(zd.mean().cpu()) if zd.numel() else None,
            "edge_gate_std": float(zd.std(unbiased=False).cpu()) if zd.numel() else None,
            "edge_gate_min": float(zd.min().cpu()) if zd.numel() else None,
            "edge_gate_max": float(zd.max().cpu()) if zd.numel() else None,
            # Compatibility with existing aggregate_gate_stats/evaluate code.
            "gate_semantic_mean": float(zd.mean().cpu()) if zd.numel() else None,
            "gate_semantic_std": float(zd.std(unbiased=False).cpu()) if zd.numel() else None,
            "gate_semantic_min": float(zd.min().cpu()) if zd.numel() else None,
            "gate_semantic_max": float(zd.max().cpu()) if zd.numel() else None,
            "gate_structure_mean": float((1.0 - zd).mean().cpu()) if zd.numel() else None,
        }


class EdgeGatedRelationTemporalLayer(nn.Module):
    """ST-HGAN layer with semantic edge-gated message passing.

    This layer keeps the MalSnif-style flow: MCBG semantic encoders produce
    node/edge attributes first, graph propagation consumes these attributes, and
    only the final propagated node embedding is classified.  The edge gate is a
    message filter, not a late semantic/structural fusion classifier.
    """

    def __init__(self, in_dim: int, edge_dim: int, out_dim: int, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.out_dim = int(out_dim)
        self.num_types = len(NODE_TYPE_TO_ID)
        self.num_relations = int(getattr(cfg, "hgan_num_relations", 128) or 128)
        self.num_time_buckets = int(getattr(cfg, "hgan_num_time_buckets", 16) or 16)
        self.type_proj = TypeSpecificProjection(in_dim, out_dim, self.num_types)
        self.root_proj = nn.Linear(in_dim, out_dim)
        self.edge_proj = nn.Linear(edge_dim, out_dim)
        self.rel_emb = nn.Embedding(self.num_relations, out_dim)
        self.time_emb = nn.Embedding(self.num_time_buckets, out_dim)
        # Attention also sees edge semantics; gate controls message amplitude.
        self.attn = nn.Linear(out_dim * 5, 1)
        self.edge_gate = EdgeGateNetwork(out_dim, cfg)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(float(getattr(cfg, "hgan_attention_dropout", getattr(cfg, "dropout", 0.2))))
        self.negative_slope = float(getattr(cfg, "hgan_leaky_relu_negative_slope", 0.2) or 0.2)

    def _attention_alpha(self, scores: torch.Tensor, dst: torch.Tensor) -> tuple[torch.Tensor, dict]:
        alpha = torch.zeros_like(scores)
        if scores.numel() == 0:
            return alpha, {"edges": 0, "kept_edges": 0, "kept_ratio": None}
        topk = int(getattr(self.cfg, "hgan_topk", 0) or 0)
        mode = str(getattr(self.cfg, "hgan_pruning_mode", "none") or "none").lower()
        soft_floor = float(getattr(self.cfg, "hgan_soft_pruning_floor", 0.05) or 0.05)
        kept_total = 0
        for d in torch.unique(dst).tolist():
            idx = torch.where(dst == int(d))[0]
            local_scores = scores[idx]
            if topk > 0 and idx.numel() > topk:
                k = min(topk, int(idx.numel()))
                local_top = torch.topk(local_scores, k=k).indices
                keep = torch.zeros(idx.numel(), dtype=torch.bool, device=scores.device)
                keep[local_top] = True
                kept_total += int(keep.sum().item())
                if mode in {"hard", "hard_pruning", "drop"}:
                    active_idx = idx[keep]
                    alpha[active_idx] = torch.softmax(local_scores[keep], dim=0)
                    continue
                if mode in {"soft", "soft_pruning", "topk"}:
                    penalty_value = torch.log(torch.tensor(max(soft_floor, 1e-8), device=scores.device, dtype=scores.dtype))
                    penalty = torch.full_like(local_scores, penalty_value)
                    penalty[keep] = 0.0
                    local_scores = local_scores + penalty
            else:
                kept_total += int(idx.numel())
            alpha[idx] = torch.softmax(local_scores, dim=0)
        return alpha, {
            "edges": int(scores.numel()),
            "kept_edges": int(kept_total),
            "kept_ratio": float(kept_total / max(int(scores.numel()), 1)),
            "topk": int(topk),
            "pruning_mode": mode,
        }

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        node_type_ids: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_type_ids: torch.Tensor,
        edge_time_buckets: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict | None]:
        n = x.size(0)
        device = x.device
        h = self.type_proj(x, node_type_ids, use_node_types=bool(getattr(self.cfg, "hgan_use_node_types", True)))
        root = self.root_proj(x).to(dtype=h.dtype)
        if edge_index.numel() == 0:
            return self.norm(root), None
        src, dst = edge_index[0], edge_index[1]
        rel = self.rel_emb(edge_type_ids.clamp_min(0) % self.num_relations).to(dtype=h.dtype) if bool(getattr(self.cfg, "hgan_use_relation_types", True)) else torch.zeros((src.numel(), self.out_dim), dtype=h.dtype, device=device)
        tvec = self.time_emb(edge_time_buckets.clamp_min(0) % self.num_time_buckets).to(dtype=h.dtype) if bool(getattr(self.cfg, "hgan_use_time_bias", True)) else torch.zeros((src.numel(), self.out_dim), dtype=h.dtype, device=device)
        edge_sem = self.edge_proj(edge_attr).to(dtype=h.dtype)
        h_src = h[src]
        h_dst = h[dst]

        score_in = torch.cat([h_src, h_dst, edge_sem, rel, tvec], dim=-1)
        scores = F.leaky_relu(self.attn(score_in).view(-1), negative_slope=self.negative_slope)
        if edge_weight is not None and edge_weight.numel() == scores.numel():
            scores = scores + torch.log(edge_weight.to(device=device, dtype=scores.dtype).clamp_min(1e-6))
        alpha, stats = self._attention_alpha(scores, dst)

        gate, gate_stats = self.edge_gate(h_src, h_dst, edge_sem, rel, tvec)
        msg_base = h_src + edge_sem + rel + tvec
        msg = gate * msg_base * alpha.unsqueeze(-1)
        out = torch.zeros((n, self.out_dim), dtype=msg.dtype, device=device)
        out.index_add_(0, dst, msg)
        if bool(getattr(self.cfg, "hgan_use_residual", True)):
            out = out + root
        out = self.norm(out)
        out = F.gelu(out)
        out = self.dropout(out)
        if stats is not None and bool(getattr(self.cfg, "return_attention_stats", True)):
            detached_alpha = alpha.detach().float()
            stats.update({
                "alpha_min": float(detached_alpha.min().cpu()) if detached_alpha.numel() else None,
                "alpha_max": float(detached_alpha.max().cpu()) if detached_alpha.numel() else None,
                "alpha_mean": float(detached_alpha.mean().cpu()) if detached_alpha.numel() else None,
            })
            stats.update(gate_stats)
        return out, stats


class EdgeGatedSTHGANEncoder(nn.Module):
    """Stacked edge-gated ST-HGAN encoder."""

    def __init__(self, in_dim: int, edge_dim: int, hidden_dim: int, out_dim: int, cfg: Config):
        super().__init__()
        layers = int(getattr(cfg, "gcn_layers", 2) or 2)
        dims = [in_dim] + [hidden_dim] * max(layers - 1, 0) + [out_dim]
        self.layers = nn.ModuleList([
            EdgeGatedRelationTemporalLayer(dims[i], edge_dim, dims[i + 1], cfg)
            for i in range(len(dims) - 1)
        ])

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        node_type_ids: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_type_ids: torch.Tensor,
        edge_time_buckets: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict | None, dict | None]:
        stats_all: list[dict] = []
        gate_rows: list[dict] = []
        for layer in self.layers:
            x, stats = layer(x, edge_index, node_type_ids, edge_attr, edge_type_ids, edge_time_buckets, edge_weight=edge_weight)
            if stats is not None:
                stats_all.append(stats)
                gate_rows.append({k: v for k, v in stats.items() if k.startswith("gate_") or k.startswith("edge_gate_")})
        attention_stats = {"layers": stats_all} if stats_all else None
        gate_stats = self._aggregate_gate_rows(gate_rows) if gate_rows else None
        return x, attention_stats, gate_stats

    @staticmethod
    def _aggregate_gate_rows(rows: list[dict]) -> dict | None:
        if not rows:
            return None
        out: dict = {"gate_mode": sorted(set(str(r.get("gate_mode", "")) for r in rows))[-1]}
        numeric = ["edge_gate_mean", "edge_gate_std", "edge_gate_min", "edge_gate_max", "gate_semantic_mean", "gate_semantic_std", "gate_semantic_min", "gate_semantic_max", "gate_structure_mean"]
        for key in numeric:
            vals = []
            for row in rows:
                v = row.get(key)
                try:
                    if v is not None and torch.isfinite(torch.tensor(float(v))):
                        vals.append(float(v))
                except Exception:
                    pass
            if vals:
                out[key] = float(sum(vals) / len(vals))
        return out
