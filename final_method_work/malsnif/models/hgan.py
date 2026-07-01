from __future__ import annotations

import hashlib
from typing import Iterable

import torch
from torch import nn
import torch.nn.functional as F

from malsnif.config import Config
from malsnif.constants import NODE_TYPE_TO_ID


def _stable_bucket(value: str, buckets: int) -> int:
    buckets = max(1, int(buckets))
    h = hashlib.sha1(str(value).encode("utf-8", errors="ignore")).hexdigest()
    return int(h[:8], 16) % buckets


def edge_type_ids_from_graph(graph: dict, num_relations: int, device: torch.device) -> torch.Tensor:
    if "edge_type_ids" in graph:
        ids = [int(x) % max(1, num_relations) for x in graph.get("edge_type_ids", [])]
    else:
        ids = [_stable_bucket(x, num_relations) for x in graph.get("edge_types", [])]
    return torch.tensor(ids, dtype=torch.long, device=device)


def edge_time_buckets_from_graph(graph: dict, num_edges: int, num_buckets: int, device: torch.device) -> torch.Tensor:
    num_buckets = max(1, int(num_buckets))
    raw = graph.get("edge_time_buckets")
    if raw is not None and len(raw) == num_edges:
        ids = [max(0, min(num_buckets - 1, int(x))) for x in raw]
        return torch.tensor(ids, dtype=torch.long, device=device)
    raw_times = graph.get("edge_times_ns") or graph.get("edge_first_time_ns")
    if raw_times is not None and len(raw_times) == num_edges:
        vals = []
        for x in raw_times:
            try:
                vals.append(float(x) if x is not None else float("nan"))
            except Exception:
                vals.append(float("nan"))
        finite = [x for x in vals if torch.isfinite(torch.tensor(x))]
        if finite:
            lo, hi = min(finite), max(finite)
            span = max(hi - lo, 1.0)
            ids = []
            for x in vals:
                if not torch.isfinite(torch.tensor(x)):
                    ids.append(0)
                else:
                    ids.append(max(0, min(num_buckets - 1, int((x - lo) / span * num_buckets))))
            return torch.tensor(ids, dtype=torch.long, device=device)
    if num_edges <= 0:
        return torch.zeros((0,), dtype=torch.long, device=device)
    # Backward-compatible fallback for old graph caches that do not store per-edge
    # timestamps: chronological edge-creation order is used as a coarse time proxy.
    pos = torch.arange(num_edges, dtype=torch.float32, device=device)
    return torch.clamp((pos / max(float(num_edges), 1.0) * num_buckets).long(), 0, num_buckets - 1)



def _autocast_target_dtype(x: torch.Tensor) -> torch.dtype:
    """Return the dtype produced by autocast for linear/embedding outputs.

    In CUDA AMP, inputs may stay fp32 while Linear outputs become fp16/bfloat16.
    Preallocating scatter destinations with x.dtype then assigning projected
    outputs can raise "Index put requires the source and destination dtypes
    match".  This helper keeps manual scatter buffers aligned with autocast.
    """
    try:
        if x.device.type == "cuda":
            try:
                enabled = torch.is_autocast_enabled("cuda")
            except TypeError:  # PyTorch < 2.0 compatibility
                enabled = torch.is_autocast_enabled()
            if enabled:
                try:
                    return torch.get_autocast_dtype("cuda")
                except Exception:
                    return torch.get_autocast_gpu_dtype()
        if x.device.type == "cpu":
            try:
                if torch.is_autocast_enabled("cpu"):
                    return torch.get_autocast_dtype("cpu")
            except Exception:
                pass
    except Exception:
        pass
    return x.dtype


class TypeSpecificProjection(nn.Module):
    """Node-type-specific linear projection for heterogeneous provenance nodes."""

    def __init__(self, in_dim: int, out_dim: int, num_types: int):
        super().__init__()
        self.projections = nn.ModuleList([nn.Linear(in_dim, out_dim) for _ in range(num_types)])
        self.fallback = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, node_type_ids: torch.Tensor, use_node_types: bool = True) -> torch.Tensor:
        if not use_node_types:
            return self.fallback(x)
        out_dtype = _autocast_target_dtype(x)
        out = torch.empty((x.size(0), self.projections[0].out_features), dtype=out_dtype, device=x.device)
        covered = torch.zeros(x.size(0), dtype=torch.bool, device=x.device)
        for tid, proj in enumerate(self.projections):
            mask = node_type_ids == tid
            if mask.any():
                out[mask] = proj(x[mask]).to(dtype=out_dtype)
                covered |= mask
        if (~covered).any():
            out[~covered] = self.fallback(x[~covered]).to(dtype=out_dtype)
        return out


class RelationTemporalAttentionLayer(nn.Module):
    """A lightweight relation/time-aware heterogeneous graph attention layer.

    The layer consumes full provenance edges and applies soft Top-k message
    pruning inside message passing only.  The graph object itself is never
    mutated, so analysts can still trace low-attention edges after inference.
    """

    def __init__(self, in_dim: int, out_dim: int, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.out_dim = out_dim
        self.num_types = len(NODE_TYPE_TO_ID)
        self.num_relations = int(getattr(cfg, "hgan_num_relations", 128) or 128)
        self.num_time_buckets = int(getattr(cfg, "hgan_num_time_buckets", 16) or 16)
        self.type_proj = TypeSpecificProjection(in_dim, out_dim, self.num_types)
        self.root_proj = nn.Linear(in_dim, out_dim)
        self.rel_emb = nn.Embedding(self.num_relations, out_dim)
        self.time_emb = nn.Embedding(self.num_time_buckets, out_dim)
        self.attn = nn.Linear(out_dim * 4, 1)
        self.out_norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(float(getattr(cfg, "hgan_attention_dropout", getattr(cfg, "dropout", 0.2))))
        self.negative_slope = float(getattr(cfg, "hgan_leaky_relu_negative_slope", 0.2) or 0.2)

    def _attention_alpha(
        self,
        scores: torch.Tensor,
        dst: torch.Tensor,
        topk: int,
        pruning_mode: str,
        soft_floor: float,
    ) -> tuple[torch.Tensor, dict]:
        # Per-destination softmax.  PyTorch core has no scatter_softmax, so we use
        # a transparent loop over destination groups.  This is slower than PyG but
        # removes a heavy dependency and is adequate for pilot/ablation runs.
        alpha = torch.zeros_like(scores)
        if scores.numel() == 0:
            return alpha, {"edges": 0, "kept_edges": 0, "kept_ratio": None}
        kept_total = 0
        mode = str(pruning_mode or "none").lower()
        soft_floor = float(soft_floor)
        unique_dst = torch.unique(dst)
        for d in unique_dst.tolist():
            idx = torch.where(dst == int(d))[0]
            local_scores = scores[idx]
            if topk > 0 and idx.numel() > topk:
                k = min(int(topk), idx.numel())
                local_top = torch.topk(local_scores, k=k).indices
                keep = torch.zeros(idx.numel(), dtype=torch.bool, device=scores.device)
                keep[local_top] = True
                kept_total += int(keep.sum().item())
                if mode in {"hard", "hard_pruning", "drop"}:
                    active_idx = idx[keep]
                    active_scores = local_scores[keep]
                    alpha[active_idx] = torch.softmax(active_scores, dim=0)
                    continue
                if mode in {"soft", "soft_pruning", "topk"}:
                    # Low-attention edges are downweighted, not removed.  This is
                    # the v1 soft-pruning invariant: aggregation is focused, but
                    # evidence remains inspectable and can still pass a small mass.
                    penalty = torch.full_like(local_scores, torch.log(torch.tensor(max(soft_floor, 1e-8), device=scores.device, dtype=scores.dtype)))
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
        edge_type_ids: torch.Tensor,
        edge_time_buckets: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict | None]:
        n = x.size(0)
        device = x.device
        h = self.type_proj(x, node_type_ids, use_node_types=bool(getattr(self.cfg, "hgan_use_node_types", True)))
        root = self.root_proj(x).to(dtype=h.dtype)
        if edge_index.numel() == 0:
            out = self.out_norm(root)
            return out, None
        src, dst = edge_index[0], edge_index[1]
        rel = self.rel_emb(edge_type_ids.clamp_min(0) % self.num_relations).to(dtype=h.dtype) if bool(getattr(self.cfg, "hgan_use_relation_types", True)) else torch.zeros((src.numel(), self.out_dim), dtype=h.dtype, device=device)
        tvec = self.time_emb(edge_time_buckets.clamp_min(0) % self.num_time_buckets).to(dtype=h.dtype) if bool(getattr(self.cfg, "hgan_use_time_bias", True)) else torch.zeros((src.numel(), self.out_dim), dtype=h.dtype, device=device)
        h_src = h[src]
        h_dst = h[dst]
        msg = h_src + rel + tvec
        score_in = torch.cat([h_src, h_dst, rel, tvec], dim=-1)
        scores = F.leaky_relu(self.attn(score_in).view(-1), negative_slope=self.negative_slope)
        if edge_weight is not None and edge_weight.numel() == scores.numel():
            # Edge-event semantic relevance acts as a small log-bias; clamp avoids
            # -inf and preserves differentiability.
            scores = scores + torch.log(edge_weight.to(device=device, dtype=scores.dtype).clamp_min(1e-6))
        topk = int(getattr(self.cfg, "hgan_topk", 0) or 0)
        pruning_mode = str(getattr(self.cfg, "hgan_pruning_mode", "none") or "none")
        soft_floor = float(getattr(self.cfg, "hgan_soft_pruning_floor", 0.05) or 0.05)
        alpha, stats = self._attention_alpha(scores, dst, topk, pruning_mode, soft_floor)
        msg = msg * alpha.unsqueeze(-1)
        out = torch.zeros((n, self.out_dim), dtype=msg.dtype, device=device)
        out.index_add_(0, dst, msg)
        if bool(getattr(self.cfg, "hgan_use_residual", True)):
            out = out + root
        out = self.out_norm(out)
        out = F.gelu(out)
        out = self.dropout(out)
        if stats is not None and bool(getattr(self.cfg, "return_attention_stats", True)):
            detached_alpha = alpha.detach().float()
            stats.update({
                "alpha_min": float(detached_alpha.min().cpu()) if detached_alpha.numel() else None,
                "alpha_max": float(detached_alpha.max().cpu()) if detached_alpha.numel() else None,
                "alpha_mean": float(detached_alpha.mean().cpu()) if detached_alpha.numel() else None,
            })
        return out, stats


class STHGANEncoder(nn.Module):
    """Stacked spatio-temporal heterogeneous graph attention encoder."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, cfg: Config):
        super().__init__()
        layers = int(getattr(cfg, "gcn_layers", 2) or 2)
        dims = [in_dim] + [hidden_dim] * max(layers - 1, 0) + [out_dim]
        self.layers = nn.ModuleList([RelationTemporalAttentionLayer(dims[i], dims[i + 1], cfg) for i in range(len(dims) - 1)])
        self.cfg = cfg

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        node_type_ids: torch.Tensor,
        edge_type_ids: torch.Tensor,
        edge_time_buckets: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict | None]:
        stats_all: list[dict] = []
        for layer in self.layers:
            x, stats = layer(x, edge_index, node_type_ids, edge_type_ids, edge_time_buckets, edge_weight=edge_weight)
            if stats is not None:
                stats_all.append(stats)
        if not stats_all:
            return x, None
        return x, {"layers": stats_all}
