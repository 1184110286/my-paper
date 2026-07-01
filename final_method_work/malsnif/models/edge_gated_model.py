from __future__ import annotations

import numpy as np
import torch
from torch import nn

from malsnif.config import Config
from malsnif.models.semantic import MCBGEncoder, GDTCMCBGEncoder, RGDBiGRUMCBGEncoder, HierarchicalLogEncoder
from malsnif.models.hgan import edge_type_ids_from_graph, edge_time_buckets_from_graph
from malsnif.models.edge_gated_hgan import EdgeGatedSTHGANEncoder


class MalSnifAlignedEdgeGatedModel(nn.Module):
    """MalSnif-aligned MCBG + edge-gated ST-HGAN model.

    Unlike the v1 AGF late-fusion model, this model does not create two final
    semantic/structural embeddings and concatenate or gate them before the
    classifier.  Instead, MCBG encodes node and edge audit-event sequences into
    MalSnif-style graph attributes, and the gate is applied per edge during
    heterogeneous message passing.  The classifier consumes only the final
    propagated node embedding.
    """

    def __init__(self, embedding_matrix: np.ndarray, cfg: Config):
        super().__init__()
        self.cfg = cfg
        semantic_name = str(getattr(cfg, "semantic_encoder", "mcbg") or "mcbg").lower()
        if semantic_name in {"baseline", "gru_bilstm", "malsnif"}:
            self.node_encoder = HierarchicalLogEncoder(embedding_matrix, cfg)
        elif semantic_name in {"mcbg", "cnn_bigru", "cnn_bigru_attention"}:
            self.node_encoder = MCBGEncoder(embedding_matrix, cfg)
        elif semantic_name in {"gdtc_mcbg", "gdtc", "e1_gdtc_mcbg", "gated_dilated_tcn"}:
            self.node_encoder = GDTCMCBGEncoder(embedding_matrix, cfg)
        elif semantic_name in {"rgd_bigru_mcbg", "rgd_bigru", "e1_rgd_bigru_mcbg", "residual_gated_dilated_bigru"}:
            self.node_encoder = RGDBiGRUMCBGEncoder(embedding_matrix, cfg)
        else:
            raise ValueError(f"Unsupported semantic_encoder={semantic_name!r}")
        # Share encoder weights for node and edge event sequences, matching the
        # v1 project convention and keeping the parameter budget modest.
        self.edge_encoder = self.node_encoder
        self.node_proj = nn.Linear(cfg.behavior_dim, cfg.hidden_dim) if cfg.behavior_dim != cfg.hidden_dim else nn.Identity()
        self.edge_weight_mlp = nn.Linear(cfg.behavior_dim, 1)
        if getattr(cfg, "edge_weight_init_zero", False):
            nn.init.zeros_(self.edge_weight_mlp.weight)
            nn.init.zeros_(self.edge_weight_mlp.bias)
        self.graph_encoder = EdgeGatedSTHGANEncoder(cfg.hidden_dim, cfg.behavior_dim, cfg.hidden_dim, cfg.hidden_dim, cfg)
        self.out = nn.Linear(cfg.hidden_dim, 1)

    def _edge_weights(self, edge_attr: torch.Tensor) -> torch.Tensor:
        raw = self.edge_weight_mlp(edge_attr).view(-1)
        mode = str(getattr(self.cfg, "edge_weight_mode", "legacy_sigmoid") or "legacy_sigmoid").lower()
        if mode in {"legacy", "legacy_sigmoid", "sigmoid"}:
            return torch.sigmoid(raw)
        if mode in {"softplus", "positive_softplus"}:
            zero = torch.zeros((), device=raw.device, dtype=raw.dtype)
            return torch.nn.functional.softplus(raw) / torch.nn.functional.softplus(zero)
        if mode in {"centered", "centered_sigmoid"}:
            return 2.0 * torch.sigmoid(raw)
        return torch.sigmoid(raw)

    def forward(self, graph: dict, device) -> dict:
        node_type_ids = torch.tensor(graph["node_type_ids"], dtype=torch.long, device=device)
        node_attr_raw = self.node_encoder.forward_nested(
            graph["node_event_ids"], self.cfg.max_events_per_node, self.cfg.max_tokens_per_event, device, graph.get("node_event_weights")
        )
        node_x = self.node_proj(node_attr_raw)

        edge_weight_stats = None
        if len(graph.get("edge_index", [])) > 0:
            edge_index = torch.tensor(graph["edge_index"], dtype=torch.long, device=device).t().contiguous()
            edge_attr = self.edge_encoder.forward_nested(
                graph["edge_event_ids"], self.cfg.max_events_per_edge, self.cfg.max_tokens_per_event, device, graph.get("edge_event_weights")
            )
            edge_weight = self._edge_weights(edge_attr) if bool(getattr(self.cfg, "use_edge_weights", True)) else None
            edge_type_ids = edge_type_ids_from_graph(graph, int(getattr(self.cfg, "hgan_num_relations", 128) or 128), device)
            edge_time_buckets = edge_time_buckets_from_graph(
                graph,
                edge_index.size(1),
                int(getattr(self.cfg, "hgan_num_time_buckets", 16) or 16),
                device,
            )
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
            edge_attr = torch.zeros((0, self.cfg.behavior_dim), device=device, dtype=node_x.dtype)
            edge_weight = None
            edge_type_ids = torch.zeros((0,), dtype=torch.long, device=device)
            edge_time_buckets = torch.zeros((0,), dtype=torch.long, device=device)

        if edge_weight is not None and edge_weight.numel():
            ew = edge_weight.detach().float().cpu()
            edge_weight_stats = {
                "count": int(ew.numel()),
                "min": float(ew.min()),
                "max": float(ew.max()),
                "mean": float(ew.mean()),
                "std": float(ew.std(unbiased=False)),
            }

        h, attention_stats, gate_stats = self.graph_encoder(
            node_x,
            edge_index,
            node_type_ids,
            edge_attr,
            edge_type_ids,
            edge_time_buckets,
            edge_weight=edge_weight,
        )
        logits = self.out(h).view(-1)
        probs = torch.sigmoid(logits)
        process_mask = torch.tensor(graph["process_mask"], dtype=torch.bool, device=device)
        if self.cfg.graph_level:
            ps = probs[process_mask] if process_mask.any() else probs
            graph_prob = ps.mean() if self.cfg.graph_readout == "mean" and ps.numel() else (ps.max() if ps.numel() else probs.max())
        else:
            graph_prob = probs.max() if probs.numel() else torch.tensor(0.0, device=device)
        return {
            "node_logits": logits,
            "node_probs": probs,
            "graph_prob": graph_prob,
            "process_mask": process_mask,
            "edge_weight_stats": edge_weight_stats,
            "gate_stats": gate_stats,
            "attention_stats": attention_stats,
            "model_variant": "edge_gated_st_hgan_mcbg",
        }
