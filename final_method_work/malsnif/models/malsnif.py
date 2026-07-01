from __future__ import annotations

import numpy as np
import torch
from torch import nn

from malsnif.config import Config
from malsnif.constants import NODE_TYPE_TO_ID
from malsnif.models.graph_encoder import build_graph_encoder
from malsnif.models.semantic import HierarchicalLogEncoder, MCBGEncoder, GDTCMCBGEncoder, RGDBiGRUMCBGEncoder
from malsnif.models.hgan import STHGANEncoder, edge_type_ids_from_graph, edge_time_buckets_from_graph
from malsnif.models.fusion import AdaptiveGatedFusion, StaticConcatFusion, MeanFusion
from malsnif.models.edge_gated_model import MalSnifAlignedEdgeGatedModel
from malsnif.models.elastic_model import MalSnifAlignedElasticModel


class MalSnifBaselineModel(nn.Module):
    """Faithful baseline reproduction kept as A0.

    This class preserves the original project flow: Word2Vec -> GRU event encoder
    -> BiLSTM sequence encoder -> GCN/GraphSAGE -> MLP/Sigmoid.  New idea code is
    implemented in a separate class and is selected only by config.
    """

    def __init__(self, embedding_matrix: np.ndarray, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.use_semantics = cfg.use_semantics
        if cfg.use_semantics:
            self.log_encoder = HierarchicalLogEncoder(embedding_matrix, cfg)
            node_in = cfg.behavior_dim
            edge_in = cfg.behavior_dim
        else:
            self.type_embed = nn.Embedding(len(NODE_TYPE_TO_ID), cfg.behavior_dim)
            node_in = cfg.behavior_dim
            edge_in = cfg.behavior_dim
        self.edge_weight_mlp = nn.Sequential(nn.Linear(edge_in, 1))
        if getattr(cfg, "edge_weight_init_zero", True):
            nn.init.zeros_(self.edge_weight_mlp[0].weight)
            nn.init.zeros_(self.edge_weight_mlp[0].bias)
        self.gcn = build_graph_encoder(node_in, cfg)
        self.out = nn.Linear(cfg.hidden_dim, 1)

    def _edge_weights(self, edge_feat: torch.Tensor) -> torch.Tensor:
        raw = self.edge_weight_mlp[0](edge_feat).view(-1)
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
        if self.use_semantics:
            node_x = self.log_encoder.forward_nested(
                graph["node_event_ids"], self.cfg.max_events_per_node, self.cfg.max_tokens_per_event, device, graph.get("node_event_weights")
            )
            if len(graph["edge_index"]) > 0:
                edge_feat = self.log_encoder.forward_nested(
                    graph["edge_event_ids"], self.cfg.max_events_per_edge, self.cfg.max_tokens_per_event, device, graph.get("edge_event_weights")
                )
            else:
                edge_feat = torch.zeros((0, self.cfg.behavior_dim), device=device)
        else:
            node_x = self.type_embed(node_type_ids)
            edge_feat = torch.zeros((len(graph["edge_index"]), self.cfg.behavior_dim), device=device)
        edge_weight_stats = None
        if len(graph["edge_index"]) > 0:
            edge_index = torch.tensor(graph["edge_index"], dtype=torch.long, device=device).t().contiguous()
            edge_weight = self._edge_weights(edge_feat) if self.cfg.use_edge_weights else None
            if edge_weight is not None and edge_weight.numel():
                ew = edge_weight.detach().float().cpu()
                edge_weight_stats = {
                    "count": int(ew.numel()),
                    "min": float(ew.min()),
                    "max": float(ew.max()),
                    "mean": float(ew.mean()),
                    "std": float(ew.std(unbiased=False)),
                }
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
            edge_weight = None
        h = self.gcn(node_x, edge_index, edge_weight)
        logits = self.out(h).view(-1)
        probs = torch.sigmoid(logits)
        process_mask = torch.tensor(graph["process_mask"], dtype=torch.bool, device=device)
        if self.cfg.graph_level:
            ps = probs[process_mask] if process_mask.any() else probs
            if self.cfg.graph_readout == "mean":
                graph_prob = ps.mean() if ps.numel() else probs.mean()
            else:
                graph_prob = ps.max() if ps.numel() else probs.max()
        else:
            graph_prob = probs.max() if probs.numel() else torch.tensor(0.0, device=device)
        return {
            "node_logits": logits,
            "node_probs": probs,
            "graph_prob": graph_prob,
            "process_mask": process_mask,
            "edge_weight_stats": edge_weight_stats,
            "model_variant": "baseline",
        }


class AGFSTHGANMCBGModel(nn.Module):
    """Implementation of AGF-ST-HGAN-MCBG v1.

    Semantic branch: Word2Vec event vectors -> multi-kernel CNN -> BiGRU ->
    multi-head attention (MCBG).  Structure branch: node/relation/time-aware
    heterogeneous graph attention with optional soft Top-k pruning.  Fusion:
    node-wise adaptive vector gate by default.
    """

    def __init__(self, embedding_matrix: np.ndarray, cfg: Config):
        super().__init__()
        self.cfg = cfg
        semantic_name = str(getattr(cfg, "semantic_encoder", "mcbg") or "mcbg").lower()
        if semantic_name in {"baseline", "gru_bilstm", "malsnif"}:
            self.semantic_encoder = HierarchicalLogEncoder(embedding_matrix, cfg)
        elif semantic_name in {"mcbg", "cnn_bigru", "cnn_bigru_attention"}:
            self.semantic_encoder = MCBGEncoder(embedding_matrix, cfg)
        elif semantic_name in {"gdtc_mcbg", "gdtc", "e1_gdtc_mcbg", "gated_dilated_tcn"}:
            self.semantic_encoder = GDTCMCBGEncoder(embedding_matrix, cfg)
        elif semantic_name in {"rgd_bigru_mcbg", "rgd_bigru", "e1_rgd_bigru_mcbg", "residual_gated_dilated_bigru"}:
            self.semantic_encoder = RGDBiGRUMCBGEncoder(embedding_matrix, cfg)
        else:
            raise ValueError(f"Unsupported semantic_encoder={semantic_name!r}")
        self.edge_encoder = self.semantic_encoder
        self.semantic_proj = nn.Linear(cfg.behavior_dim, cfg.hidden_dim) if cfg.behavior_dim != cfg.hidden_dim else nn.Identity()
        self.structure_encoder = STHGANEncoder(cfg.hidden_dim, cfg.hidden_dim, cfg.hidden_dim, cfg)
        self.edge_weight_mlp = nn.Sequential(nn.Linear(cfg.behavior_dim, 1))
        if getattr(cfg, "edge_weight_init_zero", True):
            nn.init.zeros_(self.edge_weight_mlp[0].weight)
            nn.init.zeros_(self.edge_weight_mlp[0].bias)
        mode = str(getattr(cfg, "fusion_mode", "agf") or "agf").lower()
        self.fusion_mode = mode
        if mode in {"agf", "gated", "vector_gate", "adaptive_gate"}:
            self.fusion = AdaptiveGatedFusion(cfg.hidden_dim, cfg, scalar_gate=False)
            clf_in = cfg.hidden_dim
        elif mode in {"scalar_gate", "a7_scalar_gate"}:
            self.fusion = AdaptiveGatedFusion(cfg.hidden_dim, cfg, scalar_gate=True)
            clf_in = cfg.hidden_dim
        elif mode in {"static_concat", "concat", "a3"}:
            self.fusion = StaticConcatFusion(cfg.hidden_dim, cfg)
            clf_in = cfg.hidden_dim
        elif mode in {"mean", "avg"}:
            self.fusion = MeanFusion()
            clf_in = cfg.hidden_dim
        elif mode in {"semantic_only", "mcbg_only", "a1"}:
            self.fusion = None
            clf_in = cfg.hidden_dim
        elif mode in {"structure_only", "hgan_only", "a2"}:
            self.fusion = None
            clf_in = cfg.hidden_dim
        else:
            raise ValueError(f"Unsupported fusion_mode={mode!r}")
        self.out = nn.Linear(clf_in, 1)

    def _edge_weights(self, edge_feat: torch.Tensor) -> torch.Tensor:
        raw = self.edge_weight_mlp[0](edge_feat).view(-1)
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
        semantic_raw = self.semantic_encoder.forward_nested(
            graph["node_event_ids"], self.cfg.max_events_per_node, self.cfg.max_tokens_per_event, device, graph.get("node_event_weights")
        )
        semantic = self.semantic_proj(semantic_raw)

        edge_weight_stats = None
        gate_stats = None
        gate_values = None
        attention_stats = None

        if self.fusion_mode in {"semantic_only", "mcbg_only", "a1"}:
            # A1 / semantic-only ablation: skip edge encoding and graph aggregation.
            # The previous implementation still encoded all edge_event_ids before
            # returning the semantic branch, which made A1 much slower and memory
            # heavier without changing its logits.  This is an implementation
            # optimization only; A1 remains the same hypothesis test.
            structure = None
            fused = semantic
            gate_stats = {"gate_mode": "semantic_only", "gate_semantic_mean": 1.0, "gate_structure_mean": 0.0}
        else:
            if len(graph["edge_index"]) > 0:
                edge_index = torch.tensor(graph["edge_index"], dtype=torch.long, device=device).t().contiguous()
                edge_feat = self.edge_encoder.forward_nested(
                    graph["edge_event_ids"], self.cfg.max_events_per_edge, self.cfg.max_tokens_per_event, device, graph.get("edge_event_weights")
                )
                edge_weight = self._edge_weights(edge_feat) if self.cfg.use_edge_weights else None
                edge_type_ids = edge_type_ids_from_graph(graph, int(getattr(self.cfg, "hgan_num_relations", 128) or 128), device)
                edge_time_buckets = edge_time_buckets_from_graph(
                    graph,
                    edge_index.size(1),
                    int(getattr(self.cfg, "hgan_num_time_buckets", 16) or 16),
                    device,
                )
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
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
            structure, attention_stats = self.structure_encoder(
                semantic,
                edge_index,
                node_type_ids,
                edge_type_ids,
                edge_time_buckets,
                edge_weight=edge_weight,
            )
            if self.fusion_mode in {"structure_only", "hgan_only", "a2"}:
                fused = structure
                gate_stats = {"gate_mode": "structure_only", "gate_semantic_mean": 0.0, "gate_structure_mean": 1.0}
            else:
                fused, gate_values, gate_stats = self.fusion(semantic, structure)
        logits = self.out(fused).view(-1)
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
            "gate_values": gate_values,
            "model_variant": "agf_st_hgan_mcbg",
        }


class MalSnifModel(nn.Module):
    """Config-driven model wrapper.

    Existing training/evaluation code instantiates MalSnifModel.  This wrapper
    keeps that public API stable while allowing the Research Contract v1 idea to
    be selected through config.model_variant.
    """

    def __init__(self, embedding_matrix: np.ndarray, cfg: Config):
        super().__init__()
        variant = str(getattr(cfg, "model_variant", "baseline") or "baseline").lower()
        if variant in {"baseline", "malsnif", "a0"}:
            self.impl = MalSnifBaselineModel(embedding_matrix, cfg)
        elif variant in {"agf_st_hgan_mcbg", "agf", "st_hgan_mcbg", "a4"}:
            self.impl = AGFSTHGANMCBGModel(embedding_matrix, cfg)
        elif variant in {"edge_gated_st_hgan_mcbg", "edge_gated", "egmp", "v2_edge_gated", "b3"}:
            self.impl = MalSnifAlignedEdgeGatedModel(embedding_matrix, cfg)
        elif variant in {"ea_st_hgan_mcbg", "ea_thgn_mcbg", "elastic_st_hgan_mcbg", "v3_ea", "e0", "e7"}:
            self.impl = MalSnifAlignedElasticModel(embedding_matrix, cfg)
        else:
            raise ValueError(f"Unsupported model_variant={variant!r}; expected baseline, agf_st_hgan_mcbg, edge_gated_st_hgan_mcbg or ea_st_hgan_mcbg")

    def __getattr__(self, name: str):
        # Backward compatibility for tests and older utility code that accessed
        # baseline attributes directly, e.g. model.edge_weight_mlp or model.gcn.
        try:
            return super().__getattr__(name)
        except AttributeError as exc:
            try:
                impl = super().__getattr__("impl")
            except AttributeError:
                raise exc
            if hasattr(impl, name):
                return getattr(impl, name)
            raise exc

    def load_state_dict(self, state_dict, strict: bool = True):
        # Older checkpoints from the pre-factory baseline do not have the
        # ``impl.`` prefix.  Load them directly into the wrapped implementation.
        if state_dict and not any(str(k).startswith("impl.") for k in state_dict.keys()):
            return self.impl.load_state_dict(state_dict, strict=strict)
        return super().load_state_dict(state_dict, strict=strict)

    def forward(self, graph: dict, device) -> dict:
        return self.impl(graph, device)
