from __future__ import annotations

from torch import nn

from malsnif.config import Config
from malsnif.models.gcn import WeightedGCN
from malsnif.models.graphsage import WeightedGraphSAGE


def build_graph_encoder(in_dim: int, cfg: Config) -> nn.Module:
    """Build the graph message-passing backend.

    The rest of MalSnif is intentionally decoupled from the concrete GNN layer.
    Both backends accept ``(node_x, edge_index, edge_weight)`` and return node
    embeddings of shape ``[num_nodes, hidden_dim]``.
    """
    name = str(getattr(cfg, "graph_encoder", "graphsage") or "graphsage").lower()
    layers = int(getattr(cfg, "gcn_layers", 2))
    if name in {"sage", "graphsage", "weighted_graphsage"}:
        return WeightedGraphSAGE(
            in_dim,
            cfg.hidden_dim,
            cfg.hidden_dim,
            layers=layers,
            dropout=cfg.dropout,
            edge_chunk_size=int(getattr(cfg, "edge_chunk_size", 200_000) or 200_000),
            normalize=bool(getattr(cfg, "graphsage_normalize", False)),
            use_root=bool(getattr(cfg, "graphsage_use_root", True)),
        )
    if name in {"gcn", "weighted_gcn"}:
        return WeightedGCN(in_dim, cfg.hidden_dim, cfg.hidden_dim, layers=layers, dropout=cfg.dropout)
    raise ValueError(f"Unsupported graph_encoder={name!r}; expected 'graphsage' or 'gcn'.")
