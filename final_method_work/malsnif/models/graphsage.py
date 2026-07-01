from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class WeightedSAGELayer(nn.Module):
    """Memory-conscious weighted mean GraphSAGE layer.

    The layer keeps the same MalSnif graph inputs as the previous weighted GCN:
    node features ``x``, directed ``edge_index`` and optional scalar
    ``edge_weight`` learned from edge event sequences.  For each destination
    node, neighbor features are aggregated as a weighted mean and then combined
    with the node's own representation through separate linear projections:

        h'_v = W_self h_v + W_neigh mean_{u in N(v)}(w_uv h_u)

    This is a GraphSAGE-style mean aggregator.  Message accumulation is done in
    chunks so large CADETS graphs do not materialize an ``[num_edges, dim]``
    tensor for the whole graph at once; this is the main memory reduction versus
    the earlier full-batch GCN implementation.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        bias: bool = True,
        edge_chunk_size: int = 200_000,
        normalize: bool = False,
        use_root: bool = True,
    ):
        super().__init__()
        self.lin_self = nn.Linear(in_dim, out_dim, bias=bias)
        self.lin_neigh = nn.Linear(in_dim, out_dim, bias=False)
        self.edge_chunk_size = max(1, int(edge_chunk_size or 200_000))
        self.normalize = bool(normalize)
        self.use_root = bool(use_root)

    def _aggregate_mean(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None,
    ) -> torch.Tensor:
        n = x.size(0)
        device = x.device
        # Accumulate in fp32 for AMP safety and numerical stability, then cast
        # before the linear layers/autocast continue.  This mirrors the fix for
        # the previous GCN index_add_ dtype issue.
        acc_dtype = torch.float32 if x.is_floating_point() else x.dtype
        x_acc = x.to(dtype=acc_dtype) if x.dtype != acc_dtype else x
        out = torch.zeros((n, x_acc.size(1)), device=device, dtype=acc_dtype)
        deg = torch.zeros(n, device=device, dtype=acc_dtype)

        if edge_index.numel() == 0:
            return out.to(dtype=x.dtype) if out.dtype != x.dtype else out

        src_all, dst_all = edge_index[0], edge_index[1]
        if edge_weight is None:
            ew_all = None
        else:
            ew_all = edge_weight.to(device=device, dtype=acc_dtype)

        num_edges = int(src_all.numel())
        chunk = self.edge_chunk_size
        for start in range(0, num_edges, chunk):
            end = min(start + chunk, num_edges)
            src = src_all[start:end]
            dst = dst_all[start:end]
            if ew_all is None:
                w = torch.ones(src.numel(), device=device, dtype=acc_dtype)
            else:
                w = ew_all[start:end]
            # Keep the [chunk, dim] temporary bounded by edge_chunk_size.
            msg = x_acc[src] * w.unsqueeze(-1)
            out.index_add_(0, dst, msg)
            deg.index_add_(0, dst, w)

        out = out / deg.clamp_min(1e-12).unsqueeze(-1)
        return out.to(dtype=x.dtype) if out.dtype != x.dtype else out

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor | None = None) -> torch.Tensor:
        neigh = self._aggregate_mean(x, edge_index, edge_weight)
        if self.use_root:
            out = self.lin_self(x) + self.lin_neigh(neigh)
        else:
            out = self.lin_neigh(neigh)
        if self.normalize:
            out = F.normalize(out, p=2.0, dim=-1)
        return out


class WeightedGraphSAGE(nn.Module):
    """Stacked weighted GraphSAGE encoder with MalSnif-compatible interface."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        *,
        layers: int = 2,
        dropout: float = 0.2,
        edge_chunk_size: int = 200_000,
        normalize: bool = False,
        use_root: bool = True,
    ):
        super().__init__()
        dims = [in_dim] + [hidden_dim] * max(layers - 1, 0) + [out_dim]
        self.layers = nn.ModuleList(
            [
                WeightedSAGELayer(
                    dims[i],
                    dims[i + 1],
                    edge_chunk_size=edge_chunk_size,
                    normalize=normalize,
                    use_root=use_root,
                )
                for i in range(len(dims) - 1)
            ]
        )
        self.dropout = float(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor | None = None) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index, edge_weight)
            if i < len(self.layers) - 1:
                x = torch.sigmoid(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x
