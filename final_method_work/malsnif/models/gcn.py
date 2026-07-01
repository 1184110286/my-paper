from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class WeightedGCNLayer(nn.Module):
    """A lightweight GCN layer with scalar edge weights.

    Implements D^-1/2 (A + I) D^-1/2 X W as in the MalSnif paper.
    """

    def __init__(self, in_dim: int, out_dim: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor | None = None) -> torch.Tensor:
        n = x.size(0)
        device = x.device

        # Message aggregation uses index_add_, whose destination tensor and
        # source tensor must have the same dtype.  Under CUDA AMP, node features
        # can be float16/bfloat16 while edge weights and degree normalization are
        # float32.  Accumulate the graph messages in float32, then let the linear
        # layer/autocast choose the best compute dtype.  This avoids AMP-only
        # crashes without changing the GCN equation.
        acc_dtype = torch.float32 if x.is_floating_point() else x.dtype
        x_msg = x.to(dtype=acc_dtype) if x.dtype != acc_dtype else x

        if edge_index.numel() == 0:
            loop = torch.arange(n, device=device)
            src = dst = loop
            w = torch.ones(n, device=device, dtype=acc_dtype)
        else:
            src, dst = edge_index[0], edge_index[1]
            if edge_weight is None:
                edge_weight = torch.ones(src.numel(), device=device, dtype=acc_dtype)
            else:
                edge_weight = edge_weight.to(device=device, dtype=acc_dtype)
            loop = torch.arange(n, device=device)
            src = torch.cat([src, loop], dim=0)
            dst = torch.cat([dst, loop], dim=0)
            w = torch.cat([edge_weight, torch.ones(n, device=device, dtype=acc_dtype)], dim=0)
        deg = torch.zeros(n, device=device, dtype=acc_dtype)
        deg.index_add_(0, dst, w)
        deg = deg.clamp_min(1e-12)
        norm = w * deg[src].pow(-0.5) * deg[dst].pow(-0.5)
        msg = x_msg[src] * norm.unsqueeze(-1)
        out = torch.zeros((n, x_msg.size(1)), device=device, dtype=acc_dtype)
        out.index_add_(0, dst, msg)
        return self.linear(out)


class WeightedGCN(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, layers: int = 2, dropout: float = 0.2):
        super().__init__()
        dims = [in_dim] + [hidden_dim] * max(layers - 1, 0) + [out_dim]
        self.layers = nn.ModuleList([WeightedGCNLayer(dims[i], dims[i + 1]) for i in range(len(dims) - 1)])
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor | None = None) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index, edge_weight)
            if i < len(self.layers) - 1:
                x = torch.sigmoid(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x
