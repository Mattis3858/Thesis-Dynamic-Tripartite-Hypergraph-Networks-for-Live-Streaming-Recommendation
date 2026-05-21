"""HGNN and AttentionMixer layers (from HyperHawkes / RecBole, PyG MessagePassing only)."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as fn
from torch import Tensor
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import scatter


class SimpleHypergraphConv(MessagePassing):
    def __init__(self, in_channels, out_channels, **kwargs):
        kwargs.setdefault("aggr", "add")
        super().__init__(flow="source_to_target", node_dim=0, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(
        self,
        x: Tensor,
        hyperedge_index: Tensor,
        hyperedge_weight: Optional[Tensor] = None,
        self_loops: bool = True,
    ) -> Tensor:
        num_nodes, num_edges = x.size(0), 0
        if hyperedge_index.numel() > 0:
            num_edges = int(hyperedge_index[1].max()) + 1

        if hyperedge_weight is None:
            hyperedge_weight = x.new_ones(num_edges)

        d_norm = scatter(hyperedge_weight[hyperedge_index[1]], hyperedge_index[0], dim=0, dim_size=num_nodes, reduce="sum")
        d_norm = 1.0 / d_norm
        d_norm[d_norm == float("inf")] = 0

        b_norm = scatter(x.new_ones(hyperedge_index.size(1)), hyperedge_index[1], dim=0, dim_size=num_edges, reduce="sum")
        b_norm = 1.0 / b_norm
        b_norm[b_norm == float("inf")] = 0

        out = self.propagate(hyperedge_index, x=x, norm=b_norm, size=(num_nodes, num_edges))
        out = self.propagate(hyperedge_index.flip([0]), x=out, norm=d_norm, size=(num_edges, num_nodes))
        out = out.view(-1, self.out_channels)

        if self_loops:
            zero_nodes = torch.nonzero(out.sum(dim=1) == 0).squeeze()
            if zero_nodes.numel() > 0:
                out[zero_nodes] = x[zero_nodes]

        return out

    def message(self, x_j: Tensor, norm_i: Tensor) -> Tensor:
        return norm_i.view(-1, 1, 1) * x_j.view(-1, 1, self.out_channels)


class HGNN(nn.Module):
    def __init__(self, hidden_size: int, n_layers: int = 1):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.convs = nn.ModuleList([SimpleHypergraphConv(hidden_size, hidden_size) for _ in range(n_layers)])

    def forward(self, x, edge_index, edge_weight=None):
        h = x
        final = [x]
        for i in range(self.n_layers):
            h = self.convs[i](h, edge_index, hyperedge_weight=edge_weight)
            final.append(h)
        return torch.sum(torch.stack(final), dim=0) / (self.n_layers + 1)


class AttentionMixer(nn.Module):
    def __init__(self, hidden_size: int, levels: int = 2, n_heads: int = 2, dropout: float = 0.2):
        super().__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(f"hidden_size ({hidden_size}) must be divisible by n_heads ({n_heads})")
        self.n_levels = levels
        self.dropout = dropout
        self.level_queries = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(levels)])
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.query = nn.Linear(hidden_size, hidden_size, bias=True)
        self.key = nn.Linear(hidden_size, hidden_size, bias=True)
        self.lp_pool = nn.LPPool1d(4, kernel_size=levels, stride=levels)

    def forward(self, item_seq, item_seq_emb, item_seq_len):
        device = item_seq.device
        mask = item_seq.gt(0)
        has_history = mask.any(dim=1)
        queries = []
        for i in range(self.n_levels):
            seq_ids = torch.arange(mask.size(0), device=device, dtype=torch.long)
            level_emb = [
                item_seq_emb[seq_ids, torch.clamp(item_seq_len - (j + 1), min=0)]
                for j in range(i + 1)
            ]
            level_emb = torch.sum(torch.stack(level_emb, dim=1), dim=1)
            queries.append(self.level_queries[i](level_emb).unsqueeze(1))
        queries = torch.stack(queries, dim=1)

        query_layer = self.query(queries).view(-1, queries.size(1), self.hidden_size // self.n_heads)
        key_layer = self.key(item_seq_emb).view(-1, item_seq_emb.size(1), self.hidden_size // self.n_heads)
        value = item_seq_emb.view(-1, item_seq_emb.size(1), self.hidden_size // self.n_heads)

        alpha = torch.sigmoid(torch.matmul(query_layer, key_layer.permute(0, 2, 1)))
        alpha = alpha.view(-1, query_layer.size(1) * self.n_heads, item_seq_emb.size(1)).permute(0, 2, 1)
        alpha = torch.softmax(alpha, dim=1)
        alpha = self.lp_pool(alpha)
        alpha = torch.masked_fill(alpha, ~mask.bool().unsqueeze(-1), float("-inf"))
        alpha = torch.softmax(alpha, dim=1)
        alpha = torch.nan_to_num(alpha, nan=0.0, posinf=0.0, neginf=0.0)
        alpha = fn.dropout(alpha, p=self.dropout, training=self.training)
        output = torch.sum(
            (
                alpha.unsqueeze(-1)
                * value.view(item_seq_emb.size(0), -1, self.n_heads, self.hidden_size // self.n_heads)
            ).view(item_seq_emb.size(0), -1, self.hidden_size)
            * mask.view(mask.shape[0], -1, 1).float(),
            1,
        )
        if not has_history.all():
            output = output.clone()
            output[~has_history] = 0.0
        return output
