"""PyTorch GAT node-level attention + semantic (meta-path) attention."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GATLayer(nn.Module):
    """Single-head graph attention layer with additive masking (HAN/GAT)."""

    def __init__(self, in_dim: int, out_dim: int, negative_slope: float = 0.2):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.attn_src = nn.Parameter(torch.zeros(1, out_dim))
        self.attn_dst = nn.Parameter(torch.zeros(1, out_dim))
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.xavier_uniform_(self.attn_src)
        nn.init.xavier_uniform_(self.attn_dst)

    def forward(self, x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        """
        :param x: [N, in_dim]
        :param bias: [N, N], 0 for allowed edges, large negative elsewhere
        """
        h = self.linear(x)
        e = self.attn_src @ h.T + h @ self.attn_dst.T
        e = self.leaky_relu(e) + bias.to(x.device)
        alpha = F.softmax(e, dim=-1)
        return alpha @ h


class MultiHeadGAT(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert out_dim % n_heads == 0
        self.heads = nn.ModuleList([GATLayer(in_dim, out_dim // n_heads) for _ in range(n_heads)])
        self.dropout = dropout

    def forward(self, x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        parts = [F.elu(head(x, bias)) for head in self.heads]
        out = torch.cat(parts, dim=-1)
        return F.dropout(out, p=self.dropout, training=self.training)


class SemanticAttention(nn.Module):
    """Meta-path level attention (HAN SimpleAttLayer)."""

    def __init__(self, hidden_dim: int, attn_size: int = 128):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, attn_size)
        self.u = nn.Parameter(torch.zeros(attn_size))
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.xavier_uniform_(self.u.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [N, num_meta_paths, hidden_dim] -> [N, hidden_dim]"""
        v = torch.tanh(self.proj(x))
        scores = (v * self.u).sum(dim=-1)
        alpha = F.softmax(scores, dim=-1)
        return (x * alpha.unsqueeze(-1)).sum(dim=1)
