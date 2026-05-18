"""
LightGCN for tripartite (user, streamer, room) link prediction.

Static graph convolution on the train hypergraph skeleton (edges u–v, u–w, v–w).
Timestamps are ignored; same interface as HTTransformer / HyperHawkes.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from models.lightgcn.graph import build_normalized_adj_from_tripartite
from utils.DataLoader import TripartiteData


class LightGCN(nn.Module):
    """Tripartite LightGCN: (u, v, w, t) -> (y_hat, h_u, h_v, h_w)."""

    def __init__(
        self,
        train_data: TripartiteData,
        num_nodes: int,
        device: str,
        embedding_dim: int = 64,
        n_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.device = device
        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim
        self.node_feat_dim = embedding_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(num_nodes, embedding_dim, padding_idx=0)
        nn.init.normal_(self.embedding.weight, std=0.1)
        with torch.no_grad():
            self.embedding.weight[0].zero_()

        graph = build_normalized_adj_from_tripartite(train_data, num_nodes)
        self.register_buffer("graph", graph.to(device))

        hid = max(embedding_dim, embedding_dim)
        self.pred_head = nn.Sequential(
            nn.Linear(3 * embedding_dim, hid),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hid, 1),
        )

    def set_neighbor_sampler(self, neighbor_sampler) -> None:
        """No-op: LightGCN uses a fixed adjacency from train hyperedges."""

    def computer(self) -> torch.Tensor:
        """Layer-wise mean pooling (LightGCN paper). Returns [num_nodes, embedding_dim]."""
        x = self.embedding.weight
        embs = [x]
        g = self.graph
        for _ in range(self.n_layers):
            x = torch.sparse.mm(g, x)
            embs.append(x)
        return torch.mean(torch.stack(embs, dim=0), dim=0)

    def forward(
        self,
        user_node_ids: np.ndarray,
        streamer_node_ids: np.ndarray,
        item_node_ids: np.ndarray,
        interact_times: np.ndarray,
        edge_ids: np.ndarray | None = None,
        edges_are_positive: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        del interact_times, edge_ids, edges_are_positive
        all_emb = self.computer()
        u = torch.from_numpy(user_node_ids.astype(np.int64)).to(self.device)
        v = torch.from_numpy(streamer_node_ids.astype(np.int64)).to(self.device)
        w = torch.from_numpy(item_node_ids.astype(np.int64)).to(self.device)
        h_u = all_emb[u]
        h_v = all_emb[v]
        h_w = all_emb[w]
        h = torch.cat([h_u, h_v, h_w], dim=-1)
        y_hat = torch.sigmoid(self.pred_head(h))
        return y_hat, h_u, h_v, h_w
