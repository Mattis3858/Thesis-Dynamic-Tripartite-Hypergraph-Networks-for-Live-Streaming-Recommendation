"""
HAN (Heterogeneous Graph Attention Network) for tripartite link prediction.

Node-level GAT on three meta-path views (U–S, U–R, S–R) + semantic-level fusion.
PyTorch port of the WWW'19 HAN idea; static graph from train hyperedges.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.han.graph import build_meta_path_biases
from models.han.layers import MultiHeadGAT, SemanticAttention
from utils.DataLoader import TripartiteData


class HAN(nn.Module):
    """Tripartite HAN: (u, v, w, t) -> (y_hat, h_u, h_v, h_w)."""

    def __init__(
        self,
        train_data: TripartiteData,
        node_raw_features: np.ndarray,
        num_nodes: int,
        device: str,
        hidden_dim: int = 64,
        n_heads: int = 4,
        n_gat_layers: int = 1,
        mp_att_size: int = 128,
        dropout: float = 0.1,
        meta_path_nhood: int = 1,
    ):
        super().__init__()
        self.device = device
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.node_feat_dim = hidden_dim
        self.n_heads = n_heads

        feat = torch.from_numpy(node_raw_features.astype(np.float32))
        self.register_buffer("node_features", feat)

        in_dim = int(feat.shape[1])
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        nn.init.xavier_uniform_(self.input_proj.weight)

        # Learnable per-node ID embedding. This dataset's node_raw_features are all-zero, so the
        # projected features carry no node identity; without this, every node collapses to the same
        # vector and HAN degenerates (train AUC ~0.5). Learnable ID embeddings are the standard way
        # to apply a GNN to a featureless graph (cf. LightGCN), i.e. the faithful instantiation of
        # HAN here, not a deviation. padding_idx=0 keeps the padding node at zero.
        self.node_id_embedding = nn.Embedding(num_nodes, hidden_dim, padding_idx=0)
        nn.init.normal_(self.node_id_embedding.weight, std=0.1)
        with torch.no_grad():
            self.node_id_embedding.weight[0].zero_()

        biases = build_meta_path_biases(train_data, num_nodes, nhood=meta_path_nhood)
        for i, b in enumerate(biases):
            self.register_buffer(f"bias_mp{i}", b)

        self.num_meta_paths = len(biases)
        self.gat_layers = nn.ModuleList(
            [
                MultiHeadGAT(
                    hidden_dim if _ == 0 else hidden_dim,
                    hidden_dim,
                    n_heads=n_heads,
                    dropout=dropout,
                )
                for _ in range(n_gat_layers)
            ]
        )
        self.semantic_att = SemanticAttention(hidden_dim, attn_size=mp_att_size)
        self.dropout = dropout

        hid = max(hidden_dim, hidden_dim)
        self.pred_head = nn.Sequential(
            nn.Linear(3 * hidden_dim, hid),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hid, 1),
        )

    def _bias_list(self) -> list[torch.Tensor]:
        return [getattr(self, f"bias_mp{i}") for i in range(self.num_meta_paths)]

    def set_neighbor_sampler(self, neighbor_sampler) -> None:
        """No-op: HAN uses fixed meta-path graphs from train hyperedges."""

    def encode_all_nodes(self) -> torch.Tensor:
        ids = torch.arange(self.num_nodes, device=self.device)
        x = self.input_proj(self.node_features.to(self.device)) + self.node_id_embedding(ids)
        x = F.dropout(x, p=self.dropout, training=self.training)

        mp_embeds: list[torch.Tensor] = []
        for bias in self._bias_list():
            h = x
            for gat in self.gat_layers:
                h = gat(h, bias)
            mp_embeds.append(h)
        stacked = torch.stack(mp_embeds, dim=1)
        return self.semantic_att(stacked)

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
        all_emb = self.encode_all_nodes()
        u = torch.from_numpy(user_node_ids.astype(np.int64)).to(self.device)
        v = torch.from_numpy(streamer_node_ids.astype(np.int64)).to(self.device)
        w = torch.from_numpy(item_node_ids.astype(np.int64)).to(self.device)
        h_u = all_emb[u]
        h_v = all_emb[v]
        h_w = all_emb[w]
        h = torch.cat([h_u, h_v, h_w], dim=-1)
        y_hat = torch.sigmoid(self.pred_head(h))
        return y_hat, h_u, h_v, h_w
