"""
Temporal Graph Networks (TGN; Rossi et al., 2020).

DyGLib memory + graph-attention encoder (same design as the reference implementation,
adapted to ``NeighborSampler``). Used by tripartite baselines and pairwise link prediction.
"""

from __future__ import annotations

import numpy as np
import torch.nn as nn

from models.MemoryModel import MemoryModel
from utils.utils import NeighborSampler


class TGN(nn.Module):
    """TGN backbone: GRU memory bank + temporal graph attention embedding module."""

    def __init__(
        self,
        node_raw_features: np.ndarray,
        edge_raw_features: np.ndarray,
        neighbor_sampler: NeighborSampler,
        time_feat_dim: int,
        num_layers: int = 2,
        num_heads: int = 2,
        dropout: float = 0.1,
        device: str = "cpu",
        src_node_mean_time_shift: float = 0.0,
        src_node_std_time_shift: float = 1.0,
        dst_node_mean_time_shift_dst: float = 0.0,
        dst_node_std_time_shift: float = 1.0,
    ):
        super().__init__()
        self._encoder = MemoryModel(
            node_raw_features=node_raw_features,
            edge_raw_features=edge_raw_features,
            neighbor_sampler=neighbor_sampler,
            time_feat_dim=time_feat_dim,
            model_name="TGN",
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            src_node_mean_time_shift=src_node_mean_time_shift,
            src_node_std_time_shift=src_node_std_time_shift,
            dst_node_mean_time_shift_dst=dst_node_mean_time_shift_dst,
            dst_node_std_time_shift=dst_node_std_time_shift,
            device=device,
        )

    @property
    def node_feat_dim(self) -> int:
        return self._encoder.node_feat_dim

    @property
    def memory_bank(self):
        return self._encoder.memory_bank

    def set_neighbor_sampler(self, neighbor_sampler: NeighborSampler) -> None:
        self._encoder.set_neighbor_sampler(neighbor_sampler)

    def compute_src_dst_node_temporal_embeddings(
        self,
        src_node_ids: np.ndarray,
        dst_node_ids: np.ndarray,
        node_interact_times: np.ndarray,
        edge_ids: np.ndarray | None = None,
        edges_are_positive: bool = True,
        num_neighbors: int = 20,
    ):
        return self._encoder.compute_src_dst_node_temporal_embeddings(
            src_node_ids=src_node_ids,
            dst_node_ids=dst_node_ids,
            node_interact_times=node_interact_times,
            edge_ids=edge_ids,
            edges_are_positive=edges_are_positive,
            num_neighbors=num_neighbors,
        )
