"""
Wrap DyGLib pairwise temporal encoders for tripartite (user, streamer, room) scoring.

Forward matches HTTransformer: (u, v, w, t) -> (y_hat, h_u, h_v, h_w).
Embeddings use two pairwise calls: (u, v) for h_u, h_v; (u, w) for h_w (second h_u discarded).

Supported backbones (see ``TRIPARTITE_BASELINE_MODELS``): DyGFormer, TGAT, GraphMixer, CAWN, TCL, TGN.
EdgeBank is not included (non-parametric / no ``nn.Module`` temporal encoder in DyGLib).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from models.CAWN import CAWN
from models.DyGFormer import DyGFormer
from models.GraphMixer import GraphMixer
from models.TCL import TCL
from models.TGAT import TGAT
from models.TGN import TGN
from utils.utils import NeighborSampler

TRIPARTITE_BASELINE_MODELS = frozenset({"DyGFormer", "TGAT", "GraphMixer", "CAWN", "TCL", "TGN"})


class BaselineTripartiteWrapper(nn.Module):
    """DyGLib baseline + shared tripartite MLP head (concat + MLP + sigmoid)."""

    def __init__(
        self,
        model_name: str,
        node_raw_features: np.ndarray,
        edge_raw_features: np.ndarray,
        neighbor_sampler: NeighborSampler,
        device: str,
        time_feat_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        num_neighbors: int,
        patch_size: int,
        max_input_sequence_length: int,
        channel_embedding_dim: int,
        time_gap: int,
        walk_length: int,
        position_feat_dim: int,
        num_walk_heads: int,
        num_depths: int,
    ):
        super().__init__()
        if model_name not in TRIPARTITE_BASELINE_MODELS:
            raise ValueError(
                f"Unsupported baseline {model_name!r}. Use one of {sorted(TRIPARTITE_BASELINE_MODELS)}."
            )
        self.model_name = model_name
        self.device = device
        self.num_neighbors = num_neighbors
        self.time_gap = time_gap

        if model_name == "DyGFormer":
            self.backbone = DyGFormer(
                node_raw_features=node_raw_features,
                edge_raw_features=edge_raw_features,
                neighbor_sampler=neighbor_sampler,
                time_feat_dim=time_feat_dim,
                channel_embedding_dim=channel_embedding_dim,
                patch_size=patch_size,
                num_layers=num_layers,
                num_heads=num_heads,
                dropout=dropout,
                max_input_sequence_length=max_input_sequence_length,
                device=device,
            )
        elif model_name == "TGAT":
            self.backbone = TGAT(
                node_raw_features=node_raw_features,
                edge_raw_features=edge_raw_features,
                neighbor_sampler=neighbor_sampler,
                time_feat_dim=time_feat_dim,
                num_layers=num_layers,
                num_heads=num_heads,
                dropout=dropout,
                device=device,
            )
        elif model_name == "GraphMixer":
            self.backbone = GraphMixer(
                node_raw_features=node_raw_features,
                edge_raw_features=edge_raw_features,
                neighbor_sampler=neighbor_sampler,
                time_feat_dim=time_feat_dim,
                num_tokens=num_neighbors,
                num_layers=num_layers,
                dropout=dropout,
                device=device,
            )
        elif model_name == "CAWN":
            self.backbone = CAWN(
                node_raw_features=node_raw_features,
                edge_raw_features=edge_raw_features,
                neighbor_sampler=neighbor_sampler,
                time_feat_dim=time_feat_dim,
                position_feat_dim=position_feat_dim,
                walk_length=walk_length,
                num_walk_heads=num_walk_heads,
                dropout=dropout,
                device=device,
            )
        elif model_name == "TCL":
            self.backbone = TCL(
                node_raw_features=node_raw_features,
                edge_raw_features=edge_raw_features,
                neighbor_sampler=neighbor_sampler,
                time_feat_dim=time_feat_dim,
                num_layers=num_layers,
                num_heads=num_heads,
                num_depths=num_depths,
                dropout=dropout,
                device=device,
            )
        elif model_name == "TGN":
            self.backbone = TGN(
                node_raw_features=node_raw_features,
                edge_raw_features=edge_raw_features,
                neighbor_sampler=neighbor_sampler,
                time_feat_dim=time_feat_dim,
                num_layers=num_layers,
                num_heads=num_heads,
                dropout=dropout,
                device=device,
            )
        else:
            raise AssertionError(model_name)

        d = self.backbone.node_feat_dim
        hid = max(d, channel_embedding_dim)
        self.pred_head = nn.Sequential(
            nn.Linear(3 * d, hid),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hid, 1),
        )

    def set_neighbor_sampler(self, neighbor_sampler: NeighborSampler):
        if self.model_name == "TGN":
            self.backbone.set_neighbor_sampler(neighbor_sampler)
            return
        self.backbone.neighbor_sampler = neighbor_sampler
        if self.backbone.neighbor_sampler.sample_neighbor_strategy in ("uniform", "time_interval_aware"):
            assert self.backbone.neighbor_sampler.seed is not None
            self.backbone.neighbor_sampler.reset_random_state()

    @property
    def memory_bank(self):
        """TGN memory bank (for EarlyStopping checkpoints and epoch backup/restore)."""
        if self.model_name != "TGN":
            raise AttributeError("memory_bank is only defined for TGN.")
        return self.backbone.memory_bank

    def init_memory_for_epoch(self) -> None:
        """Reset TGN memory at the start of each training epoch."""
        if self.model_name == "TGN":
            self.backbone.memory_bank.__init_memory_bank__()

    def backup_memory_bank(self):
        """Snapshot TGN memory after training (for val without polluting train state)."""
        if self.model_name != "TGN":
            raise AttributeError("backup_memory_bank is only defined for TGN.")
        return self.backbone.memory_bank.backup_memory_bank()

    def reload_memory_bank(self, backup) -> None:
        if self.model_name == "TGN":
            self.backbone.memory_bank.reload_memory_bank(backup)

    def detach_memory_after_batch(self):
        """Call after each training batch for TGN (matches DyGLib train_link_prediction)."""
        if self.model_name == "TGN":
            self.backbone.memory_bank.detach_memory_bank()

    def _pair_embeddings(
        self,
        src_node_ids: np.ndarray,
        dst_node_ids: np.ndarray,
        node_interact_times: np.ndarray,
        edge_ids: np.ndarray | None,
        edges_are_positive: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        m = self.backbone
        k = self.num_neighbors
        if self.model_name == "TGN":
            return m.compute_src_dst_node_temporal_embeddings(
                src_node_ids,
                dst_node_ids,
                node_interact_times,
                edge_ids=edge_ids,
                edges_are_positive=edges_are_positive,
                num_neighbors=k,
            )
        if self.model_name == "DyGFormer":
            return m.compute_src_dst_node_temporal_embeddings(src_node_ids, dst_node_ids, node_interact_times)
        if self.model_name == "GraphMixer":
            return m.compute_src_dst_node_temporal_embeddings(
                src_node_ids, dst_node_ids, node_interact_times, num_neighbors=k, time_gap=self.time_gap
            )
        return m.compute_src_dst_node_temporal_embeddings(
            src_node_ids, dst_node_ids, node_interact_times, num_neighbors=k
        )

    def forward(
        self,
        user_node_ids: np.ndarray,
        streamer_node_ids: np.ndarray,
        item_node_ids: np.ndarray,
        interact_times: np.ndarray,
        edge_ids: np.ndarray | None = None,
        edges_are_positive: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param edge_ids: global edge index per row (TripartiteData.edge_ids). TGN only; ignored by other backbones.
        :param edges_are_positive: if True and edge_ids set, TGN updates memory on the first (u,v) pair only,
            then scores (u,w) without a second update (single hyperedge step). If False, no memory updates (e.g. negatives / ranking).
        """
        if self.model_name == "TGN":
            if edges_are_positive and edge_ids is not None:
                h_u, h_v = self._pair_embeddings(
                    user_node_ids, streamer_node_ids, interact_times, edge_ids, edges_are_positive=True
                )
                _, h_w = self._pair_embeddings(
                    user_node_ids, item_node_ids, interact_times, None, edges_are_positive=False
                )
            else:
                h_u, h_v = self._pair_embeddings(
                    user_node_ids, streamer_node_ids, interact_times, None, edges_are_positive=False
                )
                _, h_w = self._pair_embeddings(
                    user_node_ids, item_node_ids, interact_times, None, edges_are_positive=False
                )
        else:
            h_u, h_v = self._pair_embeddings(
                user_node_ids, streamer_node_ids, interact_times, None, True
            )
            _, h_w = self._pair_embeddings(user_node_ids, item_node_ids, interact_times, None, True)
        h = torch.cat([h_u, h_v, h_w], dim=-1)
        y_hat = torch.sigmoid(self.pred_head(h))
        return y_hat, h_u, h_v, h_w
