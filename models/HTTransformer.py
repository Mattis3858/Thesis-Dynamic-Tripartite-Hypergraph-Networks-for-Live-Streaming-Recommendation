"""
HT-Transformer (Heterogeneous Tripartite Transformer) for temporal tripartite hypergraphs.

Hyperedge: (u, v, w, t) with u=User, v=Streamer, w=Item.
Defaults follow DyGFormer where the paper does not specify otherwise.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.DyGFormer import TransformerEncoder
from models.modules import TimeEncoder
from utils.utils import NeighborSampler


def _mlp_from_scalar(dim: int, dropout: float = 0.0) -> nn.Sequential:
    """Two-layer MLP mapping scalar counts to dim (DyGFormer-style depth)."""
    return nn.Sequential(
        nn.Linear(1, dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(dim, dim),
    )


class Tripartite3DCooccurrenceEncoder(nn.Module):
    """
    3D neighbor co-occurrence: for neighbor k, c_k = (C_u(k), C_v(k), C_w(k)).

    * Hetero (default): Z = MLP_u(c_u) + MLP_v(c_v) + MLP_w(c_w) with separate MLPs.
    * Homo: Z = shared_MLP(c_u) + shared_MLP(c_v) + shared_MLP(c_w) (weight sharing).
    """

    def __init__(
        self,
        cooccurrence_dim: int,
        dropout: float = 0.0,
        device: str = "cpu",
        hetero: bool = True,
    ):
        super().__init__()
        self.cooccurrence_dim = cooccurrence_dim
        self.device = device
        self.hetero = hetero
        if hetero:
            self.mlp_u = _mlp_from_scalar(cooccurrence_dim, dropout=dropout)
            self.mlp_v = _mlp_from_scalar(cooccurrence_dim, dropout=dropout)
            self.mlp_w = _mlp_from_scalar(cooccurrence_dim, dropout=dropout)
            self.shared_mlp = None
        else:
            self.shared_mlp = _mlp_from_scalar(cooccurrence_dim, dropout=dropout)
            self.mlp_u = None
            self.mlp_v = None
            self.mlp_w = None

    def _counts_tensor_for_sequence(
        self,
        seq_ids: np.ndarray,
        count_u: dict,
        count_v: dict,
        count_w: dict,
        device: torch.device,
    ) -> torch.Tensor:
        """Per-position 3D counts for one sequence in the batch."""
        cu = torch.zeros(len(seq_ids), dtype=torch.float32, device=device)
        cv = torch.zeros(len(seq_ids), dtype=torch.float32, device=device)
        cw = torch.zeros(len(seq_ids), dtype=torch.float32, device=device)
        for idx in range(seq_ids.shape[0]):
            k = int(seq_ids[idx])
            if k == 0:
                continue
            cu[idx] = float(count_u.get(k, 0))
            cv[idx] = float(count_v.get(k, 0))
            cw[idx] = float(count_w.get(k, 0))
        out = torch.stack([cu, cv, cw], dim=-1)  # shape: [seq_len, 3]
        return out

    def forward(
        self,
        u_padded_neighbor_ids: np.ndarray,
        v_padded_neighbor_ids: np.ndarray,
        w_padded_neighbor_ids: np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param u_padded_neighbor_ids: ndarray, shape [batch_size, L_u]
        :param v_padded_neighbor_ids: ndarray, shape [batch_size, L_v]
        :param w_padded_neighbor_ids: ndarray, shape [batch_size, L_w]
        :return: co-occurrence features for u/v/w sequences, each [batch_size, L_*, cooccurrence_dim]
        """
        batch_size = u_padded_neighbor_ids.shape[0]
        u_feats, v_feats, w_feats = [], [], []
        if self.hetero:
            dev = next(self.mlp_u.parameters()).device
        else:
            dev = next(self.shared_mlp.parameters()).device

        for b in range(batch_size):
            su = u_padded_neighbor_ids[b]
            sv = v_padded_neighbor_ids[b]
            sw = w_padded_neighbor_ids[b]

            def counter(ids: np.ndarray):
                d = {}
                for k in ids:
                    kk = int(k)
                    if kk == 0:
                        continue
                    d[kk] = d.get(kk, 0) + 1
                return d

            count_u, count_v, count_w = counter(su), counter(sv), counter(sw)

            counts_u = self._counts_tensor_for_sequence(su, count_u, count_v, count_w, dev)  # shape: [L_u, 3]
            counts_v = self._counts_tensor_for_sequence(sv, count_u, count_v, count_w, dev)  # shape: [L_v, 3]
            counts_w = self._counts_tensor_for_sequence(sw, count_u, count_v, count_w, dev)  # shape: [L_w, 3]

            cu = counts_u[:, 0:1]  # shape: [L_u, 1]
            cv = counts_u[:, 1:2]  # shape: [L_u, 1]
            cw = counts_u[:, 2:3]  # shape: [L_u, 1]
            if self.hetero:
                zu = self.mlp_u(cu) + self.mlp_v(cv) + self.mlp_w(cw)  # shape: [L_u, cooccurrence_dim]
            else:
                zu = self.shared_mlp(cu) + self.shared_mlp(cv) + self.shared_mlp(cw)
            u_feats.append(zu)

            cu = counts_v[:, 0:1]  # shape: [L_v, 1]
            cv = counts_v[:, 1:2]  # shape: [L_v, 1]
            cw = counts_v[:, 2:3]  # shape: [L_v, 1]
            if self.hetero:
                zv = self.mlp_u(cu) + self.mlp_v(cv) + self.mlp_w(cw)  # shape: [L_v, cooccurrence_dim]
            else:
                zv = self.shared_mlp(cu) + self.shared_mlp(cv) + self.shared_mlp(cw)
            v_feats.append(zv)

            cu = counts_w[:, 0:1]  # shape: [L_w, 1]
            cv = counts_w[:, 1:2]  # shape: [L_w, 1]
            cw = counts_w[:, 2:3]  # shape: [L_w, 1]
            if self.hetero:
                zw = self.mlp_u(cu) + self.mlp_v(cv) + self.mlp_w(cw)  # shape: [L_w, cooccurrence_dim]
            else:
                zw = self.shared_mlp(cu) + self.shared_mlp(cv) + self.shared_mlp(cw)
            w_feats.append(zw)

        u_out = torch.stack(u_feats, dim=0)  # shape: [batch_size, L_u, cooccurrence_dim]
        v_out = torch.stack(v_feats, dim=0)  # shape: [batch_size, L_v, cooccurrence_dim]
        w_out = torch.stack(w_feats, dim=0)  # shape: [batch_size, L_w, cooccurrence_dim]

        mask_u = torch.from_numpy(u_padded_neighbor_ids == 0).to(dev)  # shape: [batch_size, L_u]
        mask_v = torch.from_numpy(v_padded_neighbor_ids == 0).to(dev)  # shape: [batch_size, L_v]
        mask_w = torch.from_numpy(w_padded_neighbor_ids == 0).to(dev)  # shape: [batch_size, L_w]
        u_out = u_out.masked_fill(mask_u.unsqueeze(-1), 0.0)  # shape: [batch_size, L_u, cooccurrence_dim]
        v_out = v_out.masked_fill(mask_v.unsqueeze(-1), 0.0)  # shape: [batch_size, L_v, cooccurrence_dim]
        w_out = w_out.masked_fill(mask_w.unsqueeze(-1), 0.0)  # shape: [batch_size, L_w, cooccurrence_dim]

        return u_out, v_out, w_out


class BiasAwareTripartiteMergeLayer(nn.Module):
    """User-conditioned gates on streamer/item embeddings; MLP + sigmoid probability."""

    def __init__(self, d_out: int, dropout: float = 0.1, fusion_mode: str = "concat"):
        super().__init__()
        if fusion_mode not in ("concat", "mean"):
            raise ValueError(f"fusion_mode must be 'concat' or 'mean', got {fusion_mode!r}")
        self.fusion_mode = fusion_mode
        self.W_v = nn.Linear(d_out, d_out, bias=True)
        self.W_w = nn.Linear(d_out, d_out, bias=True)
        hidden = max(d_out, 64)
        mlp_in = 3 * d_out if fusion_mode == "concat" else d_out
        self.mlp_predict = nn.Sequential(
            nn.Linear(mlp_in, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, h_u: torch.Tensor, h_v: torch.Tensor, h_w: torch.Tensor) -> torch.Tensor:
        """
        :param h_u: [batch_size, d_out]
        :param h_v: [batch_size, d_out]
        :param h_w: [batch_size, d_out]
        :return: probabilities [batch_size, 1]
        """
        g_v = torch.sigmoid(self.W_v(h_u))  # shape: [batch_size, d_out]
        g_w = torch.sigmoid(self.W_w(h_u))  # shape: [batch_size, d_out]
        h_tilde_v = g_v * h_v  # shape: [batch_size, d_out]
        h_tilde_w = g_w * h_w  # shape: [batch_size, d_out]
        if self.fusion_mode == "concat":
            h_fused = torch.cat([h_u, h_tilde_v, h_tilde_w], dim=-1)  # shape: [batch_size, 3 * d_out]
        else:
            h_fused = (h_u + h_tilde_v + h_tilde_w) / 3.0  # shape: [batch_size, d_out]
        logits = self.mlp_predict(h_fused)  # shape: [batch_size, 1]
        y_hat = torch.sigmoid(logits)  # shape: [batch_size, 1]
        return y_hat


class HTTransformer(nn.Module):
    """
    Temporal tripartite hypergraph encoder + bias-aware prediction head.
    """

    def __init__(
        self,
        node_raw_features: np.ndarray,
        edge_raw_features: np.ndarray,
        node_type_ids: np.ndarray,
        neighbor_sampler: NeighborSampler,
        time_feat_dim: int,
        hidden_dim: int,
        cooccurrence_dim: int,
        d_out: int | None = None,
        patch_size: int = 1,
        num_layers: int = 2,
        num_heads: int = 2,
        dropout: float = 0.1,
        max_input_sequence_length: int = 512,
        device: str = "cpu",
        use_bias_gate: bool = True,
        use_type_init: bool = True,
        use_hetero_coocc: bool = True,
        fusion_mode: str = "concat",
    ):
        """
        :param node_raw_features: ndarray, shape (num_nodes + 1, node_feat_dim)
        :param edge_raw_features: ndarray, shape (num_edges + 1, edge_feat_dim)
        :param node_type_ids: ndarray, shape (num_nodes + 1,), int type index in [0, num_types)
        :param neighbor_sampler: NeighborSampler on the unified node index space
        :param time_feat_dim: time encoding dimension (DyGFormer TimeEncoder)
        :param hidden_dim: unified d (per-channel projection target)
        :param cooccurrence_dim: d_C for 3D co-occurrence branch before patching
        :param d_out: output representation dim; default node_feat_dim
        """
        super().__init__()

        self.node_raw_features = torch.from_numpy(node_raw_features.astype(np.float32)).to(device)
        self.edge_raw_features = torch.from_numpy(edge_raw_features.astype(np.float32)).to(device)
        self.node_type_ids = torch.from_numpy(node_type_ids.astype(np.int64)).to(device)
        self.num_node_types = int(self.node_type_ids.max().item()) + 1

        self.neighbor_sampler = neighbor_sampler
        self.node_feat_dim = self.node_raw_features.shape[1]
        self.edge_feat_dim = self.edge_raw_features.shape[1]
        self.time_feat_dim = time_feat_dim
        self.hidden_dim = hidden_dim
        self.cooccurrence_dim = cooccurrence_dim
        self.d_out = d_out if d_out is not None else self.node_feat_dim
        self.patch_size = patch_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.max_input_sequence_length = max_input_sequence_length
        self.device = device
        self.use_bias_gate = use_bias_gate
        self.use_type_init = use_type_init
        self.use_hetero_coocc = use_hetero_coocc
        if fusion_mode not in ("concat", "mean"):
            raise ValueError(f"fusion_mode must be 'concat' or 'mean', got {fusion_mode!r}")
        self.fusion_mode = fusion_mode

        self.time_encoder = TimeEncoder(time_dim=time_feat_dim)

        # Module 2: type-aware MLP [x || one-hot(type)] or plain linear on x only
        if use_type_init:
            self.node_init_mlp = nn.Sequential(
                nn.Linear(self.node_feat_dim + self.num_node_types, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.node_init_mlp_no_type = None
        else:
            self.node_init_mlp = None
            self.node_init_mlp_no_type = nn.Linear(self.node_feat_dim, hidden_dim, bias=True)

        self.cooccurrence_encoder = Tripartite3DCooccurrenceEncoder(
            cooccurrence_dim=cooccurrence_dim,
            dropout=dropout,
            device=device,
            hetero=use_hetero_coocc,
        )

        # Module 4: per-channel patch linear maps (d_c * P) -> d
        self.projection_layer = nn.ModuleDict(
            {
                "node": nn.Linear(patch_size * hidden_dim, hidden_dim, bias=True),
                "edge": nn.Linear(patch_size * self.edge_feat_dim, hidden_dim, bias=True),
                "time": nn.Linear(patch_size * self.time_feat_dim, hidden_dim, bias=True),
                "cooccurrence": nn.Linear(patch_size * cooccurrence_dim, hidden_dim, bias=True),
            }
        )

        self.num_channels = 4
        self.fused_dim = self.num_channels * hidden_dim

        self.transformers = nn.ModuleList(
            [
                TransformerEncoder(attention_dim=self.fused_dim, num_heads=num_heads, dropout=dropout)
                for _ in range(num_layers)
            ]
        )

        # Module 5: separate linear heads for u, v, w
        self.output_proj_u = nn.Linear(self.fused_dim, self.d_out, bias=True)
        self.output_proj_v = nn.Linear(self.fused_dim, self.d_out, bias=True)
        self.output_proj_w = nn.Linear(self.fused_dim, self.d_out, bias=True)

        # Module 6: bias-aware merge vs plain fusion + MLP (ablation)
        merge_in = 3 * self.d_out if fusion_mode == "concat" else self.d_out
        if use_bias_gate:
            self.merge_layer = BiasAwareTripartiteMergeLayer(
                d_out=self.d_out, dropout=dropout, fusion_mode=fusion_mode
            )
            self.plain_merge_mlp = None
        else:
            self.merge_layer = None
            merge_hid = max(self.d_out, 64)
            self.plain_merge_mlp = nn.Sequential(
                nn.Linear(merge_in, merge_hid),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(merge_hid, 1),
            )

    def type_aware_node_embedding(self, node_ids: torch.Tensor) -> torch.Tensor:
        """Embed node ids: type-aware [x || p] or plain x -> hidden_dim (Module 2)."""
        x = self.node_raw_features[node_ids]  # shape: [*, node_feat_dim]
        if self.use_type_init:
            types = self.node_type_ids[node_ids]  # shape: [*]
            p = F.one_hot(types, num_classes=self.num_node_types).float()  # shape: [*, num_node_types]
            xp = torch.cat([x, p], dim=-1)  # shape: [*, node_feat_dim + num_node_types]
            return self.node_init_mlp(xp)  # shape: [*, hidden_dim]
        return self.node_init_mlp_no_type(x)  # shape: [*, hidden_dim]

    def pad_sequences(
        self,
        node_ids: np.ndarray,
        node_interact_times: np.ndarray,
        nodes_neighbor_ids_list: list,
        nodes_edge_ids_list: list,
        nodes_neighbor_times_list: list,
        patch_size: int,
        max_input_sequence_length: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        assert max_input_sequence_length - 1 > 0, "max_input_sequence_length must be > 1"
        max_seq_length = 0
        for idx in range(len(nodes_neighbor_ids_list)):
            assert len(nodes_neighbor_ids_list[idx]) == len(nodes_edge_ids_list[idx]) == len(nodes_neighbor_times_list[idx])
            if len(nodes_neighbor_ids_list[idx]) > max_input_sequence_length - 1:
                nodes_neighbor_ids_list[idx] = nodes_neighbor_ids_list[idx][-(max_input_sequence_length - 1) :]
                nodes_edge_ids_list[idx] = nodes_edge_ids_list[idx][-(max_input_sequence_length - 1) :]
                nodes_neighbor_times_list[idx] = nodes_neighbor_times_list[idx][-(max_input_sequence_length - 1) :]
            if len(nodes_neighbor_ids_list[idx]) > max_seq_length:
                max_seq_length = len(nodes_neighbor_ids_list[idx])

        max_seq_length += 1
        if max_seq_length % patch_size != 0:
            max_seq_length += patch_size - (max_seq_length % patch_size)
        assert max_seq_length % patch_size == 0

        padded_nodes_neighbor_ids = np.zeros((len(node_ids), max_seq_length), dtype=np.int64)
        padded_nodes_edge_ids = np.zeros((len(node_ids), max_seq_length), dtype=np.int64)
        padded_nodes_neighbor_times = np.zeros((len(node_ids), max_seq_length), dtype=np.float32)

        for idx in range(len(node_ids)):
            padded_nodes_neighbor_ids[idx, 0] = node_ids[idx]
            padded_nodes_edge_ids[idx, 0] = 0
            padded_nodes_neighbor_times[idx, 0] = node_interact_times[idx]
            if len(nodes_neighbor_ids_list[idx]) > 0:
                ln = len(nodes_neighbor_ids_list[idx])
                padded_nodes_neighbor_ids[idx, 1 : 1 + ln] = nodes_neighbor_ids_list[idx]
                padded_nodes_edge_ids[idx, 1 : 1 + ln] = nodes_edge_ids_list[idx]
                padded_nodes_neighbor_times[idx, 1 : 1 + ln] = nodes_neighbor_times_list[idx]

        return padded_nodes_neighbor_ids, padded_nodes_edge_ids, padded_nodes_neighbor_times

    def get_features(
        self,
        node_interact_times: np.ndarray,
        padded_nodes_neighbor_ids: np.ndarray,
        padded_nodes_edge_ids: np.ndarray,
        padded_nodes_neighbor_times: np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ids = torch.from_numpy(padded_nodes_neighbor_ids).long().to(self.device)
        eids = torch.from_numpy(padded_nodes_edge_ids).long().to(self.device)

        padded_nodes_neighbor_node_emb = self.type_aware_node_embedding(ids)  # shape: [batch_size, max_seq_length, hidden_dim]
        padded_nodes_edge_raw_features = self.edge_raw_features[eids]  # shape: [batch_size, max_seq_length, edge_feat_dim]

        time_delta = torch.from_numpy(node_interact_times[:, np.newaxis] - padded_nodes_neighbor_times).float().to(self.device)
        padded_nodes_neighbor_time_features = self.time_encoder(timestamps=time_delta)  # shape: [batch_size, max_seq_length, time_feat_dim]
        padded_nodes_neighbor_time_features[ids == 0] = 0.0  # shape: [batch_size, max_seq_length, time_feat_dim]

        return padded_nodes_neighbor_node_emb, padded_nodes_edge_raw_features, padded_nodes_neighbor_time_features

    def get_patches(
        self,
        padded_nodes_neighbor_node_raw_features: torch.Tensor,
        padded_nodes_edge_raw_features: torch.Tensor,
        padded_nodes_neighbor_time_features: torch.Tensor,
        padded_nodes_neighbor_co_occurrence_features: torch.Tensor,
        patch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        assert padded_nodes_neighbor_node_raw_features.shape[1] % patch_size == 0
        num_patches = padded_nodes_neighbor_node_raw_features.shape[1] // patch_size
        batch_size = padded_nodes_neighbor_node_raw_features.shape[0]

        pn, pe, pt, pc = [], [], [], []
        for patch_id in range(num_patches):
            s = patch_id * patch_size
            e = s + patch_size
            pn.append(padded_nodes_neighbor_node_raw_features[:, s:e, :])
            pe.append(padded_nodes_edge_raw_features[:, s:e, :])
            pt.append(padded_nodes_neighbor_time_features[:, s:e, :])
            pc.append(padded_nodes_neighbor_co_occurrence_features[:, s:e, :])

        patches_n = torch.stack(pn, dim=1).reshape(batch_size, num_patches, patch_size * self.hidden_dim)  # shape: [batch_size, num_patches, patch_size * hidden_dim]
        patches_e = torch.stack(pe, dim=1).reshape(batch_size, num_patches, patch_size * self.edge_feat_dim)  # shape: [batch_size, num_patches, patch_size * edge_feat_dim]
        patches_t = torch.stack(pt, dim=1).reshape(batch_size, num_patches, patch_size * self.time_feat_dim)  # shape: [batch_size, num_patches, patch_size * time_feat_dim]
        patches_c = torch.stack(pc, dim=1).reshape(batch_size, num_patches, patch_size * self.cooccurrence_dim)  # shape: [batch_size, num_patches, patch_size * cooccurrence_dim]

        return patches_n, patches_e, patches_t, patches_c

    def _project_and_fuse_patches(
        self,
        patches_n: torch.Tensor,
        patches_e: torch.Tensor,
        patches_t: torch.Tensor,
        patches_c: torch.Tensor,
    ) -> torch.Tensor:
        zn = self.projection_layer["node"](patches_n)  # shape: [batch_size, num_patches, hidden_dim]
        ze = self.projection_layer["edge"](patches_e)  # shape: [batch_size, num_patches, hidden_dim]
        zt = self.projection_layer["time"](patches_t)  # shape: [batch_size, num_patches, hidden_dim]
        zc = self.projection_layer["cooccurrence"](patches_c)  # shape: [batch_size, num_patches, hidden_dim]
        z = torch.cat([zn, ze, zt, zc], dim=-1)  # shape: [batch_size, num_patches, 4 * hidden_dim]
        return z

    def compute_tripartite_temporal_embeddings(
        self,
        user_node_ids: np.ndarray,
        streamer_node_ids: np.ndarray,
        item_node_ids: np.ndarray,
        interact_times: np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param user_node_ids: ndarray [batch_size]
        :param streamer_node_ids: ndarray [batch_size]
        :param item_node_ids: ndarray [batch_size]
        :param interact_times: ndarray [batch_size], hyperedge time t
        :return: h_u, h_v, h_w each [batch_size, d_out]
        """
        u_nbr_list, u_edge_list, u_time_list = self.neighbor_sampler.get_all_first_hop_neighbors(
            node_ids=user_node_ids, node_interact_times=interact_times
        )
        v_nbr_list, v_edge_list, v_time_list = self.neighbor_sampler.get_all_first_hop_neighbors(
            node_ids=streamer_node_ids, node_interact_times=interact_times
        )
        w_nbr_list, w_edge_list, w_time_list = self.neighbor_sampler.get_all_first_hop_neighbors(
            node_ids=item_node_ids, node_interact_times=interact_times
        )

        u_pad_ids, u_pad_e, u_pad_t = self.pad_sequences(
            user_node_ids,
            interact_times,
            u_nbr_list,
            u_edge_list,
            u_time_list,
            self.patch_size,
            self.max_input_sequence_length,
        )
        v_pad_ids, v_pad_e, v_pad_t = self.pad_sequences(
            streamer_node_ids,
            interact_times,
            v_nbr_list,
            v_edge_list,
            v_time_list,
            self.patch_size,
            self.max_input_sequence_length,
        )
        w_pad_ids, w_pad_e, w_pad_t = self.pad_sequences(
            item_node_ids,
            interact_times,
            w_nbr_list,
            w_edge_list,
            w_time_list,
            self.patch_size,
            self.max_input_sequence_length,
        )

        u_co, v_co, w_co = self.cooccurrence_encoder(u_pad_ids, v_pad_ids, w_pad_ids)
        # shape: [batch_size, L_u, cooccurrence_dim], [batch_size, L_v, cooccurrence_dim], [batch_size, L_w, cooccurrence_dim]

        u_n, u_e, u_tf = self.get_features(interact_times, u_pad_ids, u_pad_e, u_pad_t)
        v_n, v_e, v_tf = self.get_features(interact_times, v_pad_ids, v_pad_e, v_pad_t)
        w_n, w_e, w_tf = self.get_features(interact_times, w_pad_ids, w_pad_e, w_pad_t)

        u_pn, u_pe, u_pt, u_pc = self.get_patches(u_n, u_e, u_tf, u_co, self.patch_size)
        v_pn, v_pe, v_pt, v_pc = self.get_patches(v_n, v_e, v_tf, v_co, self.patch_size)
        w_pn, w_pe, w_pt, w_pc = self.get_patches(w_n, w_e, w_tf, w_co, self.patch_size)

        zu = self._project_and_fuse_patches(u_pn, u_pe, u_pt, u_pc)  # shape: [batch_size, l_u, 4 * hidden_dim]
        zv = self._project_and_fuse_patches(v_pn, v_pe, v_pt, v_pc)  # shape: [batch_size, l_v, 4 * hidden_dim]
        zw = self._project_and_fuse_patches(w_pn, w_pe, w_pt, w_pc)  # shape: [batch_size, l_w, 4 * hidden_dim]

        z = torch.cat([zu, zv, zw], dim=1)  # shape: [batch_size, l_u + l_v + l_w, 4 * hidden_dim]

        for transformer in self.transformers:
            z = transformer(z)  # shape: [batch_size, l_u + l_v + l_w, 4 * hidden_dim]

        l_u = zu.shape[1]
        l_v = zv.shape[1]
        l_w = zw.shape[1]

        hu_seq = z[:, :l_u, :]  # shape: [batch_size, l_u, 4 * hidden_dim]
        hv_seq = z[:, l_u : l_u + l_v, :]  # shape: [batch_size, l_v, 4 * hidden_dim]
        hw_seq = z[:, l_u + l_v :, :]  # shape: [batch_size, l_w, 4 * hidden_dim]

        hu_pool = torch.mean(hu_seq, dim=1)  # shape: [batch_size, 4 * hidden_dim]
        hv_pool = torch.mean(hv_seq, dim=1)  # shape: [batch_size, 4 * hidden_dim]
        hw_pool = torch.mean(hw_seq, dim=1)  # shape: [batch_size, 4 * hidden_dim]

        h_u = self.output_proj_u(hu_pool)  # shape: [batch_size, d_out]
        h_v = self.output_proj_v(hv_pool)  # shape: [batch_size, d_out]
        h_w = self.output_proj_w(hw_pool)  # shape: [batch_size, d_out]

        return h_u, h_v, h_w

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
        Full forward: tripartite encoder + merge head (bias-aware or plain concat+MLP).
        edge_ids / edges_are_positive are unused (API parity with tripartite baselines / TGN).
        :return: y_hat [batch_size, 1], h_u, h_v, h_w [batch_size, d_out]
        """
        del edge_ids, edges_are_positive
        h_u, h_v, h_w = self.compute_tripartite_temporal_embeddings(
            user_node_ids, streamer_node_ids, item_node_ids, interact_times
        )
        if self.use_bias_gate:
            y_hat = self.merge_layer(h_u, h_v, h_w)  # shape: [batch_size, 1]
        else:
            if self.fusion_mode == "concat":
                h_fused = torch.cat([h_u, h_v, h_w], dim=-1)
            else:
                h_fused = (h_u + h_v + h_w) / 3.0
            y_hat = torch.sigmoid(self.plain_merge_mlp(h_fused))
        return y_hat, h_u, h_v, h_w

    def set_neighbor_sampler(self, neighbor_sampler: NeighborSampler):
        self.neighbor_sampler = neighbor_sampler
        if self.neighbor_sampler.sample_neighbor_strategy in ["uniform", "time_interval_aware"]:
            assert self.neighbor_sampler.seed is not None
            self.neighbor_sampler.reset_random_state()
