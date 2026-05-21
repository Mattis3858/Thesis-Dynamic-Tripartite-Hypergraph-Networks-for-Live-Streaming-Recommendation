"""
HyperHawkes for tripartite (user, streamer, room) link prediction.

Adapts the WWW'25 HyperHawkes encoder (hypergraph + Hawkes intent + short-term mixer)
to DyGLib tripartite data: user histories are streams of streamers; rooms use a separate
embedding branch. Hypergraph is mined on streamer co-occurrence sessions (train split).
"""

from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.hyperhawkes.graph import build_bipartite_from_arrays, construct_global_hyper_graph
from models.hyperhawkes.layers import AttentionMixer, HGNN
from utils.DataLoader import TripartiteData

logger = logging.getLogger(__name__)

TYPE_USER = 1
TYPE_STREAMER = 2
TYPE_ROOM = 3


def _gather_indexes(output: torch.Tensor, gather_index: torch.Tensor) -> torch.Tensor:
    """
    Gather last valid timestep from ``output`` [batch, seq_len, hidden].
    ``gather_index`` is per-row index (e.g. seq_len - 1); clamped for empty sequences.
    """
    idx = gather_index.clamp(min=0)
    index = idx.view(-1, 1, 1).expand(-1, 1, output.size(-1))
    return output.gather(dim=1, index=index).squeeze(1)


class HyperHawkes(nn.Module):
    """
    Tripartite HyperHawkes: (u, v, w, t) -> (y_hat, h_u, h_v, h_w).

    Histories are built from training triplets only (events strictly before query time).
    """

    def __init__(
        self,
        train_data: TripartiteData,
        node_type_ids: np.ndarray,
        device: str,
        hidden_size: int = 64,
        max_seq_length: int = 50,
        num_heads: int = 4,
        n_levels: int = 2,
        hgnn_layers: int = 3,
        dropout: float = 0.2,
        attn_dropout_prob: float = 0.2,
        emb_dropout_prob: float = 0.2,
        layer_norm_eps: float = 1e-12,
        sub_time_delta: float = 3600.0,
        day_factor: float = 100.0,
        min_support: float = 0.0005,
        n_clusters: int = 16,
        temp_cluster: float = 0.03,
        use_hgnn: bool = True,
        use_shortterm: bool = True,
        use_base_excitation: bool = True,
        use_self_intent_excitation: bool = True,
    ):
        super().__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.node_feat_dim = hidden_size
        self.max_seq_length = max_seq_length
        self.sub_time_delta = sub_time_delta
        self.time_scalar = 60.0 * 60.0 * 24.0 * day_factor
        self.n_clusters = n_clusters
        self.temp_cluster = temp_cluster
        self.use_hgnn = use_hgnn
        self.use_shortterm = use_shortterm
        self.use_base_excitation = use_base_excitation
        self.use_self_intent_excitation = use_self_intent_excitation

        user_globals = np.where(node_type_ids == TYPE_USER)[0]
        streamer_globals = np.where(node_type_ids == TYPE_STREAMER)[0]
        room_globals = np.where(node_type_ids == TYPE_ROOM)[0]
        self.n_users = len(user_globals)
        self.n_streamers = len(streamer_globals)
        self.n_rooms = len(room_globals)
        self.n_items = self.n_streamers + 1  # streamer locals; 0 = pad

        self.user_global_to_local = {int(g): i for i, g in enumerate(user_globals)}
        self.streamer_global_to_local = {int(g): i + 1 for i, g in enumerate(streamer_globals)}
        self.room_global_to_local = {int(g): i for i, g in enumerate(room_globals)}
        self._build_user_histories(train_data)

        self.item_embedding = nn.Embedding(self.n_items, hidden_size, padding_idx=0)
        self.room_embedding = nn.Embedding(max(self.n_rooms, 1), hidden_size)
        if use_base_excitation or use_self_intent_excitation:
            self.user_embedding = nn.Embedding(self.n_users, hidden_size)

        self.register_buffer("cluster_prob", torch.zeros(self.n_items, n_clusters))

        if use_self_intent_excitation:
            self.global_alpha = nn.Parameter(torch.tensor(0.0))
            self.intent_dist = nn.Sequential(
                nn.Linear(n_clusters + hidden_size * 2, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 5),
            )

        if use_hgnn and hgnn_layers > 0 and use_self_intent_excitation:
            u_loc, s_loc, times = self._train_bipartite_arrays(train_data)
            bipartite = build_bipartite_from_arrays(u_loc, s_loc, times)
            self.hyper_edge_index, self.hyper_edge_weight = construct_global_hyper_graph(
                bipartite, sub_time_delta, min_support, self.n_items
            )
            self.hgnn = HGNN(hidden_size, n_layers=hgnn_layers)
        else:
            self.register_buffer("hyper_edge_index", torch.zeros((2, 2), dtype=torch.long))
            self.register_buffer("hyper_edge_weight", torch.ones(1))
            self.hgnn = None

        if use_shortterm:
            self.local_encoder = AttentionMixer(
                hidden_size=hidden_size,
                levels=n_levels,
                n_heads=num_heads,
                dropout=attn_dropout_prob,
            )
            self.seq_layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
            self.last_linear = nn.Linear(hidden_size * 2, hidden_size, bias=True)

        self.emb_dropout_prob = emb_dropout_prob
        hid = max(hidden_size, hidden_size)
        self.pred_head = nn.Sequential(
            nn.Linear(3 * hidden_size, hid),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hid, 1),
        )

        self._init_parameters()

    def _init_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def _build_user_histories(self, train_data: TripartiteData) -> None:
        per_user: dict[int, list[tuple[float, int, int]]] = defaultdict(list)
        for u, v, w, t in zip(
            train_data.user_node_ids,
            train_data.streamer_node_ids,
            train_data.item_node_ids,
            train_data.node_interact_times,
        ):
            ul = self.user_global_to_local.get(int(u))
            sl = self.streamer_global_to_local.get(int(v))
            rl = self.room_global_to_local.get(int(w))
            if ul is None or sl is None or rl is None:
                continue
            per_user[ul].append((float(t), sl, rl))

        self.user_hist_times: list[np.ndarray] = []
        self.user_hist_streamers: list[np.ndarray] = []
        self.user_hist_rooms: list[np.ndarray] = []
        for ul in range(self.n_users):
            events = per_user.get(ul, [])
            events.sort(key=lambda x: x[0])
            if events:
                ts, ss, rs = zip(*events)
                self.user_hist_times.append(np.asarray(ts, dtype=np.float64))
                self.user_hist_streamers.append(np.asarray(ss, dtype=np.int64))
                self.user_hist_rooms.append(np.asarray(rs, dtype=np.int64))
            else:
                self.user_hist_times.append(np.zeros(0, dtype=np.float64))
                self.user_hist_streamers.append(np.zeros(0, dtype=np.int64))
                self.user_hist_rooms.append(np.zeros(0, dtype=np.int64))

    def _train_bipartite_arrays(self, train_data: TripartiteData):
        u_loc, s_loc, times = [], [], []
        for u, v, t in zip(train_data.user_node_ids, train_data.streamer_node_ids, train_data.node_interact_times):
            ul = self.user_global_to_local.get(int(u))
            sl = self.streamer_global_to_local.get(int(v))
            if ul is None or sl is None:
                continue
            u_loc.append(ul)
            s_loc.append(sl)
            times.append(float(t))
        return np.asarray(u_loc), np.asarray(s_loc), np.asarray(times, dtype=np.float64)

    def set_neighbor_sampler(self, neighbor_sampler) -> None:
        """No-op: HyperHawkes uses precomputed user histories, not NeighborSampler."""

    def e_step(self) -> None:
        """Refresh soft cluster assignments on streamer embeddings (call once per epoch)."""
        if not self.use_self_intent_excitation:
            return
        try:
            from torch_kmeans import SoftKMeans
        except ImportError:
            logger.warning("torch_kmeans not installed; skipping HyperHawkes e_step")
            return

        with torch.no_grad():
            if self.hgnn is not None:
                item_embs = self.hgnn(
                    self.item_embedding.weight.detach(),
                    self.hyper_edge_index.to(self.device),
                    self.hyper_edge_weight.to(self.device),
                )
            else:
                item_embs = self.item_embedding.weight.detach()

            model = SoftKMeans(
                n_clusters=self.n_clusters,
                init_method="k-means++",
                normalize="unit",
                temp=1.0 / self.temp_cluster,
                verbose=False,
            )
            result = model(item_embs.unsqueeze(0), k=self.n_clusters)
            self.cluster_prob.copy_(result.soft_assignment.squeeze().to(self.device))

    def _batch_sequences(
        self,
        user_globals: np.ndarray,
        interact_times: np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        b = len(user_globals)
        dev = self.device
        item_seq = torch.zeros(b, self.max_seq_length, dtype=torch.long, device=dev)
        time_seq = torch.zeros(b, self.max_seq_length, dtype=torch.float32, device=dev)
        item_seq_len = torch.zeros(b, dtype=torch.long, device=dev)

        for i, (ug, cut_t) in enumerate(zip(user_globals, interact_times)):
            ul = self.user_global_to_local.get(int(ug))
            if ul is None:
                continue
            ts = self.user_hist_times[ul]
            if ts.size == 0:
                continue
            end = int(np.searchsorted(ts, float(cut_t), side="left"))
            if end <= 0:
                continue
            sl = self.user_hist_streamers[ul][:end]
            tt = self.user_hist_times[ul][:end]
            take = sl[-self.max_seq_length :]
            t_take = tt[-self.max_seq_length :]
            n = len(take)
            item_seq[i, -n:] = torch.from_numpy(take).to(dev)
            time_seq[i, -n:] = torch.from_numpy(t_take.astype(np.float32)).to(dev)
            item_seq_len[i] = n

        return item_seq, time_seq, item_seq_len, item_seq

    def _refined_item_embs(self) -> torch.Tensor:
        if self.hgnn is not None:
            return self.hgnn(
                self.item_embedding.weight,
                self.hyper_edge_index.to(self.device),
                self.hyper_edge_weight.to(self.device),
            )
        return self.item_embedding.weight

    def get_user_rep(
        self,
        user_id: torch.Tensor,
        item_seq: torch.Tensor,
        target_item: torch.Tensor,
    ) -> torch.Tensor:
        mask = item_seq.gt(0).unsqueeze(2)
        mask = torch.where(mask, 0.0, -10000.0)
        user_emb = self.user_embedding(user_id)
        item_seq_emb = self.item_embedding(item_seq)
        target_item_emb = self.item_embedding(target_item)
        user_emb = F.dropout(user_emb, self.emb_dropout_prob, training=self.training)
        item_seq_emb = F.dropout(item_seq_emb, self.emb_dropout_prob, training=self.training)
        target_item_emb = F.dropout(target_item_emb, self.emb_dropout_prob, training=self.training)
        attn_scores = torch.matmul(item_seq_emb, target_item_emb.unsqueeze(2))
        attn_probs = torch.softmax(attn_scores + mask, dim=1).transpose(2, 1)
        attn_user_rep = torch.matmul(attn_probs, item_seq_emb).squeeze(1)
        return attn_user_rep + user_emb

    def _shortterm(self, item_seq: torch.Tensor, item_seq_len: torch.Tensor) -> torch.Tensor:
        item_seq_emb = self.item_embedding(item_seq)
        item_seq_emb = self.seq_layer_norm(item_seq_emb)
        item_seq_emb = F.dropout(item_seq_emb, self.emb_dropout_prob, training=self.training)
        last_idx = (item_seq_len - 1).clamp(min=0)
        last_item_emb = _gather_indexes(item_seq_emb, last_idx)
        seq_output = self.local_encoder(item_seq, item_seq_emb, item_seq_len)
        return self.last_linear(torch.cat([seq_output, last_item_emb], dim=1))

    def intent_excitation(
        self,
        item: torch.Tensor,
        item_seq: torch.Tensor,
        time: torch.Tensor,
        time_seq: torch.Tensor,
        user_rep: torch.Tensor,
    ) -> torch.Tensor:
        if self.cluster_prob.sum() <= 0:
            return torch.zeros(item.size(0), device=self.device)
        target_cluster_probs = self.cluster_prob[item]
        seq_cluster_probs = self.cluster_prob[item_seq]
        kl_div = F.kl_div(
            seq_cluster_probs.log(),
            target_cluster_probs.log().unsqueeze(1),
            reduction="none",
            log_target=True,
        ).sum(dim=2)
        intent_mask = (kl_div < 1e-12) * item_seq.gt(0)
        delta_t = (time.reshape(-1, 1) - time_seq)
        delta_mask = (delta_t > self.sub_time_delta) * intent_mask
        delta_t = (delta_t / self.time_scalar) * delta_mask
        delta_t = torch.min(delta_t + (~delta_mask) * 1e9, dim=1).values * delta_mask.sum(dim=1).bool().float()
        mask = (delta_t > 0).float()
        item_emb = self.item_embedding(item)
        dist_params = self.intent_dist(torch.cat((target_cluster_probs, item_emb, user_rep), dim=1))
        mus = dist_params[:, 0].clamp(min=1e-10, max=10)
        sigmas = dist_params[:, 1].clamp(min=1e-10, max=10)
        alphas = self.global_alpha + dist_params[:, 2]
        betas = (dist_params[:, 3] + 1).clamp(min=1e-10, max=10)
        pis = (dist_params[:, 4] + 0.5).clamp(min=1e-10, max=1)
        exp_dist = torch.distributions.exponential.Exponential(betas, validate_args=False)
        norm_dist = torch.distributions.normal.Normal(mus, sigmas)
        excitation = pis * exp_dist.log_prob(delta_t).exp() + (1 - pis) * norm_dist.log_prob(delta_t).exp()
        return alphas * excitation * mask

    def _encode_batch(
        self,
        user_globals: np.ndarray,
        streamer_globals: np.ndarray,
        room_globals: np.ndarray,
        interact_times: np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        item_seq, time_seq, item_seq_len, _ = self._batch_sequences(user_globals, interact_times)
        streamer_local = torch.tensor(
            [self.streamer_global_to_local.get(int(v), 0) for v in streamer_globals],
            dtype=torch.long,
            device=self.device,
        )
        room_local = torch.tensor(
            [self.room_global_to_local.get(int(w), 0) for w in room_globals],
            dtype=torch.long,
            device=self.device,
        )
        user_local = torch.tensor(
            [self.user_global_to_local.get(int(u), 0) for u in user_globals],
            dtype=torch.long,
            device=self.device,
        )
        time = torch.from_numpy(interact_times.astype(np.float32)).to(self.device)

        refined = self._refined_item_embs()
        h_v = refined[streamer_local]
        h_w = self.room_embedding(room_local)

        parts: list[torch.Tensor] = []
        if self.use_shortterm:
            parts.append(self._shortterm(item_seq, item_seq_len))

        if self.use_base_excitation or self.use_self_intent_excitation:
            user_rep = self.get_user_rep(user_local, item_seq, streamer_local)
            if self.use_base_excitation:
                parts.append(user_rep)
            if self.use_self_intent_excitation and self.cluster_prob.sum() > 0:
                exc = self.intent_excitation(streamer_local, item_seq, time, time_seq, user_rep)
                parts.append(exc.unsqueeze(-1).expand(-1, self.hidden_size))

        if parts:
            h_u = torch.stack(parts, dim=0).sum(dim=0)
        else:
            h_u = torch.zeros(len(user_globals), self.hidden_size, device=self.device)

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
        del edge_ids, edges_are_positive
        h_u, h_v, h_w = self._encode_batch(user_node_ids, streamer_node_ids, item_node_ids, interact_times)
        h = torch.cat([h_u, h_v, h_w], dim=-1)
        y_hat = torch.sigmoid(self.pred_head(h))
        return y_hat, h_u, h_v, h_w
