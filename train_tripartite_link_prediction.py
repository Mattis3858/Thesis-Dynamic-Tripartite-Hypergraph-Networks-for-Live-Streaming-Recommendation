"""
Tripartite (User, Streamer, Room) link prediction: HTTransformer or DyGLib baselines via BaselineTripartiteWrapper.

- Training: BCE or BPR with ``--train_neg_ratio`` random negatives (v', w') per positive (default 1).
- Each epoch: fast validation BCE on val_data (same 1-negative protocol); early stopping minimizes val_loss.
- After training: load best checkpoint, run full 1+99 ranking on test only (see utils.metrics).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import time
import warnings
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from models.BaselineTripartiteWrapper import BaselineTripartiteWrapper, TRIPARTITE_BASELINE_MODELS
from models.HTTransformer import HTTransformer
from utils.DataLoader import TripartiteData, get_idx_data_loader, get_tripartite_link_prediction_data
from utils.EarlyStopping import EarlyStopping
from utils.metrics import (
    TRIPARTITE_RANKING_KS,
    get_link_prediction_metrics,
    mean_metric_dicts,
    tripartite_ranking_metrics_per_query,
)
from utils.eval_candidates import (
    EvalCandidateConfig,
    build_eval_candidates,
    load_eval_candidates,
    save_eval_candidates,
)
from utils.utils import convert_to_gpu, create_optimizer, get_neighbor_sampler, get_parameter_sizes, set_random_seed
from utils.utils import get_tripartite_neighbor_sampler


def get_tripartite_train_args():
    parser = argparse.ArgumentParser("Interface for tripartite link prediction")
    parser.add_argument(
        "--model_name",
        type=str,
        default="HTTransformer",
        choices=[
            "HTTransformer",
            "HyperHawkes",
            "LightGCN",
            "HAN",
            "DyGFormer",
            "TGAT",
            "GraphMixer",
            "CAWN",
            "TCL",
            "TGN",
        ],
        help="HTTransformer / HyperHawkes / LightGCN / HAN, or a DyGLib temporal encoder for tripartite (u,v,w) scoring.",
    )
    g_ht = parser.add_mutually_exclusive_group()
    g_ht.add_argument("--use_bias_gate", dest="use_bias_gate", action="store_true", help="HTTransformer: bias-aware merge head (default on).")
    g_ht.add_argument("--no-use_bias_gate", dest="use_bias_gate", action="store_false", help="HTTransformer: plain concat + MLP head.")
    parser.set_defaults(use_bias_gate=True)
    g_ti = parser.add_mutually_exclusive_group()
    g_ti.add_argument("--use_type_init", dest="use_type_init", action="store_true", help="HTTransformer: type-aware node init (default on).")
    g_ti.add_argument("--no-use_type_init", dest="use_type_init", action="store_false", help="HTTransformer: linear on raw features only.")
    parser.set_defaults(use_type_init=True)
    g_co = parser.add_mutually_exclusive_group()
    g_co.add_argument("--use_hetero_coocc", dest="use_hetero_coocc", action="store_true", help="HTTransformer: 3D hetero co-occurrence (default on).")
    g_co.add_argument(
        "--no-use_hetero_coocc",
        dest="use_hetero_coocc",
        action="store_false",
        help="HTTransformer: homogeneous co-occurrence (one shared MLP on c_u,c_v,c_w, outputs summed).",
    )
    parser.set_defaults(use_hetero_coocc=True)
    parser.add_argument(
        "--train_neg_ratio",
        type=int,
        default=1,
        help="Training negatives per positive triplet (1 = legacy 1:1 BCE/BPR pair).",
    )
    parser.add_argument(
        "--train_neg_sampling",
        type=str,
        default="uniform",
        choices=["uniform", "popularity"],
        help="Training negative (v',w') sampler. 'uniform' (default): uniform over the "
        "streamer/room pools (legacy). 'popularity': sample observed (streamer,room) combos in "
        "proportion to their train-split frequency, aligning training negatives with the "
        "popularity component of the fixed eval protocol (utils.eval_candidates). Adds a '_popneg' "
        "checkpoint suffix so it never overwrites a uniform-neg run.",
    )
    parser.add_argument(
        "--train_neg_power",
        type=float,
        default=1.0,
        help="Exponent applied to combo counts for popularity sampling (1.0 = raw frequency; "
        "0.75 = word2vec-style flattening that upweights tail combos). Only used when "
        "--train_neg_sampling popularity.",
    )
    parser.add_argument(
        "--val_neg_sampling",
        type=str,
        default="popularity",
        choices=["uniform", "popularity"],
        help="Validation (early-stopping) negative (v',w') sampler for the 1-negative BCE. "
        "'popularity' (default): sample observed (streamer,room) combos in proportion to their "
        "TRAIN-split frequency, so the checkpoint-selection signal aligns with the popularity "
        "component of the fixed eval protocol -- same cost as the uniform 1-neg validation. "
        "'uniform' (legacy): uniform over the streamer/room pools.",
    )
    parser.add_argument(
        "--fusion_mode",
        type=str,
        default="concat",
        choices=["concat", "mean"],
        help="HTTransformer prediction head: concat [h_u||h_v||h_w] (default) or mean (h_u+h_v+h_w)/3.",
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        default="bce",
        choices=["bce", "bpr"],
        help="Training loss: BCE on pos/neg scores (default) or BPR -log sigmoid(pos-neg).",
    )
    parser.add_argument("--dataset_name", type=str, default="kuailive_tripartite")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="資料目錄（含 ml_* 或原始 ml_kuailive_edges.csv 等）；預設為 <專案根>/processed_data/{dataset_name}",
    )
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--num_neighbors",
        type=int,
        default=20,
        help="Sampled neighbors for TGAT/TCL/CAWN/GraphMixer; GraphMixer also uses this as num_tokens.",
    )
    parser.add_argument(
        "--sample_neighbor_strategy",
        type=str,
        default="recent",
        choices=["uniform", "recent", "time_interval_aware"],
    )
    parser.add_argument("--time_scaling_factor", default=1e-6, type=float)
    parser.add_argument("--num_heads", type=int, default=2)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--time_feat_dim", type=int, default=100)
    parser.add_argument("--patch_size", type=int, default=1)
    parser.add_argument("--channel_embedding_dim", type=int, default=50, help="hidden dim d in HT-Transformer")
    parser.add_argument("--cooccurrence_dim", type=int, default=50, help="d_C for 3D co-occurrence branch")
    parser.add_argument("--max_input_sequence_length", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--optimizer", type=str, default="Adam", choices=["SGD", "Adam", "RMSprop"])
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--num_runs", type=int, default=5)
    parser.add_argument(
        "--test_interval_epochs",
        type=int,
        default=10,
        help="Unused: mid-epoch test ranking removed; final test ranking runs once after early stopping.",
    )
    parser.add_argument("--num_ranking_negatives", type=int, default=99, help="99 negatives + 1 positive = 100 candidates")
    parser.add_argument("--eval_seed", type=int, default=0, help="seed for ranking negative sampling in val/test")

    # --- fixed, shared test-candidate set (new evaluation protocol; see utils.eval_candidates) ---
    g_fc = parser.add_mutually_exclusive_group()
    g_fc.add_argument(
        "--use_fixed_eval_candidates",
        dest="use_fixed_eval_candidates",
        action="store_true",
        help="Test ranking against a pre-generated, model-independent candidate set (default on). "
        "This is the fair, harder protocol; all models read the same negatives.",
    )
    g_fc.add_argument(
        "--no-use_fixed_eval_candidates",
        dest="use_fixed_eval_candidates",
        action="store_false",
        help="Legacy protocol: sample uniform negatives on the fly at scoring time (for old-vs-new comparison).",
    )
    parser.set_defaults(use_fixed_eval_candidates=True)
    parser.add_argument(
        "--eval_candidates_path",
        type=str,
        default=None,
        help="Path to the candidate .npz. Default: eval_candidates/{dataset}_test_{setting}_seed{S}.npz. "
        "Built once if missing, then reused (all models/ablations must point at the same file).",
    )
    parser.add_argument("--neg_ratio_popularity", type=float, default=0.5, help="Fraction of negatives from popularity-as-of-t.")
    parser.add_argument("--neg_ratio_hard", type=float, default=0.3, help="Fraction from user-history hard negatives.")
    parser.add_argument("--neg_ratio_uniform", type=float, default=0.2, help="Fraction from uniform sampling.")
    parser.add_argument(
        "--neg_popularity_window",
        type=float,
        default=None,
        help="Popularity time window ending at t (dataset ts units). Default: all history before t. "
        "TODO(GPU): tune a finite window on full data.",
    )
    parser.add_argument(
        "--neg_candidate_universe",
        type=str,
        default="observed",
        choices=["observed", "product"],
        help="'observed' = only (streamer,room) combos that occur in data (harder/realistic); "
        "'product' = full streamer x room grid (legacy-style).",
    )
    parser.add_argument(
        "--neg_sampling_seed",
        type=int,
        default=None,
        help="Seed for candidate generation. Default: eval_seed + 1 (matches the legacy test-negative seed).",
    )
    parser.add_argument(
        "--enable_user_item_setting",
        action="store_true",
        help="Also build the user-item (replace-room-only) candidate set. Off by default: with a tiny "
        "room pool it cannot form distinct negatives and its numbers duplicate the main setting.",
    )
    parser.add_argument(
        "--early_stop_metric",
        type=str,
        default="val_loss",
        choices=["val_loss", "val_ndcg@5", "val_ndcg@10", "val_recall@10"],
        help="Model-selection metric. Default 'val_loss' (1-neg BCE, current behaviour). The ranking "
        "options monitor validation under the fixed-candidate protocol -- interface for the next "
        "phase (using it for the thesis means RE-TRAINING all models; not done this round).",
    )
    parser.add_argument(
        "--log_val_ranking",
        action="store_true",
        help="Diagnostic: each epoch, compute and LOG validation ranking metrics (ndcg@5/@10, "
        "recall@10) under the fixed-candidate protocol, WITHOUT using them for model selection "
        "(early stopping stays on --early_stop_metric). Use this to observe whether ranking peaks "
        "later than the val_loss-selected epoch before committing to a ranking-based early stop. "
        "Adds one subsampled ranking pass per epoch (see --val_ranking_subsample).",
    )
    parser.add_argument(
        "--val_ranking_subsample",
        type=int,
        default=2000,
        help="When --log_val_ranking (or ranking-based --early_stop_metric) is on, evaluate ranking "
        "on this many randomly-chosen (fixed across runs) val queries per epoch instead of all of "
        "them, to keep the per-epoch cost small. 0 = use the full val split.",
    )
    parser.add_argument(
        "--time_gap",
        type=int,
        default=2000,
        help="GraphMixer: time-gap neighbor count for the node branch (see GraphMixer.compute_node_temporal_embeddings).",
    )
    parser.add_argument("--walk_length", type=int, default=1, help="CAWN: random walk length")
    parser.add_argument("--position_feat_dim", type=int, default=172, help="CAWN: position feature dimension")
    parser.add_argument("--num_walk_heads", type=int, default=8, help="CAWN: walk aggregation heads")
    parser.add_argument(
        "--num_depths",
        type=int,
        default=None,
        help="TCL: depth embedding size (= num_neighbors + 1 in DyGLib). Default: num_neighbors + 1.",
    )
    parser.add_argument(
        "--metrics_out",
        type=str,
        default=None,
        help="If set, write a JSON summary (test metrics averaged over num_runs) for orchestration scripts.",
    )
    parser.add_argument(
        "--hyperhawkes_sub_time_delta",
        type=float,
        default=3600.0,
        help="HyperHawkes: session split threshold (seconds) for intent mining.",
    )
    parser.add_argument(
        "--hyperhawkes_min_support",
        type=float,
        default=0.0005,
        help="HyperHawkes: minimum support for frequent intent itemsets.",
    )
    parser.add_argument(
        "--hyperhawkes_day_factor",
        type=float,
        default=100.0,
        help="HyperHawkes: scales timestamps to day units in Hawkes excitation.",
    )
    parser.add_argument(
        "--lightgcn_n_layers",
        type=int,
        default=None,
        help="LightGCN: number of propagation layers (default: --num_layers).",
    )
    parser.add_argument(
        "--han_n_heads",
        type=int,
        default=None,
        help="HAN: number of GAT heads (default: --num_heads).",
    )
    parser.add_argument(
        "--han_gat_layers",
        type=int,
        default=1,
        help="HAN: node-level GAT layers per meta-path.",
    )
    parser.add_argument(
        "--han_meta_path_nhood",
        type=int,
        default=1,
        help="HAN: expand meta-path adjacency hops for attention mask (original HAN uses 1).",
    )
    args = parser.parse_args()
    if args.lightgcn_n_layers is None:
        args.lightgcn_n_layers = args.num_layers
    if args.han_n_heads is None:
        args.han_n_heads = args.num_heads
    args.device = f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu"
    # DyGLib TCL: depth_embedding covers target + num_neighbors hops (see train_link_prediction.py)
    if args.num_depths is None:
        args.num_depths = args.num_neighbors + 1
    elif args.model_name == "TCL" and args.num_depths < args.num_neighbors + 1:
        warnings.warn(
            f"TCL: num_depths={args.num_depths} < num_neighbors+1={args.num_neighbors + 1}; "
            f"using num_depths={args.num_neighbors + 1}.",
            stacklevel=2,
        )
        args.num_depths = args.num_neighbors + 1
    if args.train_neg_ratio < 1:
        raise ValueError(f"train_neg_ratio must be >= 1, got {args.train_neg_ratio}")
    if args.neg_sampling_seed is None:
        args.neg_sampling_seed = args.eval_seed + 1
    return args


def _file_sha256(path: str) -> str:
    """First 16 hex chars of a file's sha256 -- logged so two model runs can be confirmed to read
    the same candidate file."""
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _type_pools(node_type_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Global ids for streamers (type 2) and rooms (type 3)."""
    streamer_pool = np.where(node_type_ids == 2)[0].astype(np.int64)
    room_pool = np.where(node_type_ids == 3)[0].astype(np.int64)
    if len(streamer_pool) == 0 or len(room_pool) == 0:
        raise ValueError("node_type_ids must contain at least one streamer (2) and one room (3).")
    return streamer_pool, room_pool


def _sample_one_negative_pair(
    v_pos: int,
    w_pos: int,
    streamer_pool: np.ndarray,
    room_pool: np.ndarray,
    rng: np.random.RandomState,
) -> tuple[int, int]:
    for _ in range(50):
        vn = int(rng.choice(streamer_pool))
        wn = int(rng.choice(room_pool))
        if vn != v_pos or wn != w_pos:
            return vn, wn
    wn = int(rng.choice(room_pool))
    if wn == w_pos:
        wn = int(rng.choice(room_pool))
    return v_pos, wn


def _sample_negative_pairs(
    batch_v_pos: np.ndarray,
    batch_w_pos: np.ndarray,
    streamer_pool: np.ndarray,
    room_pool: np.ndarray,
    rng: np.random.RandomState,
    num_negatives: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample ``num_negatives`` negative (v', w') pairs per positive row."""
    if num_negatives < 1:
        raise ValueError(f"num_negatives must be >= 1, got {num_negatives}")
    b = len(batch_v_pos)
    if num_negatives == 1:
        v_neg = np.empty(b, dtype=np.int64)
        w_neg = np.empty(b, dtype=np.int64)
        for i in range(b):
            v_neg[i], w_neg[i] = _sample_one_negative_pair(
                int(batch_v_pos[i]), int(batch_w_pos[i]), streamer_pool, room_pool, rng
            )
        return v_neg, w_neg

    v_neg = np.empty((b, num_negatives), dtype=np.int64)
    w_neg = np.empty((b, num_negatives), dtype=np.int64)
    for i in range(b):
        vp, wp = int(batch_v_pos[i]), int(batch_w_pos[i])
        for j in range(num_negatives):
            v_neg[i, j], w_neg[i, j] = _sample_one_negative_pair(vp, wp, streamer_pool, room_pool, rng)
    return v_neg, w_neg


class _PopularityNegativeSampler:
    """
    Sample negative (v', w') combos in proportion to their frequency in ``train_data``.

    Aligns the training-negative distribution with the popularity component of the fixed
    evaluation protocol (utils.eval_candidates), without per-batch ``as-of-t`` windowing: the
    combo universe and weights are the **observed (streamer, room) pairs in the train split**
    (no val/test leakage), precomputed once. Sampling is vectorised (one ``rng.choice`` per
    batch), so it adds negligible time vs. the uniform sampler.

    Returns the same shapes as ``_sample_negative_pairs`` (1-D for ``num_negatives == 1``, else
    ``[b, num_negatives]``) and applies the same false-negative policy: only the exact positive
    combo ``(v_pos, w_pos)`` is rejected (resampled), matching the uniform sampler.
    """

    def __init__(self, train_data: TripartiteData, power: float = 1.0):
        combo_counter = Counter(
            zip(train_data.streamer_node_ids.tolist(), train_data.item_node_ids.tolist())
        )
        combos = sorted(combo_counter)  # deterministic order
        if not combos:
            raise ValueError("train_data has no (streamer, room) combos for popularity sampling.")
        self.combos = np.asarray(combos, dtype=np.int64).reshape(-1, 2)  # shape: [C, 2]
        self.num_combos = self.combos.shape[0]
        counts = np.array([combo_counter[c] for c in combos], dtype=np.float64)
        if power != 1.0:
            counts = counts**power
        self.prob = counts / counts.sum()  # shape: [C]
        self._combo_to_idx = {(int(v), int(w)): i for i, (v, w) in enumerate(combos)}

    def sample(
        self,
        batch_v_pos: np.ndarray,
        batch_w_pos: np.ndarray,
        rng: np.random.RandomState,
        num_negatives: int = 1,
    ) -> tuple[np.ndarray, np.ndarray]:
        if num_negatives < 1:
            raise ValueError(f"num_negatives must be >= 1, got {num_negatives}")
        b = len(batch_v_pos)
        total = b * num_negatives
        idx = rng.choice(self.num_combos, size=total, p=self.prob)  # shape: [total]
        pos_idx = np.repeat(
            np.array(
                [self._combo_to_idx.get((int(batch_v_pos[i]), int(batch_w_pos[i])), -1) for i in range(b)],
                dtype=np.int64,
            ),
            num_negatives,
        )
        # resample any draw that collided with its own positive combo (rare; popular combos collide more)
        for j in np.where(idx == pos_idx)[0]:
            for _ in range(50):
                ci = int(rng.choice(self.num_combos, p=self.prob))
                if ci != pos_idx[j]:
                    idx[j] = ci
                    break
        sel = self.combos[idx]  # shape: [total, 2]
        v_neg = sel[:, 0].astype(np.int64)
        w_neg = sel[:, 1].astype(np.int64)
        if num_negatives == 1:
            return v_neg, w_neg
        return v_neg.reshape(b, num_negatives), w_neg.reshape(b, num_negatives)


def _forward_pos_neg_scores(
    model: nn.Module,
    bu: np.ndarray,
    bv: np.ndarray,
    bw: np.ndarray,
    bt: np.ndarray,
    bv_neg: np.ndarray,
    bw_neg: np.ndarray,
    batch_edge_ids: np.ndarray,
    *,
    tgn: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns y_pos [batch_size] and y_neg [batch_size, num_negatives].
    """
    if bv_neg.ndim == 1:
        num_neg = 1
        bv_neg_flat = bv_neg
        bw_neg_flat = bw_neg
    else:
        num_neg = bv_neg.shape[1]
        batch_size = len(bu)
        bv_neg_flat = bv_neg.reshape(-1)
        bw_neg_flat = bw_neg.reshape(-1)
        bu_neg = np.repeat(bu, num_neg)
        bt_neg = np.repeat(bt, num_neg)
    if tgn:
        if num_neg == 1:
            y_neg, _, _, _ = model(bu, bv_neg_flat, bw_neg_flat, bt, edge_ids=None, edges_are_positive=False)
            y_pos, _, _, _ = model(
                bu, bv, bw, bt, edge_ids=batch_edge_ids, edges_are_positive=True
            )
            return y_pos.view(-1), y_neg.view(-1).unsqueeze(1)
        y_pos, _, _, _ = model(bu, bv, bw, bt, edge_ids=batch_edge_ids, edges_are_positive=True)
        y_neg, _, _, _ = model(
            bu_neg, bv_neg_flat, bw_neg_flat, bt_neg, edge_ids=None, edges_are_positive=False
        )
        return y_pos.view(-1), y_neg.view(-1).reshape(len(bu), num_neg)

    y_pos, _, _, _ = model(bu, bv, bw, bt, edge_ids=batch_edge_ids, edges_are_positive=True)
    if num_neg == 1:
        y_neg, _, _, _ = model(bu, bv_neg_flat, bw_neg_flat, bt, edge_ids=None, edges_are_positive=False)
        return y_pos.view(-1), y_neg.view(-1).unsqueeze(1)
    y_neg, _, _, _ = model(
        bu_neg, bv_neg_flat, bw_neg_flat, bt_neg, edge_ids=None, edges_are_positive=False
    )
    return y_pos.view(-1), y_neg.view(-1).reshape(len(bu), num_neg)


def _compute_tripartite_training_loss(
    y_pos: torch.Tensor,
    y_neg: torch.Tensor,
    *,
    loss_type: str,
    loss_fn: nn.Module | None,
    device: str,
) -> torch.Tensor:
    """y_pos [B]; y_neg [B, K]. BCE or BPR over all pos–neg pairs in the batch."""
    y_pos = y_pos.view(-1)
    if y_neg.dim() == 1:
        y_neg = y_neg.unsqueeze(1)

    if loss_type == "bce":
        if loss_fn is None:
            raise ValueError("BCE training requires loss_fn (nn.BCELoss).")
        predicts = torch.cat([y_pos, y_neg.reshape(-1)], dim=0)
        labels = torch.cat(
            [
                torch.ones_like(y_pos, device=device, dtype=torch.float32),
                torch.zeros(y_neg.numel(), device=device, dtype=torch.float32),
            ],
            dim=0,
        )
        return loss_fn(predicts.float(), labels)

    if loss_type == "bpr":
        diff = y_pos.unsqueeze(1) - y_neg
        return (-torch.log(torch.sigmoid(diff) + 1e-10)).mean()

    raise ValueError(f"Unknown loss_type: {loss_type}")


def _training_metric_tensors(y_pos: torch.Tensor, y_neg: torch.Tensor, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Flatten pos/neg scores and binary labels for diagnostic AUC/AP during training."""
    y_pos = y_pos.view(-1)
    if y_neg.dim() == 1:
        y_neg = y_neg.unsqueeze(1)
    predicts = torch.cat([y_pos, y_neg.reshape(-1)], dim=0)
    labels = torch.cat(
        [
            torch.ones_like(y_pos, device=device, dtype=torch.float32),
            torch.zeros(y_neg.numel(), device=device, dtype=torch.float32),
        ],
        dim=0,
    )
    return predicts, labels


def _is_tgn_model(model: nn.Module) -> bool:
    return getattr(model, "model_name", None) == "TGN"


def _sort_batch_indices_by_time(idx: np.ndarray, interact_times: np.ndarray) -> np.ndarray:
    """TGN memory updates require non-decreasing times within each batch."""
    return idx[np.argsort(interact_times[idx], kind="stable")]


def _train_tripartite_node_set(train_data: TripartiteData) -> set[int]:
    """Any global id that appears in train triplets (any role)."""
    s = set(np.unique(train_data.user_node_ids).astype(np.int64).tolist())
    s.update(np.unique(train_data.streamer_node_ids).astype(np.int64).tolist())
    s.update(np.unique(train_data.item_node_ids).astype(np.int64).tolist())
    return s


def _cold_start_bucket(u: int, v: int, w: int, train_nodes: set[int]) -> str | None:
    """Exactly-one-new-node buckets; mutual exclusion."""
    ui, vi, wi = u in train_nodes, v in train_nodes, w in train_nodes
    if (not ui) and vi and wi:
        return "new_u_old_v_old_w"
    if ui and (not vi) and wi:
        return "old_u_new_v_old_w"
    if ui and vi and (not wi):
        return "old_u_old_v_new_w"
    return None


def _cold_start_public_metrics(agg: dict) -> dict:
    """AUC, P@10, R@10, N@10 for tables."""
    keys = ("roc_auc", "precision@10", "recall@10", "ndcg@10")
    return {k: float(agg.get(k, float("nan"))) for k in keys}


def evaluate_tripartite_cold_start_by_split(
    train_data: TripartiteData,
    test_data: TripartiteData,
    per_query_metrics: list[dict],
) -> dict[str, dict]:
    """
    Split test ranking metrics into three cold-start subsets (exactly one endpoint unseen in train).
    """
    if len(per_query_metrics) != len(test_data.user_node_ids):
        raise ValueError("per_query_metrics length must match test_data size.")
    train_nodes = _train_tripartite_node_set(train_data)
    buckets: dict[str, list[dict]] = {
        "new_u_old_v_old_w": [],
        "old_u_new_v_old_w": [],
        "old_u_old_v_new_w": [],
    }
    for i, m in enumerate(per_query_metrics):
        u = int(test_data.user_node_ids[i])
        v = int(test_data.streamer_node_ids[i])
        w = int(test_data.item_node_ids[i])
        b = _cold_start_bucket(u, v, w, train_nodes)
        if b is not None:
            buckets[b].append(m)
    out: dict[str, dict] = {}
    for name, lst in buckets.items():
        out[name] = _cold_start_public_metrics(mean_metric_dicts(lst)) if lst else _cold_start_public_metrics({})
    return out


def _empty_ranking_metrics() -> dict:
    """NaN-filled metric dict (same keys as a real one) for queries with < 2 candidates."""
    d: dict[str, float] = {"roc_auc": float("nan"), "average_precision": float("nan")}
    for k in TRIPARTITE_RANKING_KS:
        d[f"recall@{k}"] = float("nan")
        d[f"precision@{k}"] = float("nan")
        d[f"ndcg@{k}"] = float("nan")
    return d


@torch.no_grad()
def _rank_from_fixed_candidates(
    model: nn.Module,
    data: TripartiteData,
    idx_data_loader,
    eval_candidates,
    device: str,
    return_per_query: bool,
) -> tuple[float, dict] | tuple[float, dict, list[dict]]:
    """
    Score the pre-generated, model-independent candidate set (see utils.eval_candidates).

    Per query: candidate 0 is the positive (v_pos, w_pos), candidates 1.. are the stored
    negatives (variable count per query -- a small dev subset may not fill N). All candidates of
    a batch are flattened into one forward call, then split back per query for ranking metrics.
    Loader order must be sequential so per-query metrics align with data rows (cold-start /
    group-wise subsetting relies on this).
    """
    loss_fn = nn.BCELoss()
    neg_v, neg_w, neg_count = eval_candidates.neg_v, eval_candidates.neg_w, eval_candidates.neg_count

    all_query_metrics: list[dict] = []
    bce_losses: list[float] = []

    for batch_indices in tqdm(idx_data_loader, ncols=120, desc="eval ranking (fixed cand)"):
        idx = batch_indices.numpy()
        flat_u: list[int] = []
        flat_v: list[int] = []
        flat_w: list[int] = []
        flat_t: list[float] = []
        lengths: list[int] = []
        for i in idx:
            i = int(i)
            m = int(neg_count[i])
            cv = [int(data.streamer_node_ids[i])] + neg_v[i, :m].tolist()
            cw = [int(data.item_node_ids[i])] + neg_w[i, :m].tolist()
            L = len(cv)
            flat_u.extend([int(data.user_node_ids[i])] * L)
            flat_t.extend([float(data.node_interact_times[i])] * L)
            flat_v.extend(cv)
            flat_w.extend(cw)
            lengths.append(L)

        y_hat, _, _, _ = model(
            np.asarray(flat_u, dtype=np.int64),
            np.asarray(flat_v, dtype=np.int64),
            np.asarray(flat_w, dtype=np.int64),
            np.asarray(flat_t, dtype=np.float64),
            edge_ids=None,
            edges_are_positive=False,
        )
        y_hat = y_hat.view(-1).float().cpu().numpy()

        offset = 0
        batch_pos: list[float] = []
        batch_neg: list[float] = []
        for L in lengths:
            s = y_hat[offset : offset + L]
            offset += L
            if L >= 2:
                all_query_metrics.append(tripartite_ranking_metrics_per_query(s, positive_index=0))
            else:
                all_query_metrics.append(_empty_ranking_metrics())
            batch_pos.append(float(s[0]))
            if L > 1:
                batch_neg.extend(s[1:].tolist())

        if batch_pos:
            pos_scores = torch.tensor(batch_pos, dtype=torch.float32, device=device)
            neg_scores = torch.tensor(batch_neg if batch_neg else [0.0], dtype=torch.float32, device=device)
            bce = loss_fn(
                torch.cat([pos_scores, neg_scores], dim=0),
                torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)], dim=0),
            )
            bce_losses.append(float(bce.item()))

    mean_bce = float(np.mean(bce_losses)) if bce_losses else float("nan")
    agg = mean_metric_dicts(all_query_metrics)
    if return_per_query:
        return mean_bce, agg, all_query_metrics
    return mean_bce, agg


@torch.no_grad()
def evaluate_tripartite_ranking(
    model: nn.Module,
    neighbor_sampler,
    data: TripartiteData,
    idx_data_loader,
    node_type_ids: np.ndarray,
    device: str,
    num_negatives: int,
    eval_rng: np.random.RandomState,
    return_per_query: bool = False,
    eval_candidates=None,
) -> tuple[float, dict] | tuple[float, dict, list[dict]]:
    """
    Rank 1 positive against negatives per query; return diagnostic BCE + aggregated ranking metrics.

    Two modes:
      * ``eval_candidates`` given (new protocol): score the fixed, model-independent candidate set
        -- ``num_negatives`` / ``eval_rng`` are ignored. This is what makes cross-model comparison
        fair (every model reads the same negatives). See utils.eval_candidates.
      * ``eval_candidates is None`` (legacy protocol): sample ``num_negatives`` uniform (v', w')
        on the fly with ``eval_rng``. Kept for the old-vs-new comparison.

    If return_per_query, also returns metrics per row (loader order must match sequential indices).
    """
    model.eval()
    model.set_neighbor_sampler(neighbor_sampler)

    if eval_candidates is not None:
        return _rank_from_fixed_candidates(model, data, idx_data_loader, eval_candidates, device, return_per_query)

    streamer_pool, room_pool = _type_pools(node_type_ids)
    loss_fn = nn.BCELoss()

    all_query_metrics: list[dict] = []
    bce_losses: list[float] = []

    for batch_indices in tqdm(idx_data_loader, ncols=120, desc="eval ranking"):
        idx = batch_indices.numpy()
        bu = data.user_node_ids[idx]
        bv = data.streamer_node_ids[idx]
        bw = data.item_node_ids[idx]
        bt = data.node_interact_times[idx]
        batch_size = len(bu)
        num_cand = num_negatives + 1

        u_all = np.repeat(bu, num_cand)
        t_all = np.repeat(bt, num_cand)
        v_all = np.zeros(batch_size * num_cand, dtype=np.int64)
        w_all = np.zeros(batch_size * num_cand, dtype=np.int64)

        offset = 0
        for i in range(batch_size):
            v_p = int(bv[i])
            w_p = int(bw[i])
            v_all[offset] = v_p
            w_all[offset] = w_p
            offset += 1
            for _j in range(num_negatives):
                for _ in range(50):
                    vn = int(eval_rng.choice(streamer_pool))
                    wn = int(eval_rng.choice(room_pool))
                    if vn != v_p or wn != w_p:
                        v_all[offset] = vn
                        w_all[offset] = wn
                        offset += 1
                        break
                else:
                    v_all[offset] = v_p
                    w_all[offset] = int(eval_rng.choice(room_pool))
                    if v_all[offset] == v_p and w_all[offset] == w_p:
                        w_all[offset] = int(eval_rng.choice(room_pool))
                    offset += 1

        y_hat, _, _, _ = model(
            u_all, v_all, w_all, t_all, edge_ids=None, edges_are_positive=False
        )
        y_hat = y_hat.view(-1).float()
        scores_mat = y_hat.cpu().numpy().reshape(batch_size, num_cand)

        pos_scores = torch.from_numpy(scores_mat[:, 0]).to(device)
        neg_scores = torch.from_numpy(scores_mat[:, 1:].reshape(-1)).to(device)
        labels_pos = torch.ones_like(pos_scores)
        labels_neg = torch.zeros_like(neg_scores)
        bce = loss_fn(
            torch.cat([pos_scores, neg_scores], dim=0),
            torch.cat([labels_pos, labels_neg], dim=0),
        )
        bce_losses.append(float(bce.item()))

        for row in range(batch_size):
            all_query_metrics.append(tripartite_ranking_metrics_per_query(scores_mat[row], positive_index=0))

    mean_bce = float(np.mean(bce_losses))
    agg = mean_metric_dicts(all_query_metrics)
    if return_per_query:
        return mean_bce, agg, all_query_metrics
    return mean_bce, agg


def train_one_epoch(
    model: nn.Module,
    neighbor_sampler,
    train_data: TripartiteData,
    train_loader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module | None,
    node_type_ids: np.ndarray,
    device: str,
    train_rng: np.random.RandomState,
    train_neg_ratio: int = 1,
    loss_type: str = "bce",
    neg_sampler: _PopularityNegativeSampler | None = None,
) -> tuple[float, dict]:
    model.train()
    model.set_neighbor_sampler(neighbor_sampler)

    streamer_pool, room_pool = _type_pools(node_type_ids)
    losses: list[float] = []
    batch_metrics: list[dict] = []

    tgn = _is_tgn_model(model)
    for batch_indices in tqdm(train_loader, ncols=120, desc="train"):
        idx = batch_indices.numpy()
        if tgn:
            idx = _sort_batch_indices_by_time(idx, train_data.node_interact_times)
        bu = train_data.user_node_ids[idx]
        bv = train_data.streamer_node_ids[idx]
        bw = train_data.item_node_ids[idx]
        bt = train_data.node_interact_times[idx]

        if neg_sampler is not None:
            bv_neg, bw_neg = neg_sampler.sample(bv, bw, train_rng, num_negatives=train_neg_ratio)
        else:
            bv_neg, bw_neg = _sample_negative_pairs(
                bv, bw, streamer_pool, room_pool, train_rng, num_negatives=train_neg_ratio
            )
        batch_edge_ids = train_data.edge_ids[idx]
        y_pos, y_neg = _forward_pos_neg_scores(
            model, bu, bv, bw, bt, bv_neg, bw_neg, batch_edge_ids, tgn=tgn
        )
        loss = _compute_tripartite_training_loss(
            y_pos, y_neg, loss_type=loss_type, loss_fn=loss_fn, device=device
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if hasattr(model, "detach_memory_after_batch"):
            model.detach_memory_after_batch()

        losses.append(loss.item())
        with torch.no_grad():
            predicts, labels = _training_metric_tensors(y_pos, y_neg, device)
            batch_metrics.append(get_link_prediction_metrics(predicts=predicts, labels=labels))

    train_metrics_mean = {k: float(np.mean([m[k] for m in batch_metrics])) for k in batch_metrics[0]}
    return float(np.mean(losses)), train_metrics_mean


@torch.no_grad()
def evaluate_valid_loss(
    model: nn.Module,
    neighbor_sampler,
    val_data: TripartiteData,
    val_loader,
    loss_fn: nn.Module,
    node_type_ids: np.ndarray,
    device: str,
    val_rng: np.random.RandomState,
    neg_sampler: _PopularityNegativeSampler | None = None,
) -> float:
    """
    Lightweight validation: one negative (v', w') per val triplet, mean BCE (no optimizer step).
    Same forward protocol as train_one_epoch.

    If ``neg_sampler`` is given (popularity, train-split), the single negative is drawn from it so
    that checkpoint selection aligns with the eval popularity component at the same cost; otherwise
    the negative is uniform over the streamer/room pools (legacy).
    """
    model.eval()
    model.set_neighbor_sampler(neighbor_sampler)

    streamer_pool, room_pool = _type_pools(node_type_ids)
    losses: list[float] = []
    tgn = _is_tgn_model(model)

    for batch_indices in tqdm(val_loader, ncols=120, desc="val loss"):
        idx = batch_indices.numpy()
        if tgn:
            idx = _sort_batch_indices_by_time(idx, val_data.node_interact_times)
        bu = val_data.user_node_ids[idx]
        bv = val_data.streamer_node_ids[idx]
        bw = val_data.item_node_ids[idx]
        bt = val_data.node_interact_times[idx]

        if neg_sampler is not None:
            bv_neg, bw_neg = neg_sampler.sample(bv, bw, val_rng, num_negatives=1)
        else:
            bv_neg, bw_neg = _sample_negative_pairs(bv, bw, streamer_pool, room_pool, val_rng)

        batch_edge_ids = val_data.edge_ids[idx]
        if tgn:
            y_neg, _, _, _ = model(
                bu, bv_neg, bw_neg, bt, edge_ids=None, edges_are_positive=False
            )
            y_pos, _, _, _ = model(
                bu, bv, bw, bt, edge_ids=batch_edge_ids, edges_are_positive=True
            )
        else:
            y_pos, _, _, _ = model(
                bu, bv, bw, bt, edge_ids=batch_edge_ids, edges_are_positive=True
            )
            y_neg, _, _, _ = model(
                bu, bv_neg, bw_neg, bt, edge_ids=None, edges_are_positive=False
            )

        y_pos = y_pos.view(-1)
        y_neg = y_neg.view(-1)
        predicts = torch.cat([y_pos, y_neg], dim=0)
        labels = torch.cat(
            [
                torch.ones_like(y_pos, device=device, dtype=torch.float32),
                torch.zeros_like(y_neg, device=device, dtype=torch.float32),
            ],
            dim=0,
        )
        loss = loss_fn(predicts.float(), labels)
        losses.append(float(loss.item()))

    return float(np.mean(losses))


def main():
    warnings.filterwarnings("ignore")
    args = get_tripartite_train_args()

    (
        node_raw_features,
        edge_raw_features,
        node_type_ids,
        full_data,
        train_data,
        val_data,
        test_data,
        new_node_val_data,
        new_node_test_data,
        temporal_full_data,
        temporal_train_data,
    ) = get_tripartite_link_prediction_data(
        dataset_name=args.dataset_name,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        data_dir=args.data_dir,
    )

    num_nodes = len(node_type_ids)
    if temporal_full_data is not None and temporal_train_data is not None:
        train_neighbor_sampler = get_neighbor_sampler(
            data=temporal_train_data,
            sample_neighbor_strategy=args.sample_neighbor_strategy,
            time_scaling_factor=args.time_scaling_factor,
            seed=0,
            num_nodes=num_nodes,
        )
        full_neighbor_sampler = get_neighbor_sampler(
            data=temporal_full_data,
            sample_neighbor_strategy=args.sample_neighbor_strategy,
            time_scaling_factor=args.time_scaling_factor,
            seed=1,
            num_nodes=num_nodes,
        )
    else:
        train_neighbor_sampler = get_tripartite_neighbor_sampler(
            data=train_data,
            sample_neighbor_strategy=args.sample_neighbor_strategy,
            time_scaling_factor=args.time_scaling_factor,
            seed=0,
            num_nodes=num_nodes,
        )
        full_neighbor_sampler = get_tripartite_neighbor_sampler(
            data=full_data,
            sample_neighbor_strategy=args.sample_neighbor_strategy,
            time_scaling_factor=args.time_scaling_factor,
            seed=1,
            num_nodes=num_nodes,
        )

    # TGN memory must see events in chronological order (DyGLib train_link_prediction uses shuffle=False).
    train_loader = get_idx_data_loader(
        indices_list=list(range(len(train_data.user_node_ids))),
        batch_size=args.batch_size,
        shuffle=args.model_name != "TGN",
    )
    val_loader = get_idx_data_loader(
        indices_list=list(range(len(val_data.user_node_ids))),
        batch_size=args.batch_size,
        shuffle=False,
    )
    test_loader = get_idx_data_loader(
        indices_list=list(range(len(test_data.user_node_ids))),
        batch_size=args.batch_size,
        shuffle=False,
    )

    # --- build (or load) the fixed, model-independent test candidate set ONCE ---
    # It depends only on (data, config, seed), never on any model, so every run / model / ablation
    # that points at the same file ranks against byte-identical negatives.
    test_candidates = None
    test_candidates_path = None
    if args.use_fixed_eval_candidates:
        if args.eval_candidates_path is not None:
            test_candidates_path = args.eval_candidates_path
        else:
            test_candidates_path = (
                f"./eval_candidates/{args.dataset_name}_test_joint_vw_seed{args.neg_sampling_seed}.npz"
            )
        cand_cfg = EvalCandidateConfig(
            num_negatives=args.num_ranking_negatives,
            ratio_popularity=args.neg_ratio_popularity,
            ratio_hard=args.neg_ratio_hard,
            ratio_uniform=args.neg_ratio_uniform,
            popularity_window=args.neg_popularity_window,
            candidate_universe=args.neg_candidate_universe,
            setting="joint_vw",
            seed=args.neg_sampling_seed,
        )
        if os.path.exists(test_candidates_path):
            test_candidates = load_eval_candidates(test_candidates_path)
            logging.info("Loaded fixed eval candidates from %s", test_candidates_path)
        else:
            test_candidates = build_eval_candidates(
                test_data, full_data, node_type_ids, cand_cfg, train_data=train_data
            )
            save_eval_candidates(test_candidates_path, test_candidates)
            logging.info("Built and saved fixed eval candidates to %s", test_candidates_path)
        logging.info(
            "Fixed eval candidates sha256=%s | %s",
            _file_sha256(test_candidates_path),
            test_candidates.summary(),
        )
        if args.enable_user_item_setting:
            ui_cfg = EvalCandidateConfig(
                num_negatives=args.num_ranking_negatives, candidate_universe=args.neg_candidate_universe,
                setting="item_only", seed=args.neg_sampling_seed,
            )
            ui_path = f"./eval_candidates/{args.dataset_name}_test_item_only_seed{args.neg_sampling_seed}.npz"
            if not os.path.exists(ui_path):
                save_eval_candidates(
                    ui_path,
                    build_eval_candidates(test_data, full_data, node_type_ids, ui_cfg, train_data=train_data),
                )
            logging.info("Built user-item (item_only) candidates at %s (setting is opt-in).", ui_path)
    else:
        logging.warning(
            "Legacy eval protocol: sampling uniform test negatives on the fly (eval_seed+1=%s). "
            "Use --use_fixed_eval_candidates for the fair/harder shared protocol.",
            args.eval_seed + 1,
        )

    # Ranking-based validation monitoring. Built once (model-independent) when either ranking-based
    # early stopping is selected, or --log_val_ranking asks to observe the ranking trend per epoch.
    # The per-epoch ranking pass uses a fixed subsample loader (val_ranking_loader) to stay cheap.
    val_candidates = None
    val_ranking_loader = val_loader
    val_ranking_n = len(val_data.user_node_ids)
    if args.early_stop_metric != "val_loss" or args.log_val_ranking:
        val_cfg = EvalCandidateConfig(
            num_negatives=args.num_ranking_negatives,
            ratio_popularity=args.neg_ratio_popularity,
            ratio_hard=args.neg_ratio_hard,
            ratio_uniform=args.neg_ratio_uniform,
            popularity_window=args.neg_popularity_window,
            candidate_universe=args.neg_candidate_universe,
            setting="joint_vw",
            seed=args.neg_sampling_seed,
        )
        val_candidates = build_eval_candidates(
            val_data, full_data, node_type_ids, val_cfg, train_data=train_data
        )
        n_val = len(val_data.user_node_ids)
        if 0 < args.val_ranking_subsample < n_val:
            sub_rng = np.random.RandomState(args.eval_seed)
            sub_idx = np.sort(sub_rng.choice(n_val, size=args.val_ranking_subsample, replace=False))
            val_ranking_loader = get_idx_data_loader(
                indices_list=sub_idx.tolist(), batch_size=args.batch_size, shuffle=False
            )
            val_ranking_n = int(args.val_ranking_subsample)
        if args.log_val_ranking:
            logging.info(
                "Observation: logging val ranking (ndcg@5/@10, recall@10) on %d val queries per "
                "epoch; early stopping stays on '%s' (NOT changed).",
                val_ranking_n,
                args.early_stop_metric,
            )
        if args.early_stop_metric != "val_loss":
            logging.warning(
                "early_stop_metric=%s selects checkpoints by VALIDATION RANKING, not BCE. This "
                "changes model selection -- re-train all models before reporting.",
                args.early_stop_metric,
            )

    # Training negative sampler: built once from the TRAIN split only (no val/test leakage).
    train_neg_sampler = None
    if args.train_neg_sampling == "popularity":
        train_neg_sampler = _PopularityNegativeSampler(train_data, power=args.train_neg_power)
        logging.info(
            "Training negatives: popularity (power=%s) over %d observed train combos.",
            args.train_neg_power,
            train_neg_sampler.num_combos,
        )

    # Validation (early-stopping) negative sampler: built once from the TRAIN split only (no
    # val/test leakage). Aligns the 1-neg validation BCE -- and therefore checkpoint selection --
    # with the eval popularity component, at the same cost as the uniform 1-neg validation.
    val_neg_sampler = None
    if args.val_neg_sampling == "popularity":
        if train_neg_sampler is not None and args.train_neg_power == 1.0:
            val_neg_sampler = train_neg_sampler  # identical config; stateless aside from the rng
        else:
            val_neg_sampler = _PopularityNegativeSampler(train_data, power=1.0)
        logging.info(
            "Validation negatives: popularity (train-split, power=1.0) over %d observed train combos.",
            val_neg_sampler.num_combos,
        )
    else:
        logging.info("Validation negatives: uniform (legacy 1-neg BCE).")

    best_val_loss_all_runs: list[float] = []
    test_metric_all_runs: list[dict] = []

    for run in range(args.num_runs):
        set_random_seed(seed=run)
        args.seed = run
        # === 動態命名修復：根據消融參數自動調整儲存檔名 ===
        suffix = ""
        if args.model_name == "HTTransformer":
            if not args.use_bias_gate:
                suffix += "_no_bias_gate"
            if not args.use_type_init:
                suffix += "_no_type_init"
            if not args.use_hetero_coocc:
                suffix += "_no_hetero_coocc"
            if args.fusion_mode != "concat":
                suffix += f"_{args.fusion_mode}"
            if args.loss_type != "bce":
                suffix += f"_{args.loss_type}"
            if args.train_neg_ratio != 1:
                suffix += f"_neg{args.train_neg_ratio}"

        if args.train_neg_sampling != "uniform":
            suffix += "_popneg"

        args.save_model_name = f"{args.model_name}{suffix}_seed{args.seed}"
        # =================================================

        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        os.makedirs(
            f"./logs/{args.model_name}/{args.dataset_name}/{args.save_model_name}/",
            exist_ok=True,
        )
        fh = logging.FileHandler(
            f"./logs/{args.model_name}/{args.dataset_name}/{args.save_model_name}/{str(time.time())}.log"
        )
        fh.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setLevel(logging.WARNING)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        logger.addHandler(fh)
        logger.addHandler(ch)

        run_start_time = time.time()
        logger.info("********** Run %s starts. **********", run + 1)
        logger.info("configuration: %s", args)

        if args.model_name == "HAN":
            from models.HAN import HAN

            num_nodes = int(node_raw_features.shape[0])
            model = HAN(
                train_data=train_data,
                node_raw_features=node_raw_features,
                num_nodes=num_nodes,
                device=args.device,
                hidden_dim=args.channel_embedding_dim,
                n_heads=args.han_n_heads,
                n_gat_layers=args.han_gat_layers,
                dropout=args.dropout,
                meta_path_nhood=args.han_meta_path_nhood,
            )
        elif args.model_name == "LightGCN":
            from models.LightGCN import LightGCN

            num_nodes = int(node_raw_features.shape[0])
            model = LightGCN(
                train_data=train_data,
                num_nodes=num_nodes,
                device=args.device,
                embedding_dim=args.channel_embedding_dim,
                n_layers=args.lightgcn_n_layers,
                dropout=args.dropout,
            )
        elif args.model_name == "HyperHawkes":
            from models.HyperHawkes import HyperHawkes

            model = HyperHawkes(
                train_data=train_data,
                node_type_ids=node_type_ids,
                device=args.device,
                hidden_size=args.channel_embedding_dim,
                max_seq_length=args.max_input_sequence_length,
                num_heads=args.num_heads,
                n_levels=2,
                hgnn_layers=args.num_layers,
                dropout=args.dropout,
                sub_time_delta=args.hyperhawkes_sub_time_delta,
                day_factor=args.hyperhawkes_day_factor,
                min_support=args.hyperhawkes_min_support,
            )
        elif args.model_name == "HTTransformer":
            model = HTTransformer(
                node_raw_features=node_raw_features,
                edge_raw_features=edge_raw_features,
                node_type_ids=node_type_ids,
                neighbor_sampler=train_neighbor_sampler,
                time_feat_dim=args.time_feat_dim,
                hidden_dim=args.channel_embedding_dim,
                cooccurrence_dim=args.cooccurrence_dim,
                d_out=None,
                patch_size=args.patch_size,
                num_layers=args.num_layers,
                num_heads=args.num_heads,
                dropout=args.dropout,
                max_input_sequence_length=args.max_input_sequence_length,
                device=args.device,
                use_bias_gate=args.use_bias_gate,
                use_type_init=args.use_type_init,
                use_hetero_coocc=args.use_hetero_coocc,
                fusion_mode=args.fusion_mode,
            )
        elif args.model_name in TRIPARTITE_BASELINE_MODELS:
            model = BaselineTripartiteWrapper(
                model_name=args.model_name,
                node_raw_features=node_raw_features,
                edge_raw_features=edge_raw_features,
                neighbor_sampler=train_neighbor_sampler,
                device=args.device,
                time_feat_dim=args.time_feat_dim,
                num_layers=args.num_layers,
                num_heads=args.num_heads,
                dropout=args.dropout,
                num_neighbors=args.num_neighbors,
                patch_size=args.patch_size,
                max_input_sequence_length=args.max_input_sequence_length,
                channel_embedding_dim=args.channel_embedding_dim,
                time_gap=args.time_gap,
                walk_length=args.walk_length,
                position_feat_dim=args.position_feat_dim,
                num_walk_heads=args.num_walk_heads,
                num_depths=args.num_depths,
            )
        else:
            raise ValueError(f"Unknown model_name: {args.model_name}")
        logger.info(
            "model #parameters (trainable): %s",
            get_parameter_sizes(model) * 4,
        )

        optimizer = create_optimizer(
            model=model,
            optimizer_name=args.optimizer,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        model = convert_to_gpu(model, device=args.device)
        train_loss_fn = nn.BCELoss() if args.loss_type == "bce" else None
        val_loss_fn = nn.BCELoss()

        if args.model_name != "HTTransformer" and args.fusion_mode != "concat":
            logger.warning(
                "fusion_mode=%s applies only to HTTransformer; ignored for %s.",
                args.fusion_mode,
                args.model_name,
            )

        save_model_folder = f"./saved_models/{args.model_name}/{args.dataset_name}/{args.save_model_name}/"
        shutil.rmtree(save_model_folder, ignore_errors=True)
        os.makedirs(save_model_folder, exist_ok=True)

        early_stopping = EarlyStopping(
            patience=args.patience,
            save_model_folder=save_model_folder,
            save_model_name=args.save_model_name,
            logger=logger,
            model_name=args.model_name,
        )

        train_rng = np.random.RandomState(seed=run + 12345)
        val_loss_rng = np.random.RandomState(seed=args.eval_seed)
        test_eval_rng = np.random.RandomState(seed=args.eval_seed + 1)

        for epoch in range(args.num_epochs):
            if args.model_name == "HyperHawkes" and hasattr(model, "e_step"):
                model.e_step()
            if args.model_name == "TGN" and hasattr(model, "init_memory_for_epoch"):
                model.init_memory_for_epoch()

            train_loss, train_metrics = train_one_epoch(
                model=model,
                neighbor_sampler=train_neighbor_sampler,
                train_data=train_data,
                train_loader=train_loader,
                optimizer=optimizer,
                loss_fn=train_loss_fn,
                node_type_ids=node_type_ids,
                device=args.device,
                train_rng=train_rng,
                train_neg_ratio=args.train_neg_ratio,
                loss_type=args.loss_type,
                neg_sampler=train_neg_sampler,
            )

            train_mem_backup = None
            if args.model_name == "TGN":
                train_mem_backup = model.backup_memory_bank()

            val_loss = evaluate_valid_loss(
                model=model,
                neighbor_sampler=train_neighbor_sampler,
                val_data=val_data,
                val_loader=val_loader,
                loss_fn=val_loss_fn,
                node_type_ids=node_type_ids,
                device=args.device,
                val_rng=val_loss_rng,
                neg_sampler=val_neg_sampler,
            )

            # ranking-based validation monitoring: log the trend (observation) and/or feed early stop
            val_monitor_value = None
            if val_candidates is not None:
                _vl, val_rank_agg = evaluate_tripartite_ranking(
                    model=model,
                    neighbor_sampler=train_neighbor_sampler,
                    data=val_data,
                    idx_data_loader=val_ranking_loader,
                    node_type_ids=node_type_ids,
                    device=args.device,
                    num_negatives=args.num_ranking_negatives,
                    eval_rng=val_loss_rng,
                    eval_candidates=val_candidates,
                )
                if args.log_val_ranking:
                    logger.info(
                        "Epoch %s | val ranking (n=%s) ndcg@5 %.4f | ndcg@10 %.4f | recall@10 %.4f",
                        epoch + 1,
                        val_ranking_n,
                        val_rank_agg.get("ndcg@5", float("nan")),
                        val_rank_agg.get("ndcg@10", float("nan")),
                        val_rank_agg.get("recall@10", float("nan")),
                    )
                if args.early_stop_metric != "val_loss":
                    val_monitor_value = val_rank_agg.get(args.early_stop_metric.replace("val_", ""))

            if args.model_name == "TGN" and train_mem_backup is not None:
                model.reload_memory_bank(train_mem_backup)

            logger.info(
                "Epoch %s | train loss %.4f | val loss (BCE, 1 neg) %.4f",
                epoch + 1,
                train_loss,
                val_loss,
            )
            for mk, mv in train_metrics.items():
                logger.info("train %s %.4f", mk, mv)

            if args.early_stop_metric == "val_loss":
                stop = early_stopping.step([("val_loss", val_loss, False)], model)
            else:
                logger.info("Epoch %s | %s %.4f", epoch + 1, args.early_stop_metric, val_monitor_value)
                stop = early_stopping.step([(args.early_stop_metric, val_monitor_value, True)], model)
            if stop:
                logger.info("Early stopping at epoch %s (best %s).", epoch + 1, args.early_stop_metric)
                break

        early_stopping.load_checkpoint(model)
        best_val_loss = early_stopping.best_metrics.get("val_loss", float("nan"))
        best_val_loss_all_runs.append(float(best_val_loss))
        logger.info("Loaded best checkpoint (lowest val_loss = %.4f). Final test ranking (1+%s negatives) ...", best_val_loss, args.num_ranking_negatives)

        test_loss, test_rank_metrics, test_per_query_metrics = evaluate_tripartite_ranking(
            model=model,
            neighbor_sampler=full_neighbor_sampler,
            data=test_data,
            idx_data_loader=test_loader,
            node_type_ids=node_type_ids,
            device=args.device,
            num_negatives=args.num_ranking_negatives,
            eval_rng=test_eval_rng,
            return_per_query=True,
            eval_candidates=test_candidates,
        )
        test_cold_start_metrics = evaluate_tripartite_cold_start_by_split(
            train_data, test_data, test_per_query_metrics
        )

        logger.info("test ranking loss (diagnostic BCE on 1+99) %.4f", test_loss)
        for mk, mv in test_rank_metrics.items():
            logger.info("test %s %.4f", mk, mv)
        logger.info("test cold-start splits (AUC, P@10, R@10, N@10, subset means):")
        for split_name, split_m in test_cold_start_metrics.items():
            logger.info("  %s: %s", split_name, split_m)

        test_metric_all_runs.append(test_rank_metrics)

        logger.info("Run %s cost %.2f s", run + 1, time.time() - run_start_time)

        def _json_float(x: float) -> str:
            return "nan" if isinstance(x, float) and np.isnan(x) else f"{x:.4f}"

        result_json = {
            "best_val_loss": _json_float(best_val_loss),
            "test metrics": {k: _json_float(v) for k, v in test_rank_metrics.items()},
            "test cold-start splits": {
                split: {m: _json_float(val) for m, val in metrics.items()}
                for split, metrics in test_cold_start_metrics.items()
            },
        }
        save_result_folder = f"./saved_results/{args.model_name}/{args.dataset_name}"
        os.makedirs(save_result_folder, exist_ok=True)
        with open(
            os.path.join(save_result_folder, f"{args.save_model_name}.json"),
            "w",
            encoding="utf-8",
        ) as f:
            f.write(json.dumps(result_json, indent=4))

        if run < args.num_runs - 1:
            logger.removeHandler(fh)
            logger.removeHandler(ch)

    logging.info("Metrics over %s runs:", args.num_runs)
    if best_val_loss_all_runs:
        xs = best_val_loss_all_runs
        std = np.std(xs, ddof=1) if len(xs) > 1 else 0.0
        logging.info("best_val_loss %s | mean %.4f ± %.4f", xs, np.mean(xs), std)
    if test_metric_all_runs:
        for name in test_metric_all_runs[0].keys():
            xs = [r[name] for r in test_metric_all_runs]
            std = np.std(xs, ddof=1) if len(xs) > 1 else 0.0
            logging.info("test %s %s | mean %.4f ± %.4f", name, xs, np.mean(xs), std)

    if args.metrics_out:
        test_mean = mean_metric_dicts(test_metric_all_runs)

        def _json_scalar(x: float) -> float | None:
            if isinstance(x, float) and (np.isnan(x) or np.isinf(x)):
                return None
            return float(x)

        summary: dict = {
            "model_name": args.model_name,
            "dataset_name": args.dataset_name,
            "num_runs": args.num_runs,
            "patch_size": args.patch_size,
            "channel_embedding_dim": args.channel_embedding_dim,
            "cooccurrence_dim": args.cooccurrence_dim,
            "best_val_loss": _json_scalar(float(np.mean(best_val_loss_all_runs))) if best_val_loss_all_runs else None,
        }
        if args.model_name == "HTTransformer":
            summary["use_bias_gate"] = args.use_bias_gate
            summary["use_type_init"] = args.use_type_init
            summary["use_hetero_coocc"] = args.use_hetero_coocc
            summary["fusion_mode"] = args.fusion_mode
        else:
            summary["use_bias_gate"] = None
            summary["use_type_init"] = None
            summary["use_hetero_coocc"] = None
            summary["fusion_mode"] = None
        summary["train_neg_ratio"] = args.train_neg_ratio
        summary["train_neg_sampling"] = args.train_neg_sampling
        summary["loss_type"] = args.loss_type
        for k, v in test_mean.items():
            summary[f"test_{k}"] = _json_scalar(v)
        os.makedirs(os.path.dirname(os.path.abspath(args.metrics_out)) or ".", exist_ok=True)
        with open(args.metrics_out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
