"""
Tripartite (User, Streamer, Room) link prediction: HTTransformer or DyGLib baselines via BaselineTripartiteWrapper.

- Training: BCE loss with one random negative (streamer, room) pair per positive triplet.
- Validation / Test: ranking with 1 ground-truth + 99 random negatives (100 candidates),
  metrics: ROC-AUC, AP, Precision@K, Recall@K, NDCG@K for K in {5, 10, 20, 50, 100} (see utils.metrics).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from models.BaselineTripartiteWrapper import BaselineTripartiteWrapper, TRIPARTITE_BASELINE_MODELS
from models.HTTransformer import HTTransformer
from utils.DataLoader import TripartiteData, get_idx_data_loader, get_tripartite_link_prediction_data
from utils.EarlyStopping import EarlyStopping
from utils.metrics import get_link_prediction_metrics, mean_metric_dicts, tripartite_ranking_metrics_per_query
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
    parser.add_argument("--test_interval_epochs", type=int, default=10)
    parser.add_argument("--num_ranking_negatives", type=int, default=99, help="99 negatives + 1 positive = 100 candidates")
    parser.add_argument("--eval_seed", type=int, default=0, help="seed for ranking negative sampling in val/test")
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
    return args


def _type_pools(node_type_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Global ids for streamers (type 2) and rooms (type 3)."""
    streamer_pool = np.where(node_type_ids == 2)[0].astype(np.int64)
    room_pool = np.where(node_type_ids == 3)[0].astype(np.int64)
    if len(streamer_pool) == 0 or len(room_pool) == 0:
        raise ValueError("node_type_ids must contain at least one streamer (2) and one room (3).")
    return streamer_pool, room_pool


def _sample_negative_pairs(
    batch_v_pos: np.ndarray,
    batch_w_pos: np.ndarray,
    streamer_pool: np.ndarray,
    room_pool: np.ndarray,
    rng: np.random.RandomState,
) -> tuple[np.ndarray, np.ndarray]:
    """One negative (v', w') per positive; resample until pair differs from (v_pos, w_pos)."""
    b = len(batch_v_pos)
    v_neg = np.empty(b, dtype=np.int64)
    w_neg = np.empty(b, dtype=np.int64)
    for i in range(b):
        vp, wp = int(batch_v_pos[i]), int(batch_w_pos[i])
        for _ in range(50):
            vn = int(rng.choice(streamer_pool))
            wn = int(rng.choice(room_pool))
            if vn != vp or wn != wp:
                v_neg[i] = vn
                w_neg[i] = wn
                break
        else:
            v_neg[i] = vp
            w_neg[i] = wn if wp != wn else int(rng.choice(room_pool))
    return v_neg, w_neg


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
) -> tuple[float, dict] | tuple[float, dict, list[dict]]:
    """
    For each query triplet, build 1 positive + num_negatives negatives (random v,w), score all, rank.
    Returns mean BCE on pos/neg pairs (diagnostic) and aggregated ranking metrics.
    If return_per_query, also returns metrics per test row (loader order must match sequential indices).
    """
    model.eval()
    model.set_neighbor_sampler(neighbor_sampler)

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
    loss_fn: nn.Module,
    node_type_ids: np.ndarray,
    device: str,
    train_rng: np.random.RandomState,
) -> tuple[float, dict]:
    model.train()
    model.set_neighbor_sampler(neighbor_sampler)

    streamer_pool, room_pool = _type_pools(node_type_ids)
    losses: list[float] = []
    batch_metrics: list[dict] = []

    for batch_indices in tqdm(train_loader, ncols=120, desc="train"):
        idx = batch_indices.numpy()
        bu = train_data.user_node_ids[idx]
        bv = train_data.streamer_node_ids[idx]
        bw = train_data.item_node_ids[idx]
        bt = train_data.node_interact_times[idx]

        bv_neg, bw_neg = _sample_negative_pairs(bv, bw, streamer_pool, room_pool, train_rng)

        batch_edge_ids = train_data.edge_ids[idx]
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

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if hasattr(model, "detach_memory_after_batch"):
            model.detach_memory_after_batch()

        losses.append(loss.item())
        with torch.no_grad():
            batch_metrics.append(get_link_prediction_metrics(predicts=predicts, labels=labels))

    train_metrics_mean = {k: float(np.mean([m[k] for m in batch_metrics])) for k in batch_metrics[0]}
    return float(np.mean(losses)), train_metrics_mean


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

    train_loader = get_idx_data_loader(
        indices_list=list(range(len(train_data.user_node_ids))),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = get_idx_data_loader(
        indices_list=list(range(len(val_data.user_node_ids))),
        batch_size=args.batch_size,
        shuffle=False,
    )
    new_node_val_loader = get_idx_data_loader(
        indices_list=list(range(len(new_node_val_data.user_node_ids))),
        batch_size=args.batch_size,
        shuffle=False,
    )
    test_loader = get_idx_data_loader(
        indices_list=list(range(len(test_data.user_node_ids))),
        batch_size=args.batch_size,
        shuffle=False,
    )
    new_node_test_loader = get_idx_data_loader(
        indices_list=list(range(len(new_node_test_data.user_node_ids))),
        batch_size=args.batch_size,
        shuffle=False,
    )

    val_metric_all_runs: list[dict] = []
    new_node_val_metric_all_runs: list[dict] = []
    test_metric_all_runs: list[dict] = []
    new_node_test_metric_all_runs: list[dict] = []

    for run in range(args.num_runs):
        set_random_seed(seed=run)
        args.seed = run
        args.save_model_name = f"{args.model_name}_seed{args.seed}"

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
        loss_fn = nn.BCELoss()

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
        val_eval_rng = np.random.RandomState(seed=args.eval_seed)
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
                loss_fn=loss_fn,
                node_type_ids=node_type_ids,
                device=args.device,
                train_rng=train_rng,
            )

            train_backup_memory_bank = None
            if args.model_name == "TGN" and hasattr(model, "memory_bank"):
                train_backup_memory_bank = model.memory_bank.backup_memory_bank()

            val_loss, val_rank_metrics = evaluate_tripartite_ranking(
                model=model,
                neighbor_sampler=full_neighbor_sampler,
                data=val_data,
                idx_data_loader=val_loader,
                node_type_ids=node_type_ids,
                device=args.device,
                num_negatives=args.num_ranking_negatives,
                eval_rng=val_eval_rng,
            )

            new_node_val_loss, new_node_val_rank_metrics = evaluate_tripartite_ranking(
                model=model,
                neighbor_sampler=full_neighbor_sampler,
                data=new_node_val_data,
                idx_data_loader=new_node_val_loader,
                node_type_ids=node_type_ids,
                device=args.device,
                num_negatives=args.num_ranking_negatives,
                eval_rng=val_eval_rng,
            )

            val_backup_memory_bank = None
            if args.model_name == "TGN" and hasattr(model, "memory_bank"):
                val_backup_memory_bank = model.memory_bank.backup_memory_bank()
                if train_backup_memory_bank is not None:
                    model.memory_bank.reload_memory_bank(train_backup_memory_bank)

            logger.info(
                "Epoch %s | train loss %.4f | val rank loss %.4f | val AP %.4f | val NDCG@10 %.4f",
                epoch + 1,
                train_loss,
                val_loss,
                val_rank_metrics.get("average_precision", float("nan")),
                val_rank_metrics.get("ndcg@10", float("nan")),
            )
            for mk, mv in train_metrics.items():
                logger.info("train %s %.4f", mk, mv)
            for mk, mv in val_rank_metrics.items():
                logger.info("val %s %.4f", mk, mv)

            if (epoch + 1) % args.test_interval_epochs == 0:
                test_loss_ep, test_rank_metrics = evaluate_tripartite_ranking(
                    model=model,
                    neighbor_sampler=full_neighbor_sampler,
                    data=test_data,
                    idx_data_loader=test_loader,
                    node_type_ids=node_type_ids,
                    device=args.device,
                    num_negatives=args.num_ranking_negatives,
                    eval_rng=test_eval_rng,
                )
                nn_test_loss_ep, nn_test_rank_metrics = evaluate_tripartite_ranking(
                    model=model,
                    neighbor_sampler=full_neighbor_sampler,
                    data=new_node_test_data,
                    idx_data_loader=new_node_test_loader,
                    node_type_ids=node_type_ids,
                    device=args.device,
                    num_negatives=args.num_ranking_negatives,
                    eval_rng=test_eval_rng,
                )
                logger.info("epoch %s test rank loss %.4f", epoch + 1, test_loss_ep)
                for mk, mv in test_rank_metrics.items():
                    logger.info("epoch %s test %s %.4f", epoch + 1, mk, mv)
                logger.info("epoch %s new node test rank loss %.4f", epoch + 1, nn_test_loss_ep)
                for mk, mv in nn_test_rank_metrics.items():
                    logger.info("epoch %s new node test %s %.4f", epoch + 1, mk, mv)

            if args.model_name == "TGN" and val_backup_memory_bank is not None:
                model.memory_bank.reload_memory_bank(val_backup_memory_bank)

            val_indicator = [(name, val, True) for name, val in val_rank_metrics.items()]
            if early_stopping.step(val_indicator, model):
                break

        early_stopping.load_checkpoint(model)

        val_eval_rng = np.random.RandomState(seed=args.eval_seed)
        test_eval_rng = np.random.RandomState(seed=args.eval_seed + 1)

        logger.info("Final evaluation on %s ...", args.dataset_name)
        val_loss, val_rank_metrics = evaluate_tripartite_ranking(
            model=model,
            neighbor_sampler=full_neighbor_sampler,
            data=val_data,
            idx_data_loader=val_loader,
            node_type_ids=node_type_ids,
            device=args.device,
            num_negatives=args.num_ranking_negatives,
            eval_rng=val_eval_rng,
        )
        new_node_val_loss, new_node_val_rank_metrics = evaluate_tripartite_ranking(
            model=model,
            neighbor_sampler=full_neighbor_sampler,
            data=new_node_val_data,
            idx_data_loader=new_node_val_loader,
            node_type_ids=node_type_ids,
            device=args.device,
            num_negatives=args.num_ranking_negatives,
            eval_rng=val_eval_rng,
        )
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
        )
        test_cold_start_metrics = evaluate_tripartite_cold_start_by_split(
            train_data, test_data, test_per_query_metrics
        )
        new_node_test_loss, new_node_test_rank_metrics = evaluate_tripartite_ranking(
            model=model,
            neighbor_sampler=full_neighbor_sampler,
            data=new_node_test_data,
            idx_data_loader=new_node_test_loader,
            node_type_ids=node_type_ids,
            device=args.device,
            num_negatives=args.num_ranking_negatives,
            eval_rng=test_eval_rng,
        )

        logger.info("validate ranking loss %.4f", val_loss)
        for mk, mv in val_rank_metrics.items():
            logger.info("validate %s %.4f", mk, mv)
        logger.info("new node validate ranking loss %.4f", new_node_val_loss)
        for mk, mv in new_node_val_rank_metrics.items():
            logger.info("new node validate %s %.4f", mk, mv)
        logger.info("test ranking loss %.4f", test_loss)
        for mk, mv in test_rank_metrics.items():
            logger.info("test %s %.4f", mk, mv)
        logger.info("test cold-start splits (AUC, P@10, R@10, N@10, subset means):")
        for split_name, split_m in test_cold_start_metrics.items():
            logger.info("  %s: %s", split_name, split_m)
        logger.info("new node test ranking loss %.4f", new_node_test_loss)
        for mk, mv in new_node_test_rank_metrics.items():
            logger.info("new node test %s %.4f", mk, mv)

        val_metric_all_runs.append(val_rank_metrics)
        new_node_val_metric_all_runs.append(new_node_val_rank_metrics)
        test_metric_all_runs.append(test_rank_metrics)
        new_node_test_metric_all_runs.append(new_node_test_rank_metrics)

        logger.info("Run %s cost %.2f s", run + 1, time.time() - run_start_time)

        def _json_float(x: float) -> str:
            return "nan" if isinstance(x, float) and np.isnan(x) else f"{x:.4f}"

        result_json = {
            "validate metrics": {k: _json_float(v) for k, v in val_rank_metrics.items()},
            "new node validate metrics": {k: _json_float(v) for k, v in new_node_val_rank_metrics.items()},
            "test metrics": {k: _json_float(v) for k, v in test_rank_metrics.items()},
            "test cold-start splits": {
                split: {m: _json_float(val) for m, val in metrics.items()}
                for split, metrics in test_cold_start_metrics.items()
            },
            "new node test metrics": {k: _json_float(v) for k, v in new_node_test_rank_metrics.items()},
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
    for name in val_metric_all_runs[0].keys():
        xs = [r[name] for r in val_metric_all_runs]
        std = np.std(xs, ddof=1) if len(xs) > 1 else 0.0
        logging.info("validate %s %s | mean %.4f ± %.4f", name, xs, np.mean(xs), std)
    for name in test_metric_all_runs[0].keys():
        xs = [r[name] for r in test_metric_all_runs]
        std = np.std(xs, ddof=1) if len(xs) > 1 else 0.0
        logging.info("test %s %s | mean %.4f ± %.4f", name, xs, np.mean(xs), std)

    if args.metrics_out:
        test_mean = mean_metric_dicts(test_metric_all_runs)
        val_mean = mean_metric_dicts(val_metric_all_runs)

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
        }
        if args.model_name == "HTTransformer":
            summary["use_bias_gate"] = args.use_bias_gate
            summary["use_type_init"] = args.use_type_init
            summary["use_hetero_coocc"] = args.use_hetero_coocc
        else:
            summary["use_bias_gate"] = None
            summary["use_type_init"] = None
            summary["use_hetero_coocc"] = None
        for k, v in test_mean.items():
            summary[f"test_{k}"] = _json_scalar(v)
        for k, v in val_mean.items():
            summary[f"val_{k}"] = _json_scalar(v)
        os.makedirs(os.path.dirname(os.path.abspath(args.metrics_out)) or ".", exist_ok=True)
        with open(args.metrics_out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
