"""
Evaluate HT-Transformer on Simulated Cold-Start Scenarios (Zero-History).
Forces the history of User, Streamer, or Item to be empty during inference.
"""

from __future__ import annotations

import argparse
import logging
import types
import warnings
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from eval_table10 import (
    _checkpoint_exists,
    _resolve_checkpoint_dir,
    build_ht_transformer,
    load_ht_checkpoint,
)
from models.HTTransformer import HTTransformer
from train_tripartite_link_prediction import evaluate_tripartite_ranking
from utils.DataLoader import get_idx_data_loader, get_tripartite_link_prediction_data
from utils.utils import convert_to_gpu, get_neighbor_sampler, get_tripartite_neighbor_sampler

ROOT = Path(__file__).resolve().parent
DEFAULT_FULL_DIR = ROOT / "saved_models" / "HTTransformer" / "kuailive_tripartite" / "HTTransformer_full_seed0"

EVAL_SETTINGS = (
    ("New u + Old s + Old i", True, False, False),
    ("Old u + New s + Old i", False, True, False),
    ("Old u + Old s + New i", False, False, True),
)

@contextmanager
def cold_start_simulation_context(model: HTTransformer, mask_u: bool, mask_s: bool, mask_i: bool):
    """
    Temporarily overrides compute_tripartite_temporal_embeddings to return empty
    history sequences for the masked node role, simulating a zero-history Cold Start.
    """
    _orig_compute = model.compute_tripartite_temporal_embeddings

    def simulated_compute(
        self,
        user_node_ids: np.ndarray,
        streamer_node_ids: np.ndarray,
        item_node_ids: np.ndarray,
        interact_times: np.ndarray,
    ):
        # 1. Get real neighbors
        u_nbr, u_edge, u_time = self.neighbor_sampler.get_all_first_hop_neighbors(user_node_ids, interact_times)
        v_nbr, v_edge, v_time = self.neighbor_sampler.get_all_first_hop_neighbors(streamer_node_ids, interact_times)
        w_nbr, w_edge, w_time = self.neighbor_sampler.get_all_first_hop_neighbors(item_node_ids, interact_times)

        # 2. Simulate Cold Start by clearing the respective history
        def empty(length): return [np.array([], dtype=np.int64) for _ in range(length)]
        def empty_time(length): return [np.array([], dtype=np.float32) for _ in range(length)]

        if mask_u:
            u_nbr, u_edge, u_time = empty(len(user_node_ids)), empty(len(user_node_ids)), empty_time(len(user_node_ids))
        if mask_s:
            v_nbr, v_edge, v_time = empty(len(streamer_node_ids)), empty(len(streamer_node_ids)), empty_time(len(streamer_node_ids))
        if mask_i:
            w_nbr, w_edge, w_time = empty(len(item_node_ids)), empty(len(item_node_ids)), empty_time(len(item_node_ids))

        # 3. Resume the original padding and encoding logic
        u_pad_ids, u_pad_e, u_pad_t = self.pad_sequences(user_node_ids, interact_times, u_nbr, u_edge, u_time, self.patch_size, self.max_input_sequence_length)
        v_pad_ids, v_pad_e, v_pad_t = self.pad_sequences(streamer_node_ids, interact_times, v_nbr, v_edge, v_time, self.patch_size, self.max_input_sequence_length)
        w_pad_ids, w_pad_e, w_pad_t = self.pad_sequences(item_node_ids, interact_times, w_nbr, w_edge, w_time, self.patch_size, self.max_input_sequence_length)

        u_co, v_co, w_co = self.cooccurrence_encoder(u_pad_ids, v_pad_ids, w_pad_ids)

        u_n, u_e, u_tf = self.get_features(interact_times, u_pad_ids, u_pad_e, u_pad_t)
        v_n, v_e, v_tf = self.get_features(interact_times, v_pad_ids, v_pad_e, v_pad_t)
        w_n, w_e, w_tf = self.get_features(interact_times, w_pad_ids, w_pad_e, w_pad_t)

        u_pn, u_pe, u_pt, u_pc = self.get_patches(u_n, u_e, u_tf, u_co, self.patch_size)
        v_pn, v_pe, v_pt, v_pc = self.get_patches(v_n, v_e, v_tf, v_co, self.patch_size)
        w_pn, w_pe, w_pt, w_pc = self.get_patches(w_n, w_e, w_tf, w_co, self.patch_size)

        zu = self._project_and_fuse_patches(u_pn, u_pe, u_pt, u_pc)
        zv = self._project_and_fuse_patches(v_pn, v_pe, v_pt, v_pc)
        zw = self._project_and_fuse_patches(w_pn, w_pe, w_pt, w_pc)

        z = torch.cat([zu, zv, zw], dim=1)
        for transformer in self.transformers:
            z = transformer(z)

        l_u, l_v, l_w = zu.shape[1], zv.shape[1], zw.shape[1]
        hu_seq = z[:, :l_u, :]
        hv_seq = z[:, l_u : l_u + l_v, :]
        hw_seq = z[:, l_u + l_v :, :]

        hu_pool = torch.mean(hu_seq, dim=1)
        hv_pool = torch.mean(hv_seq, dim=1)
        hw_pool = torch.mean(hw_seq, dim=1)

        h_u = self.output_proj_u(hu_pool)
        h_v = self.output_proj_v(hv_pool)
        h_w = self.output_proj_w(hw_pool)

        return h_u, h_v, h_w

    model.compute_tripartite_temporal_embeddings = types.MethodType(simulated_compute, model)
    try:
        yield
    finally:
        model.compute_tripartite_temporal_embeddings = _orig_compute

def get_eval_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate Cold-Start by zeroing history.")
    parser.add_argument("--dataset_name", type=str, default="kuailive_tripartite")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--sample_neighbor_strategy", type=str, default="recent")
    parser.add_argument("--time_scaling_factor", type=float, default=1e-6)
    parser.add_argument("--num_heads", type=int, default=2)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--time_feat_dim", type=int, default=100)
    parser.add_argument("--patch_size", type=int, default=1)
    parser.add_argument("--channel_embedding_dim", type=int, default=50)
    parser.add_argument("--cooccurrence_dim", type=int, default=50)
    parser.add_argument("--max_input_sequence_length", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num_ranking_negatives", type=int, default=99) 
    parser.add_argument("--eval_seed", type=int, default=0)
    parser.add_argument("--full_model_dir", type=str, default=str(DEFAULT_FULL_DIR))
    parser.add_argument("--run_seed", type=int, default=0)
    parser.add_argument("--full_save_model_name", type=str, default=None)
    return parser.parse_args()

def main() -> None:
    warnings.filterwarnings("ignore")
    args = get_eval_args()
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu"

    default_stem = f"HTTransformer_seed{args.run_seed}"
    full_stem = args.full_save_model_name or default_stem
    full_folder, full_stem = _resolve_checkpoint_dir(model_dir=args.full_model_dir, dataset_name=args.dataset_name, save_model_name=full_stem)
    
    (
        node_raw_features, edge_raw_features, node_type_ids, full_data, train_data,
        _val_data, test_data, _new_node_val_data, _new_node_test_data,
        temporal_full_data, temporal_train_data,
    ) = get_tripartite_link_prediction_data(dataset_name=args.dataset_name, val_ratio=args.val_ratio, test_ratio=args.test_ratio, data_dir=args.data_dir)

    num_nodes = len(node_type_ids)
    if temporal_full_data is not None and temporal_train_data is not None:
        train_neighbor_sampler = get_neighbor_sampler(data=temporal_train_data, sample_neighbor_strategy=args.sample_neighbor_strategy, time_scaling_factor=args.time_scaling_factor, seed=0, num_nodes=num_nodes)
        full_neighbor_sampler = get_neighbor_sampler(data=temporal_full_data, sample_neighbor_strategy=args.sample_neighbor_strategy, time_scaling_factor=args.time_scaling_factor, seed=1, num_nodes=num_nodes)
    else:
        train_neighbor_sampler = get_tripartite_neighbor_sampler(data=train_data, sample_neighbor_strategy=args.sample_neighbor_strategy, time_scaling_factor=args.time_scaling_factor, seed=0, num_nodes=num_nodes)
        full_neighbor_sampler = get_tripartite_neighbor_sampler(data=full_data, sample_neighbor_strategy=args.sample_neighbor_strategy, time_scaling_factor=args.time_scaling_factor, seed=1, num_nodes=num_nodes)

    test_loader = get_idx_data_loader(indices_list=list(range(len(test_data.user_node_ids))), batch_size=args.batch_size, shuffle=False)
    eval_rank_seed = args.eval_seed + 1

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("eval_coldstart")

    model = build_ht_transformer(
        node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, node_type_ids=node_type_ids,
        neighbor_sampler=train_neighbor_sampler, device=device, use_bias_gate=True, use_type_init=True, use_hetero_coocc=True,
        time_feat_dim=args.time_feat_dim, channel_embedding_dim=args.channel_embedding_dim, cooccurrence_dim=args.cooccurrence_dim,
        patch_size=args.patch_size, num_layers=args.num_layers, num_heads=args.num_heads, dropout=args.dropout, max_input_sequence_length=args.max_input_sequence_length,
    )
    load_ht_checkpoint(model, full_folder, full_stem, logger)
    model = convert_to_gpu(model, device=device)

    print("\n### Table 13 — Simulated Cold-Start Scenario Analysis ###")
    print("| Scenario | AUC | P@10 | R@10 | N@10 |")
    print("|----------|-----|------|------|------|")

    for setting_name, mu, ms, mi in EVAL_SETTINGS:
        test_eval_rng = np.random.RandomState(seed=eval_rank_seed)
        with cold_start_simulation_context(model, mask_u=mu, mask_s=ms, mask_i=mi):
            _loss, agg = evaluate_tripartite_ranking(
                model=model, neighbor_sampler=full_neighbor_sampler, data=test_data, idx_data_loader=test_loader,
                node_type_ids=node_type_ids, device=device, num_negatives=args.num_ranking_negatives, eval_rng=test_eval_rng, return_per_query=False,
            )
        print(f"| {setting_name} | {agg.get('roc_auc', float('nan')):.4f} | {agg.get('precision@10', float('nan')):.4f} | {agg.get('recall@10', float('nan')):.4f} | {agg.get('ndcg@10', float('nan')):.4f} |")

if __name__ == "__main__":
    main()