"""
Table 13: Tripartite vs masked 2D link-prediction settings (same Full HT-Transformer checkpoint).

Loads saved_models/.../HTTransformer_full_seed0, runs test 1+99 ranking via
evaluate_tripartite_ranking, and masks h_v (streamer) or h_w (room) to zero before the merge head.
Prints strict top-K metrics (AUC, AP, P@1, N@3, P@5, N@5, N@10).
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

# (display name, mask streamer h_v, mask room/item h_w)
EVAL_SETTINGS: tuple[tuple[str, bool, bool], ...] = (
    ("Tripartite prediction", False, False),
    ("User-item link prediction", True, False),
    ("User-streamer link prediction", False, True),
)

# [MODIFIED] Using ultra-strict top-K metrics
TABLE13_METRIC_KEYS = ("roc_auc", "average_precision", "precision@1", "ndcg@3", "precision@5", "ndcg@5", "ndcg@10")
TABLE13_HEADERS = ("Task setting", "AUC", "AP", "P@1", "N@3", "P@5", "N@5", "N@10")

_BASE_FORWARD = HTTransformer.forward


@contextmanager
def embedding_mask_context(model: HTTransformer, *, mask_streamer: bool, mask_room: bool):
    """
    Replace ``model.forward`` so that h_v and/or h_w are zeroed before merge_layer / plain_merge_mlp.
    Restores the original forward on exit.
    """

    def masked_forward(
        self: HTTransformer,
        user_node_ids: np.ndarray,
        streamer_node_ids: np.ndarray,
        item_node_ids: np.ndarray,
        interact_times: np.ndarray,
        edge_ids: np.ndarray | None = None,
        edges_are_positive: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        del edge_ids, edges_are_positive
        h_u, h_v, h_w = self.compute_tripartite_temporal_embeddings(
            user_node_ids, streamer_node_ids, item_node_ids, interact_times
        )
        if mask_streamer:
            h_v = torch.zeros_like(h_v)
        if mask_room:
            h_w = torch.zeros_like(h_w)
        if self.use_bias_gate:
            y_hat = self.merge_layer(h_u, h_v, h_w)
        else:
            y_hat = torch.sigmoid(self.plain_merge_mlp(torch.cat([h_u, h_v, h_w], dim=-1)))
        return y_hat, h_u, h_v, h_w

    model.forward = types.MethodType(masked_forward, model)
    try:
        yield
    finally:
        model.forward = types.MethodType(_BASE_FORWARD, model)


def _fmt_metric(x: float) -> str:
    if isinstance(x, float) and np.isnan(x):
        return "—"
    return f"{x:.4f}"


def print_table13_markdown(results: dict[str, dict[str, float]]) -> None:
    rows: list[tuple[str, str, str, str, str, str, str, str]] = []
    for setting_name, _mv, _mw in EVAL_SETTINGS:
        m = results[setting_name]
        rows.append(
            (
                setting_name,
                _fmt_metric(m.get("roc_auc", float("nan"))),
                _fmt_metric(m.get("average_precision", float("nan"))),
                _fmt_metric(m.get("precision@1", float("nan"))),
                _fmt_metric(m.get("ndcg@3", float("nan"))),
                _fmt_metric(m.get("precision@5", float("nan"))),
                _fmt_metric(m.get("ndcg@5", float("nan"))),
                _fmt_metric(m.get("ndcg@10", float("nan"))),
            )
        )

    col_widths = [len(h) for h in TABLE13_HEADERS]
    for row in rows:
        for j, cell in enumerate(row):
            col_widths[j] = max(col_widths[j], len(cell))

    def _pad(cells: tuple[str, ...]) -> str:
        return "| " + " | ".join(c.ljust(col_widths[i]) for i, c in enumerate(cells)) + " |"

    sep = "|" + "|".join("-" * (w + 2) for w in col_widths) + "|"

    print("\n### Table 13 — Strict Tripartite vs link-prediction masking (test, 1+99 negatives)\n")
    print(_pad(TABLE13_HEADERS))
    print(sep)
    for row in rows:
        print(_pad(row))


def get_eval_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Table 13: tripartite vs masked 2D link prediction (Full HT-Transformer)."
    )
    parser.add_argument("--dataset_name", type=str, default="kuailive_tripartite")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    
    # [RESTORED] Back to batch_size 200 for fast inference
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--sample_neighbor_strategy",
        type=str,
        default="recent",
        choices=["uniform", "recent", "time_interval_aware"],
    )
    parser.add_argument("--time_scaling_factor", type=float, default=1e-6)
    parser.add_argument("--num_heads", type=int, default=2)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--time_feat_dim", type=int, default=100)
    parser.add_argument("--patch_size", type=int, default=1)
    parser.add_argument("--channel_embedding_dim", type=int, default=50)
    parser.add_argument("--cooccurrence_dim", type=int, default=50)
    parser.add_argument("--max_input_sequence_length", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.1)
    
    # [RESTORED] Back to 99 negatives
    parser.add_argument("--num_ranking_negatives", type=int, default=99)

    parser.add_argument("--eval_seed", type=int, default=0)
    parser.add_argument(
        "--eval_candidates_path",
        type=str,
        default=None,
        help="If set, rank against this shared fixed candidate .npz (new protocol). "
        "Default None = legacy on-the-fly uniform sampling.",
    )
    parser.add_argument(
        "--full_model_dir",
        type=str,
        default=str(DEFAULT_FULL_DIR),
        help="Directory with Full HTTransformer checkpoint (HTTransformer_seed{run_seed}.pkl).",
    )
    parser.add_argument("--run_seed", type=int, default=0)
    parser.add_argument(
        "--full_save_model_name",
        type=str,
        default=None,
        help="Checkpoint stem (default HTTransformer_seed{run_seed}).",
    )
    return parser.parse_args()


def main() -> None:
    warnings.filterwarnings("ignore")
    args = get_eval_args()
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu"

    default_stem = f"HTTransformer_seed{args.run_seed}"
    full_stem = args.full_save_model_name or default_stem
    full_folder, full_stem = _resolve_checkpoint_dir(
        model_dir=args.full_model_dir,
        dataset_name=args.dataset_name,
        save_model_name=full_stem,
    )
    if not _checkpoint_exists(full_folder, full_stem):
        raise FileNotFoundError(
            f"Full model checkpoint not found: {full_folder / (full_stem + '.pkl')}\n"
            "Train and copy to HTTransformer_full_seed0, or pass --full_model_dir."
        )
    print(f"Full model: {full_folder / (full_stem + '.pkl')}")

    (
        node_raw_features,
        edge_raw_features,
        node_type_ids,
        full_data,
        train_data,
        _val_data,
        test_data,
        _new_node_val_data,
        _new_node_test_data,
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

    test_loader = get_idx_data_loader(
        indices_list=list(range(len(test_data.user_node_ids))),
        batch_size=args.batch_size,
        shuffle=False,
    )
    eval_rank_seed = args.eval_seed + 1

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("eval_table13")

    model = build_ht_transformer(
        node_raw_features=node_raw_features,
        edge_raw_features=edge_raw_features,
        node_type_ids=node_type_ids,
        neighbor_sampler=train_neighbor_sampler,
        device=device,
        use_bias_gate=True,
        use_type_init=True,
        use_hetero_coocc=True,
        time_feat_dim=args.time_feat_dim,
        channel_embedding_dim=args.channel_embedding_dim,
        cooccurrence_dim=args.cooccurrence_dim,
        patch_size=args.patch_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        max_input_sequence_length=args.max_input_sequence_length,
    )
    load_ht_checkpoint(model, full_folder, full_stem, logger)
    model = convert_to_gpu(model, device=device)

    if not model.use_bias_gate:
        logger.warning(
            "Checkpoint was trained with use_bias_gate=False; Table 13 masking still uses loaded head."
        )

    eval_candidates = None
    if args.eval_candidates_path:
        from utils.eval_candidates import load_eval_candidates

        eval_candidates = load_eval_candidates(args.eval_candidates_path)

    results: dict[str, dict[str, float]] = {}

    for setting_name, mask_v, mask_w in EVAL_SETTINGS:
        print(f"\n=== {setting_name} (mask h_v={mask_v}, mask h_w={mask_w}) ===")
        test_eval_rng = np.random.RandomState(seed=eval_rank_seed)
        with embedding_mask_context(model, mask_streamer=mask_v, mask_room=mask_w):
            _loss, agg = evaluate_tripartite_ranking(
                model=model,
                neighbor_sampler=full_neighbor_sampler,
                data=test_data,
                idx_data_loader=test_loader,
                node_type_ids=node_type_ids,
                device=device,
                num_negatives=args.num_ranking_negatives,
                eval_rng=test_eval_rng,
                return_per_query=False,
                eval_candidates=eval_candidates,
            )
        results[setting_name] = agg
        print(
            f"  AUC={agg.get('roc_auc', float('nan')):.4f}  "
            f"AP={agg.get('average_precision', float('nan')):.4f}  "
            f"P@1={agg.get('precision@1', float('nan')):.4f}  "
            f"NDCG@3={agg.get('ndcg@3', float('nan')):.4f}  "
            f"P@5={agg.get('precision@5', float('nan')):.4f}  "
            f"NDCG@5={agg.get('ndcg@5', float('nan')):.4f}  "
            f"NDCG@10={agg.get('ndcg@10', float('nan')):.4f}"
        )

    print_table13_markdown(results)


if __name__ == "__main__":
    main()