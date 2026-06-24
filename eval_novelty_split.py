"""
Novelty-split evaluation: does each HT-Transformer component help on NON-continuation queries?

The overall ranking metric saturates (~0.99) because most test queries are session continuations
(the true (streamer, room) is already in the user's recent history), which any model nails. This
masks the value of the type-aware init / 3D co-occurrence / bias-gate components. Here we split the
test set into:
  * continuation: the (v, w) combo HAS appeared in this user's history strictly before t.
  * novel:        the (v, w) combo has NEVER appeared for this user before t (first-time combo).
and report each model on both subsets, so a component's contribution can show up where it matters.

The novelty mask is model-independent (data + time only, leakage-free: history strictly < t) and is
computed once. ``--mask_only`` prints the split sizes without running any model (cheap sanity check).

Models evaluated (default): Full + each single-component-off ablation, loaded from the checkpoints
run_experiments.py already produced (no retraining). All rank against the SAME fixed candidate set
used by the main comparison.

Usage:
    python eval_novelty_split.py --mask_only          # just report split sizes
    python eval_novelty_split.py --gpu 0              # full evaluation -> experiment_results/novelty_split.csv
"""

from __future__ import annotations

import argparse
import csv as _csv
import logging
import warnings
from bisect import bisect_left
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from eval_table10 import _checkpoint_exists, _resolve_checkpoint_dir, build_ht_transformer, load_ht_checkpoint
from train_tripartite_link_prediction import evaluate_tripartite_ranking
from utils.DataLoader import get_idx_data_loader, get_tripartite_link_prediction_data
from utils.metrics import mean_metric_dicts
from utils.utils import convert_to_gpu, get_neighbor_sampler, get_tripartite_neighbor_sampler

ROOT = Path(__file__).resolve().parent

# (label, use_bias_gate, use_type_init, use_hetero_coocc, ablation_suffix)
# ablation_suffix must match train_tripartite_link_prediction.py's naming (flags in this order:
# bias_gate, type_init, hetero_coocc), placed BEFORE the shared _neg5_popneg / _seed tail.
DEFAULT_CONFIGS: tuple[tuple[str, bool, bool, bool, str], ...] = (
    ("full", True, True, True, ""),
    ("no_bias_gate", False, True, True, "_no_bias_gate"),
    ("no_type_init", True, False, True, "_no_type_init"),
    ("no_hetero_coocc", True, True, False, "_no_hetero_coocc"),
)

HEADLINE = ["roc_auc", "average_precision", "precision@1", "ndcg@3", "precision@5", "ndcg@5", "ndcg@10", "recall@10"]


def build_novelty_mask(test_data, full_data) -> np.ndarray:
    """Boolean [num_test]; True = the (v,w) combo never appeared for this user strictly before t."""
    user_combo_times: dict[int, dict[tuple[int, int], list[float]]] = defaultdict(lambda: defaultdict(list))
    for u, v, w, t in zip(
        full_data.user_node_ids, full_data.streamer_node_ids, full_data.item_node_ids, full_data.node_interact_times
    ):
        user_combo_times[int(u)][(int(v), int(w))].append(float(t))
    for u in user_combo_times:
        for c in user_combo_times[u]:
            user_combo_times[u][c].sort()

    novel = np.zeros(test_data.num_interactions, dtype=bool)
    for i in range(test_data.num_interactions):
        u = int(test_data.user_node_ids[i])
        v = int(test_data.streamer_node_ids[i])
        w = int(test_data.item_node_ids[i])
        t = float(test_data.node_interact_times[i])
        times = user_combo_times.get(u, {}).get((v, w), [])
        # bisect_left == count of occurrences strictly before t (the query itself sits at t)
        novel[i] = bisect_left(times, t) == 0
    return novel


def _ordered_metrics(metric_dicts: list[dict]) -> list[str]:
    present: list[str] = []
    for d in metric_dicts:
        for k in d:
            if k not in present:
                present.append(k)
    return [m for m in HEADLINE if m in present] + [m for m in sorted(present) if m not in HEADLINE]


def _cell(v) -> str:
    if isinstance(v, float):
        return "nan" if np.isnan(v) else f"{v:.6f}"
    return "" if v is None else str(v)


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset_name", type=str, default="kuailive_tripartite")
    p.add_argument("--data_dir", type=str, default=None)
    p.add_argument("--val_ratio", type=float, default=0.15)
    p.add_argument("--test_ratio", type=float, default=0.15)
    p.add_argument("--mask_only", action="store_true", help="Only compute & print the split sizes; run no models.")
    p.add_argument("--batch_size", type=int, default=200)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--sample_neighbor_strategy", type=str, default="recent",
                   choices=["uniform", "recent", "time_interval_aware"])
    p.add_argument("--time_scaling_factor", type=float, default=1e-6)
    p.add_argument("--num_heads", type=int, default=2)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--time_feat_dim", type=int, default=100)
    p.add_argument("--patch_size", type=int, default=1)
    p.add_argument("--channel_embedding_dim", type=int, default=50)
    p.add_argument("--cooccurrence_dim", type=int, default=50)
    p.add_argument("--max_input_sequence_length", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--num_ranking_negatives", type=int, default=99)
    p.add_argument("--eval_seed", type=int, default=0)
    p.add_argument("--run_seed", type=int, default=0)
    p.add_argument("--ckpt_suffix", type=str, default="_neg5_popneg",
                   help="Shared checkpoint-name tail after the ablation suffix (matches the aligned protocol).")
    p.add_argument("--eval_candidates_path", type=str, default=None,
                   help="Fixed candidate .npz; default auto-resolves to the main comparison's file.")
    p.add_argument("--out_csv", type=str, default=str(ROOT / "experiment_results" / "novelty_split.csv"))
    return p.parse_args()


def main() -> None:
    warnings.filterwarnings("ignore")
    args = get_args()
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu"
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("eval_novelty_split")

    (
        node_raw_features, edge_raw_features, node_type_ids, full_data, train_data,
        _val_data, test_data, _nv, _nt, temporal_full_data, temporal_train_data,
    ) = get_tripartite_link_prediction_data(
        dataset_name=args.dataset_name, val_ratio=args.val_ratio, test_ratio=args.test_ratio, data_dir=args.data_dir,
    )

    novel = build_novelty_mask(test_data, full_data)
    n_total = int(test_data.num_interactions)
    n_novel = int(novel.sum())
    n_cont = n_total - n_novel
    print(f"\n=== Novelty split (test queries = {n_total}) ===")
    print(f"  continuation (combo seen before t): {n_cont}  ({100*n_cont/n_total:.1f}%)")
    print(f"  novel        (first-time combo)   : {n_novel}  ({100*n_novel/n_total:.1f}%)")
    if args.mask_only:
        print("\n--mask_only: skipping model evaluation. Re-run without it once the novel subset looks big enough.")
        return
    if n_novel == 0:
        print("\nNo novel queries -- nothing to differentiate. Stopping.")
        return

    num_nodes = len(node_type_ids)
    if temporal_full_data is not None and temporal_train_data is not None:
        train_ns = get_neighbor_sampler(data=temporal_train_data, sample_neighbor_strategy=args.sample_neighbor_strategy,
                                        time_scaling_factor=args.time_scaling_factor, seed=0, num_nodes=num_nodes)
        full_ns = get_neighbor_sampler(data=temporal_full_data, sample_neighbor_strategy=args.sample_neighbor_strategy,
                                       time_scaling_factor=args.time_scaling_factor, seed=1, num_nodes=num_nodes)
    else:
        train_ns = get_tripartite_neighbor_sampler(data=train_data, sample_neighbor_strategy=args.sample_neighbor_strategy,
                                                   time_scaling_factor=args.time_scaling_factor, seed=0, num_nodes=num_nodes)
        full_ns = get_tripartite_neighbor_sampler(data=full_data, sample_neighbor_strategy=args.sample_neighbor_strategy,
                                                  time_scaling_factor=args.time_scaling_factor, seed=1, num_nodes=num_nodes)

    test_loader = get_idx_data_loader(indices_list=list(range(n_total)), batch_size=args.batch_size, shuffle=False)

    from utils.eval_candidates import load_eval_candidates
    cand_path = args.eval_candidates_path or str(
        ROOT / "eval_candidates" / f"{args.dataset_name}_test_joint_vw_seed{args.eval_seed + 1}.npz"
    )
    if not Path(cand_path).exists():
        raise FileNotFoundError(f"Fixed eval candidates not found: {cand_path} (build via training or build_eval_candidates.py)")
    eval_candidates = load_eval_candidates(cand_path)
    logger.info("Ranking against fixed candidates: %s", cand_path)

    ht_kw = dict(
        node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, node_type_ids=node_type_ids,
        neighbor_sampler=train_ns, device=device, time_feat_dim=args.time_feat_dim,
        channel_embedding_dim=args.channel_embedding_dim, cooccurrence_dim=args.cooccurrence_dim,
        patch_size=args.patch_size, num_layers=args.num_layers, num_heads=args.num_heads,
        dropout=args.dropout, max_input_sequence_length=args.max_input_sequence_length,
    )

    rows: list[tuple[str, str, int, dict]] = []  # (label, split, n, agg)
    for label, bg, ti, co, abl in DEFAULT_CONFIGS:
        stem = f"HTTransformer{abl}{args.ckpt_suffix}_seed{args.run_seed}"
        folder, stem = _resolve_checkpoint_dir(
            model_dir=str(ROOT / "saved_models" / "HTTransformer" / args.dataset_name / stem),
            dataset_name=args.dataset_name, save_model_name=stem,
        )
        if not _checkpoint_exists(folder, stem):
            logger.warning("SKIP %s: checkpoint not found at %s", label, folder / f"{stem}.pkl")
            continue
        model = build_ht_transformer(use_bias_gate=bg, use_type_init=ti, use_hetero_coocc=co, **ht_kw)
        load_ht_checkpoint(model, folder, stem, logger)
        model = convert_to_gpu(model, device=device)

        _loss, _agg, per_query = evaluate_tripartite_ranking(
            model=model, neighbor_sampler=full_ns, data=test_data, idx_data_loader=test_loader,
            node_type_ids=node_type_ids, device=device, num_negatives=args.num_ranking_negatives,
            eval_rng=np.random.RandomState(args.eval_seed + 1), return_per_query=True, eval_candidates=eval_candidates,
        )
        pq = list(per_query)
        novel_m = [pq[i] for i in range(n_total) if novel[i]]
        cont_m = [pq[i] for i in range(n_total) if not novel[i]]
        rows.append((label, "all", n_total, mean_metric_dicts(pq)))
        rows.append((label, "continuation", n_cont, mean_metric_dicts(cont_m)))
        rows.append((label, "novel", n_novel, mean_metric_dicts(novel_m)))
        nm = rows[-1][3]
        logger.info("%-16s NOVEL: ndcg@10=%.4f p@1=%.4f ap=%.4f",
                    label, nm.get("ndcg@10", float("nan")), nm.get("precision@1", float("nan")),
                    nm.get("average_precision", float("nan")))

    if not rows:
        print("No checkpoints evaluated. Check --ckpt_suffix / --run_seed and that ablation runs completed.")
        return

    metric_names = _ordered_metrics([r[3] for r in rows])
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = _csv.writer(f)
        w.writerow(["model", "split", "n_queries"] + metric_names)
        for label, split, n, agg in rows:
            w.writerow([label, split, n] + [_cell(agg.get(m)) for m in metric_names])
    print(f"\nWrote novelty-split metrics -> {args.out_csv}")

    # focused comparison: Full vs each ablation on the NOVEL subset (the point of this script)
    novel_by_label = {label: agg for (label, split, _n, agg) in rows if split == "novel"}
    if "full" in novel_by_label:
        full_n10 = novel_by_label["full"].get("ndcg@10", float("nan"))
        full_p1 = novel_by_label["full"].get("precision@1", float("nan"))
        print(f"\n### NOVEL subset (n={n_novel}) — Full vs ablations ###")
        print("| model | N@10 | Δ(full-this) | P@1 | Δ(full-this) |")
        print("|---|---|---|---|---|")
        for label in novel_by_label:
            a = novel_by_label[label]
            n10, p1 = a.get("ndcg@10", float("nan")), a.get("precision@1", float("nan"))
            print(f"| {label} | {n10:.4f} | {full_n10 - n10:+.4f} | {p1:.4f} | {full_p1 - p1:+.4f} |")


if __name__ == "__main__":
    main()
