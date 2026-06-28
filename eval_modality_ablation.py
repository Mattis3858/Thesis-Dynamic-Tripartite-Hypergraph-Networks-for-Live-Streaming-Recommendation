"""
Modality / temporal ablation: does the model genuinely USE the streamer modality and time?

Unlike Table 13 (test-time masking of a tripartite-trained model), here each variant is TRAINED with
a channel disabled, then evaluated on the SAME full tripartite ranking task. If a variant does worse
than Full, that modality genuinely contributes.

Variants (train each first with train_tripartite_link_prediction.py, then run this script):
  * full              — the tripartite model (reference)
  * drop_streamer     — streamer channel zeroed at train+test (--drop_streamer)
  * drop_room         — room channel zeroed at train+test (--drop_room)        [expected to hurt most]
  * no_time           — time-encoding channel zeroed, recent sampling kept (--no_time)
  * no_time_uniform   — time channel zeroed AND uniform neighbour sampling (--no_time
                        --sample_neighbor_strategy uniform): fully time-blind (removes recency too)

Each variant is built with its training-time flags and (for no_time_uniform) its uniform neighbour
sampler, ranked against the SAME fixed candidate set as the main comparison. Reports Full-vs-variant
deltas and per-query paired Wilcoxon p-values (same queries/candidates), to the terminal and to
experiment_results/modality_ablation.csv (+ _paired.csv). Variants whose checkpoint is missing are
skipped (train them first).

Usage:
    python eval_modality_ablation.py --gpu 0
"""

from __future__ import annotations

import argparse
import csv as _csv
import logging
import warnings
from pathlib import Path

import numpy as np
import torch
from scipy.stats import wilcoxon

from eval_table10 import _checkpoint_exists, _resolve_checkpoint_dir, build_ht_transformer, load_ht_checkpoint
from train_tripartite_link_prediction import evaluate_tripartite_ranking
from utils.DataLoader import get_idx_data_loader, get_tripartite_link_prediction_data
from utils.metrics import mean_metric_dicts
from utils.utils import convert_to_gpu, get_neighbor_sampler, get_tripartite_neighbor_sampler

ROOT = Path(__file__).resolve().parent
HEADLINE = ["roc_auc", "average_precision", "precision@1", "ndcg@3", "precision@5", "ndcg@5", "ndcg@10", "recall@10"]


def _cell(v) -> str:
    if isinstance(v, float):
        return "nan" if np.isnan(v) else f"{v:.6f}"
    return "" if v is None else str(v)


def paired_test(ref_pq: list[dict], var_pq: list[dict], metric: str) -> dict:
    """Per-query paired Full-vs-variant on a metric (ties dropped); two-sided Wilcoxon p-value."""
    a = np.array([d.get(metric, np.nan) for d in ref_pq], dtype=np.float64)
    b = np.array([d.get(metric, np.nan) for d in var_pq], dtype=np.float64)
    keep = ~(np.isnan(a) | np.isnan(b))
    a, b = a[keep], b[keep]
    diff = a - b
    wins, losses, ties = int(np.sum(diff > 0)), int(np.sum(diff < 0)), int(np.sum(diff == 0))
    p = float("nan")
    if np.any(diff != 0):
        try:
            _s, p = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
            p = float(p)
        except ValueError:
            p = float("nan")
    return {"metric": metric, "n": int(a.size), "wins": wins, "losses": losses, "ties": ties,
            "mean_diff": float(np.mean(diff)) if a.size else float("nan"), "p_value": p}


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset_name", type=str, default="kuailive_tripartite")
    p.add_argument("--data_dir", type=str, default=None)
    p.add_argument("--val_ratio", type=float, default=0.15)
    p.add_argument("--test_ratio", type=float, default=0.15)
    p.add_argument("--batch_size", type=int, default=200)
    p.add_argument("--gpu", type=int, default=0)
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
    p.add_argument("--neg_tag", type=str, default="_neg5", help="Checkpoint name tag for train_neg_ratio (aligned protocol).")
    p.add_argument("--pop_tag", type=str, default="_popneg", help="Checkpoint name tag for popularity negatives.")
    p.add_argument("--eval_candidates_path", type=str, default=None)
    p.add_argument("--out_csv", type=str, default=str(ROOT / "experiment_results" / "modality_ablation.csv"))
    return p.parse_args()


def main() -> None:
    warnings.filterwarnings("ignore")
    args = get_args()
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu"
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("eval_modality_ablation")

    run = args.run_seed
    # Checkpoint-name order matches the trainer's suffix block: _neg5 -> ablation flag -> _popneg ->
    # _uniformsmp (drop_*/no_time are appended AFTER _neg5, before _popneg).
    NEG, POP = args.neg_tag, args.pop_tag

    def stem(abl: str, smp: str = "") -> str:
        return f"HTTransformer{NEG}{abl}{POP}{smp}_seed{run}"

    # (label, drop_streamer, drop_room, no_time, sample_strategy, checkpoint stem)
    configs = [
        ("full", False, False, False, "recent", stem("")),
        ("drop_streamer", True, False, False, "recent", stem("_drop_streamer")),
        ("drop_room", False, True, False, "recent", stem("_drop_room")),
        ("no_time", False, False, True, "recent", stem("_no_time")),
        ("no_time_uniform", False, False, True, "uniform", stem("_no_time", "_uniformsmp")),
    ]

    (
        node_raw_features, edge_raw_features, node_type_ids, full_data, train_data,
        _val, test_data, _nv, _nt, temporal_full_data, temporal_train_data,
    ) = get_tripartite_link_prediction_data(
        dataset_name=args.dataset_name, val_ratio=args.val_ratio, test_ratio=args.test_ratio, data_dir=args.data_dir,
    )
    num_nodes = len(node_type_ids)
    n_total = int(test_data.num_interactions)
    test_loader = get_idx_data_loader(indices_list=list(range(n_total)), batch_size=args.batch_size, shuffle=False)

    from utils.eval_candidates import load_eval_candidates
    cand_path = args.eval_candidates_path or str(
        ROOT / "eval_candidates" / f"{args.dataset_name}_test_joint_vw_seed{args.eval_seed + 1}.npz"
    )
    if not Path(cand_path).exists():
        raise FileNotFoundError(f"Fixed eval candidates not found: {cand_path}")
    eval_candidates = load_eval_candidates(cand_path)
    logger.info("Ranking against fixed candidates: %s", cand_path)

    # one full neighbour sampler per sampling strategy (seed=1 matches the trainer's full sampler)
    sampler_cache: dict[str, object] = {}

    def get_full_sampler(strategy: str):
        if strategy not in sampler_cache:
            if temporal_full_data is not None:
                sampler_cache[strategy] = get_neighbor_sampler(
                    data=temporal_full_data, sample_neighbor_strategy=strategy,
                    time_scaling_factor=args.time_scaling_factor, seed=1, num_nodes=num_nodes)
            else:
                sampler_cache[strategy] = get_tripartite_neighbor_sampler(
                    data=full_data, sample_neighbor_strategy=strategy,
                    time_scaling_factor=args.time_scaling_factor, seed=1, num_nodes=num_nodes)
        return sampler_cache[strategy]

    ht_kw = dict(
        node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, node_type_ids=node_type_ids,
        device=device, time_feat_dim=args.time_feat_dim, channel_embedding_dim=args.channel_embedding_dim,
        cooccurrence_dim=args.cooccurrence_dim, patch_size=args.patch_size, num_layers=args.num_layers,
        num_heads=args.num_heads, dropout=args.dropout, max_input_sequence_length=args.max_input_sequence_length,
        use_bias_gate=True, use_type_init=True, use_hetero_coocc=True,
    )

    agg_by_label: dict[str, dict] = {}
    pq_by_label: dict[str, list] = {}
    for label, ds, dr, nt, strategy, stem in configs:
        sampler = get_full_sampler(strategy)
        folder, stem = _resolve_checkpoint_dir(
            model_dir=str(ROOT / "saved_models" / "HTTransformer" / args.dataset_name / stem),
            dataset_name=args.dataset_name, save_model_name=stem,
        )
        if not _checkpoint_exists(folder, stem):
            logger.warning("SKIP %s: checkpoint not found (%s.pkl). Train it first.", label, folder / stem)
            continue
        model = build_ht_transformer(neighbor_sampler=sampler, drop_streamer=ds, drop_room=dr, no_time=nt, **ht_kw)
        load_ht_checkpoint(model, folder, stem, logger)
        model = convert_to_gpu(model, device=device)
        _loss, agg, per_query = evaluate_tripartite_ranking(
            model=model, neighbor_sampler=sampler, data=test_data, idx_data_loader=test_loader,
            node_type_ids=node_type_ids, device=device, num_negatives=args.num_ranking_negatives,
            eval_rng=np.random.RandomState(args.eval_seed + 1), return_per_query=True, eval_candidates=eval_candidates,
        )
        agg_by_label[label] = agg
        pq_by_label[label] = list(per_query)
        logger.info("%-16s N@10=%.4f P@1=%.4f AP=%.4f", label,
                    agg.get("ndcg@10", float("nan")), agg.get("precision@1", float("nan")),
                    agg.get("average_precision", float("nan")))

    if "full" not in agg_by_label:
        print("Full checkpoint missing — cannot compute deltas. Train HTTransformer (full) first.")
        return

    metric_names = [m for m in HEADLINE if any(m in a for a in agg_by_label.values())]
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = _csv.writer(f)
        w.writerow(["variant"] + metric_names)
        for label in agg_by_label:
            w.writerow([label] + [_cell(agg_by_label[label].get(m)) for m in metric_names])
    print(f"\nWrote modality-ablation metrics -> {args.out_csv}")

    full_agg = agg_by_label["full"]
    print("\n### Modality / temporal ablation (test, full set) — Full vs variant ###")
    print("| variant | N@10 | Δ(full−this) | P@1 | Δ(full−this) | AP | Δ(full−this) |")
    print("|---|---|---|---|---|---|---|")
    for label in agg_by_label:
        a = agg_by_label[label]
        def d(m):
            return full_agg.get(m, float("nan")) - a.get(m, float("nan"))
        print(f"| {label} | {a.get('ndcg@10', float('nan')):.4f} | {d('ndcg@10'):+.4f} "
              f"| {a.get('precision@1', float('nan')):.4f} | {d('precision@1'):+.4f} "
              f"| {a.get('average_precision', float('nan')):.4f} | {d('average_precision'):+.4f} |")

    # per-query paired Wilcoxon: Full vs each variant
    full_pq = pq_by_label["full"]
    pt_rows = []
    print("\n### Paired Wilcoxon (full test set): Full vs variant ###")
    print("| variant | metric | wins | losses | ties | mean Δ | p-value |")
    print("|---|---|---|---|---|---|---|")
    for label in pq_by_label:
        if label == "full":
            continue
        for metric in ("ndcg@10", "average_precision"):
            r = paired_test(full_pq, pq_by_label[label], metric)
            pstr = "nan" if np.isnan(r["p_value"]) else f"{r['p_value']:.2e}"
            print(f"| {label} | {metric} | {r['wins']} | {r['losses']} | {r['ties']} | {r['mean_diff']:+.4f} | {pstr} |")
            pt_rows.append({"variant": label, **r})
    pt_path = str(Path(args.out_csv).with_name("modality_ablation_paired.csv"))
    with open(pt_path, "w", newline="", encoding="utf-8") as f:
        w = _csv.DictWriter(f, fieldnames=["variant", "metric", "n", "wins", "losses", "ties", "mean_diff", "p_value"])
        w.writeheader()
        for r in pt_rows:
            w.writerow(r)
    print(f"\nWrote paired tests -> {pt_path}")


if __name__ == "__main__":
    main()
