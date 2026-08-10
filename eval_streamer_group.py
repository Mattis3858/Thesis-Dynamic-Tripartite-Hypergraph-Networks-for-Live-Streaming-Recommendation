"""
Group-wise model comparison across STREAMER bias groups (committee follow-up #2).

Answers "is the model general, or does it only help one kind of streamer?" by evaluating
HT-Transformer and the strongest dynamic baselines on the same test set, stratified by the
streamer bias groups from ``analyze_streamer_bias.py``.

Protocol notes
  * Same fixed shared candidate set as the main comparison, so numbers stay comparable.
  * 5 seeds per model; the per-(group, model) table reports mean +/- std over seeds.
  * Significance: each test query's metric is first averaged over the seeds, then a paired
    Wilcoxon signed-rank test (HT-Transformer vs. each baseline) is run inside each group,
    together with a bootstrap CI on the mean difference. Per-seed p-values are also written out.
  * Reported metrics are strict top-rank ones (P@1 = HR@1 = NDCG@1 in this leave-one-out
    setting, N@3, N@5, N@10); broad retrieval saturates and cannot separate the models.

Streamer ids in the bias CSV are RAW edge-table ids, while the dataset uses remapped global
node ids. The mapping is rebuilt with the converter's own id-map function, from the edge CSV
recorded in the dataset's ``*_tripartite_meta.json`` (override with ``--converter_edges``).

Usage:
  python eval_streamer_group.py --dataset_name kuailive_streamer_groups \
      --streamer_groups experiment_results/streamer_bias_groups.csv \
      --models HTTransformer CAWN GraphMixer DyGFormer --run_seeds 0 1 2 3 4
"""

from __future__ import annotations

import argparse
import csv as _csv
import json
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon

from eval_table10 import build_ht_transformer
from models.BaselineTripartiteWrapper import BaselineTripartiteWrapper, TRIPARTITE_BASELINE_MODELS
from train_tripartite_link_prediction import evaluate_tripartite_ranking
from utils.DataLoader import get_idx_data_loader, get_tripartite_link_prediction_data
from utils.EarlyStopping import EarlyStopping
from utils.metrics import mean_metric_dicts
from utils.roles import detect_roles, role_ids
from utils.utils import convert_to_gpu, get_neighbor_sampler, get_tripartite_neighbor_sampler

sys.path.append(str(Path(__file__).resolve().parent / "preprocess_data"))

ROOT = Path(__file__).resolve().parent
DEFAULT_GROUPS_CSV = ROOT / "experiment_results" / "streamer_bias_groups.csv"
GROUP_UNKNOWN = "Unassigned"

METRIC_KEYS = ("precision@1", "ndcg@3", "ndcg@5", "ndcg@10")
METRIC_HEADERS = ("P@1", "N@3", "N@5", "N@10")
TEST_METRICS = ("precision@1", "ndcg@10")  # metrics carried through the significance tests


# --------------------------------------------------------------------------------------
# streamer id mapping and grouping
# --------------------------------------------------------------------------------------
def build_raw_to_global_streamer_map(dataset_name: str, data_dir: str | None,
                                     converter_edges: str | None) -> dict[int, int]:
    """Rebuild raw->global streamer ids exactly as preprocess_kuailive_tripartite did."""
    from preprocess_kuailive_tripartite import (  # noqa: E402  (path appended above)
        _build_global_id_maps_from_edge_table,
        _dedupe_labeled_edges,
    )

    if converter_edges is None:
        base = Path(data_dir) if data_dir else ROOT / "processed_data" / dataset_name
        meta_path = base / f"{dataset_name}_tripartite_meta.json"
        if not meta_path.is_file():
            raise FileNotFoundError(
                f"{meta_path} not found; pass --converter_edges with the edge CSV used to build the dataset."
            )
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        converter_edges = meta.get("source_edges_csv")
        if not converter_edges or not Path(converter_edges).is_file():
            raise FileNotFoundError(
                f"The edge CSV recorded in {meta_path} ({converter_edges}) is not readable here; "
                "pass --converter_edges explicitly."
            )

    df = _dedupe_labeled_edges(pd.read_csv(converter_edges))
    _user_map, streamer_map, _room_map, *_ = _build_global_id_maps_from_edge_table(df)
    print(f"Rebuilt streamer id map from {converter_edges} ({len(streamer_map)} streamers).")
    return {int(k): int(v) for k, v in streamer_map.items()}


def load_streamer_group_map(csv_path: Path, raw_to_global: dict[int, int]) -> tuple[dict[int, str], list[str]]:
    df = pd.read_csv(csv_path)
    for col in ("streamer_id", "bias_group"):
        if col not in df.columns:
            raise ValueError(f"{csv_path} must contain columns streamer_id and bias_group.")

    mapping: dict[int, str] = {}
    missing = 0
    for row in df.itertuples(index=False):
        gid = raw_to_global.get(int(row.streamer_id))
        if gid is None:
            missing += 1
            continue
        mapping[gid] = str(row.bias_group)

    if missing:
        print(f"Note: {missing} clustered streamers are absent from this dataset (expected -- the "
              f"dataset is a sampled subset of the clustered slice).")
    covered = len(mapping)
    if covered == 0:
        raise RuntimeError(
            "No streamer in the bias CSV maps into this dataset. The CSV and the dataset were most "
            "likely built from different edge tables."
        )
    print(f"Mapped {covered} dataset streamers to bias groups.")

    order = sorted({g for g in mapping.values()})
    return mapping, order + [GROUP_UNKNOWN]


def group_indices(test_data, group_map: dict[int, str], group_order: list[str],
                  streamer_ids: np.ndarray) -> dict[str, list[int]]:
    known = set(group_order) - {GROUP_UNKNOWN}
    idx: dict[str, list[int]] = {g: [] for g in group_order}
    for i in range(test_data.num_interactions):
        g = group_map.get(int(streamer_ids[i]), GROUP_UNKNOWN)
        idx[g if g in known else GROUP_UNKNOWN].append(i)
    return idx


# --------------------------------------------------------------------------------------
# checkpoints
# --------------------------------------------------------------------------------------
def resolve_checkpoint(model_name: str, dataset_name: str, seed: int, suffix: str | None) -> tuple[Path, str]:
    """Locate saved_models/<model>/<dataset>/<model><suffix>_seed<seed>/<stem>.pkl."""
    base = ROOT / "saved_models" / model_name / dataset_name
    if suffix is not None:
        stem = f"{model_name}{suffix}_seed{seed}"
        folder = base / stem
        if not (folder / f"{stem}.pkl").is_file():
            raise FileNotFoundError(f"Checkpoint not found: {folder / (stem + '.pkl')}")
        return folder, stem

    candidates = sorted(p for p in base.glob(f"{model_name}*_seed{seed}") if (p / f"{p.name}.pkl").is_file())
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint matching {base / (model_name + '*_seed' + str(seed))}/<same>.pkl. "
            "Train the model first or pass an explicit --suffix_<model>."
        )
    if len(candidates) > 1:
        names = "\n  ".join(p.name for p in candidates)
        raise RuntimeError(
            f"Multiple checkpoints for {model_name} seed {seed}; disambiguate with a suffix flag:\n  {names}"
        )
    return candidates[0], candidates[0].name


def load_checkpoint(model, folder: Path, stem: str, model_name: str, logger: logging.Logger) -> None:
    EarlyStopping(
        patience=0, save_model_folder=str(folder), save_model_name=stem,
        logger=logger, model_name=model_name,
    ).load_checkpoint(model, map_location="cpu")


def build_model(model_name: str, args, *, node_raw_features, edge_raw_features, node_type_ids,
                neighbor_sampler, device: str):
    if model_name == "HTTransformer":
        return build_ht_transformer(
            node_raw_features=node_raw_features, edge_raw_features=edge_raw_features,
            node_type_ids=node_type_ids, neighbor_sampler=neighbor_sampler, device=device,
            use_bias_gate=True, use_type_init=True, use_hetero_coocc=True,
            time_feat_dim=args.time_feat_dim, channel_embedding_dim=args.channel_embedding_dim,
            cooccurrence_dim=args.cooccurrence_dim, patch_size=args.patch_size,
            num_layers=args.num_layers, num_heads=args.num_heads, dropout=args.dropout,
            max_input_sequence_length=args.max_input_sequence_length,
        )
    if model_name in TRIPARTITE_BASELINE_MODELS:
        return BaselineTripartiteWrapper(
            model_name=model_name, node_raw_features=node_raw_features,
            edge_raw_features=edge_raw_features, neighbor_sampler=neighbor_sampler, device=device,
            time_feat_dim=args.time_feat_dim, num_layers=args.num_layers, num_heads=args.num_heads,
            dropout=args.dropout, num_neighbors=args.num_neighbors, patch_size=args.patch_size,
            max_input_sequence_length=args.max_input_sequence_length,
            channel_embedding_dim=args.channel_embedding_dim, time_gap=args.time_gap,
            walk_length=args.walk_length, position_feat_dim=args.position_feat_dim,
            num_walk_heads=args.num_walk_heads, num_depths=args.num_depths,
        )
    raise ValueError(f"Unsupported model for this script: {model_name}")


# --------------------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------------------
def seed_averaged_per_query(per_query_by_seed: list[list[dict]], metric: str) -> np.ndarray:
    """One value per test query: its metric averaged over the seeds."""
    mat = np.array([[q.get(metric, np.nan) for q in pq] for pq in per_query_by_seed], dtype=np.float64)
    return np.nanmean(mat, axis=0)


def paired_test(a: np.ndarray, b: np.ndarray, n_boot: int, rng: np.random.RandomState) -> dict:
    """Paired Wilcoxon plus a bootstrap CI on the mean difference (a - b)."""
    diff = a - b
    n = len(diff)
    out = {
        "n": n,
        "mean_diff": float(np.mean(diff)) if n else float("nan"),
        "wins": int(np.sum(diff > 0)),
        "losses": int(np.sum(diff < 0)),
        "ties": int(np.sum(diff == 0)),
        "p_value": float("nan"),
        "ci_low": float("nan"),
        "ci_high": float("nan"),
    }
    if n == 0 or np.all(diff == 0):
        return out
    try:
        out["p_value"] = float(wilcoxon(a, b, zero_method="wilcox").pvalue)
    except ValueError:
        pass
    boot = np.array([np.mean(rng.choice(diff, size=n, replace=True)) for _ in range(n_boot)])
    out["ci_low"], out["ci_high"] = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))
    return out


# --------------------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------------------
def _fmt(mean: float, std: float) -> str:
    if np.isnan(mean):
        return "--"
    return f"{mean:.4f}+/-{std:.4f}"


def print_group_table(results: dict, group_order: list[str], models: list[str],
                      counts: dict[str, int]) -> None:
    headers = ("Group", "n", "Model", *METRIC_HEADERS)
    rows = []
    for g in group_order:
        if counts.get(g, 0) == 0:
            continue
        for m in models:
            cells = [_fmt(*results[g][m][k]) for k in METRIC_KEYS]
            rows.append((g, str(counts[g]), m, *cells))

    widths = [len(h) for h in headers]
    for row in rows:
        for j, c in enumerate(row):
            widths[j] = max(widths[j], len(c))

    def _pad(cells) -> str:
        return "| " + " | ".join(str(c).ljust(widths[i]) for i, c in enumerate(cells)) + " |"

    print("\n### Streamer-group ranking performance (test, mean+/-std over seeds)\n")
    print(_pad(headers))
    print("|" + "|".join("-" * (w + 2) for w in widths) + "|")
    for row in rows:
        print(_pad(row))


def print_significance(sig_rows: list[dict]) -> None:
    print("\n### HT-Transformer vs. baselines, per streamer group (seed-averaged per-query, paired)\n")
    print("| Group | Baseline | Metric | n | mean diff | 95% CI | wins/losses | p |")
    print("|-------|----------|--------|---|--------|--------|-------------|---|")
    for r in sig_rows:
        print(
            f"| {r['group']} | {r['baseline']} | {r['metric']} | {r['n']} | {r['mean_diff']:+.4f} "
            f"| [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] | {r['wins']}/{r['losses']} | {r['p_value']:.3g} |"
        )


def write_csvs(results: dict, group_order: list[str], models: list[str], counts: dict[str, int],
               per_seed_rows: list[dict], sig_rows: list[dict], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = _csv.writer(f)
        w.writerow(["group", "model", "n_queries"]
                   + [f"{m}_{s}" for m in METRIC_KEYS for s in ("mean", "std")])
        for g in group_order:
            for m in models:
                vals = []
                for k in METRIC_KEYS:
                    mean, std = results[g][m][k]
                    vals += [f"{mean:.6f}", f"{std:.6f}"]
                w.writerow([g, m, counts.get(g, 0)] + vals)
    print(f"\nWrote group metrics    -> {out_csv}")

    per_seed_path = out_csv.with_name(out_csv.stem + "_per_seed.csv")
    pd.DataFrame(per_seed_rows).to_csv(per_seed_path, index=False)
    print(f"Wrote per-seed metrics -> {per_seed_path}")

    sig_path = out_csv.with_name(out_csv.stem + "_significance.csv")
    pd.DataFrame(sig_rows).to_csv(sig_path, index=False)
    print(f"Wrote significance     -> {sig_path}")


def plot_groups(results: dict, group_order: list[str], models: list[str], counts: dict[str, int],
                out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = [g for g in group_order if counts.get(g, 0) > 0]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    width = 0.8 / max(len(models), 1)
    x = np.arange(len(groups))

    for ax, metric, title in ((axes[0], "precision@1", "HR@1 by streamer group"),
                              (axes[1], "ndcg@10", "NDCG@10 by streamer group")):
        for i, m in enumerate(models):
            means = [results[g][m][metric][0] for g in groups]
            stds = [results[g][m][metric][1] for g in groups]
            ax.bar(x + i * width, means, width, yerr=stds, capsize=3, label=m)
        ax.set_xticks(x + width * (len(models) - 1) / 2)
        ax.set_xticklabels(groups, rotation=15, ha="right")
        ax.set_ylabel(metric)
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    p = out_dir / "fig_streamer_group_performance.pdf"
    plt.savefig(p, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"Wrote figure           -> {p}")


# --------------------------------------------------------------------------------------
def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stratified test ranking across streamer bias groups.")
    p.add_argument("--dataset_name", type=str, default="kuailive_streamer_groups")
    p.add_argument("--data_dir", type=str, default=None)
    p.add_argument("--streamer_groups", type=str, default=str(DEFAULT_GROUPS_CSV))
    p.add_argument("--converter_edges", type=str, default=None,
                   help="Edge CSV the dataset was built from; default: read from the dataset meta json.")
    p.add_argument("--group_role", type=str, default="auto", choices=("auto", "streamer", "item"),
                   help="Which node column carries the streamers to group by. 'auto' detects it "
                        "from the data (utils/roles.py) because some datasets have the streamer and "
                        "room columns swapped; 'streamer'/'item' force a column.")
    p.add_argument("--models", type=str, nargs="+",
                   default=["HTTransformer", "CAWN", "GraphMixer", "DyGFormer"])
    p.add_argument("--run_seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--suffix", type=str, nargs="*", default=None,
                   help="Checkpoint suffix per model, e.g. --suffix HTTransformer=_neg5_popneg CAWN=_popneg. "
                        "Omitted models are auto-discovered (fails loudly if ambiguous).")
    p.add_argument("--val_ratio", type=float, default=0.15)
    p.add_argument("--test_ratio", type=float, default=0.15)
    p.add_argument("--batch_size", type=int, default=200)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--sample_neighbor_strategy", type=str, default="recent")
    p.add_argument("--time_scaling_factor", type=float, default=1e-6)
    p.add_argument("--num_heads", type=int, default=2)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--time_feat_dim", type=int, default=100)
    p.add_argument("--patch_size", type=int, default=1)
    p.add_argument("--channel_embedding_dim", type=int, default=50)
    p.add_argument("--cooccurrence_dim", type=int, default=50)
    p.add_argument("--max_input_sequence_length", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--num_neighbors", type=int, default=20)
    p.add_argument("--time_gap", type=int, default=2000)
    p.add_argument("--walk_length", type=int, default=1)
    p.add_argument("--position_feat_dim", type=int, default=172)
    p.add_argument("--num_walk_heads", type=int, default=8)
    p.add_argument("--num_depths", type=int, default=21)
    p.add_argument("--num_ranking_negatives", type=int, default=99)
    p.add_argument("--eval_seed", type=int, default=0)
    p.add_argument("--eval_candidates_path", type=str, default=None)
    p.add_argument("--no_fixed_eval_candidates", action="store_true",
                   help="Legacy on-the-fly uniform negatives; NOT comparable to the main table.")
    p.add_argument("--n_bootstrap", type=int, default=2000)
    p.add_argument("--no_plots", action="store_true")
    p.add_argument("--out_csv", type=str,
                   default=str(ROOT / "experiment_results" / "streamer_group_metrics.csv"))
    return p.parse_args()


def parse_suffixes(raw: list[str] | None) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in raw or []:
        if "=" not in item:
            raise ValueError(f"--suffix entries must look like Model=_suffix; got '{item}'")
        k, v = item.split("=", 1)
        out[k] = v
    return out


def main() -> None:
    warnings.filterwarnings("ignore")
    args = get_args()
    suffixes = parse_suffixes(args.suffix)
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu"
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("eval_streamer_group")

    raw_to_global = build_raw_to_global_streamer_map(args.dataset_name, args.data_dir, args.converter_edges)
    group_map, group_order = load_streamer_group_map(Path(args.streamer_groups), raw_to_global)

    (
        node_raw_features, edge_raw_features, node_type_ids, full_data, train_data,
        _val_data, test_data, _nn_val, _nn_test, temporal_full_data, temporal_train_data,
    ) = get_tripartite_link_prediction_data(
        dataset_name=args.dataset_name, val_ratio=args.val_ratio,
        test_ratio=args.test_ratio, data_dir=args.data_dir,
    )

    num_nodes = len(node_type_ids)
    if temporal_full_data is not None and temporal_train_data is not None:
        train_sampler = get_neighbor_sampler(data=temporal_train_data, sample_neighbor_strategy=args.sample_neighbor_strategy, time_scaling_factor=args.time_scaling_factor, seed=0, num_nodes=num_nodes)
        full_sampler = get_neighbor_sampler(data=temporal_full_data, sample_neighbor_strategy=args.sample_neighbor_strategy, time_scaling_factor=args.time_scaling_factor, seed=1, num_nodes=num_nodes)
    else:
        train_sampler = get_tripartite_neighbor_sampler(data=train_data, sample_neighbor_strategy=args.sample_neighbor_strategy, time_scaling_factor=args.time_scaling_factor, seed=0, num_nodes=num_nodes)
        full_sampler = get_tripartite_neighbor_sampler(data=full_data, sample_neighbor_strategy=args.sample_neighbor_strategy, time_scaling_factor=args.time_scaling_factor, seed=1, num_nodes=num_nodes)

    test_loader = get_idx_data_loader(indices_list=list(range(len(test_data.user_node_ids))),
                                      batch_size=args.batch_size, shuffle=False)
    eval_rank_seed = args.eval_seed + 1

    eval_candidates = None
    if args.no_fixed_eval_candidates:
        logger.warning("Using LEGACY uniform negatives; not comparable to the main table.")
    else:
        from utils.eval_candidates import load_eval_candidates

        cand_path = args.eval_candidates_path or str(
            ROOT / "eval_candidates" / f"{args.dataset_name}_test_joint_vw_seed{eval_rank_seed}.npz"
        )
        if not Path(cand_path).exists():
            raise FileNotFoundError(
                f"Fixed eval candidates not found: {cand_path}\n"
                "Run training once (auto-builds) or `python build_eval_candidates.py`, or pass "
                "--eval_candidates_path."
            )
        eval_candidates = load_eval_candidates(cand_path)
        logger.info("Ranking against fixed candidates: %s", cand_path)

    if args.group_role == "auto":
        mapping = detect_roles(full_data)
        streamer_ids = role_ids(test_data, mapping, "streamer")
    else:
        attr = "streamer_node_ids" if args.group_role == "streamer" else "item_node_ids"
        print(f"Grouping by data.{attr} (forced via --group_role {args.group_role}).")
        streamer_ids = np.asarray(getattr(test_data, attr), dtype=np.int64)

    idx_by_group = group_indices(test_data, group_map, group_order, streamer_ids)
    counts = {g: len(v) for g, v in idx_by_group.items()}
    print("\nTest queries per streamer group: "
          + ", ".join(f"{g}={counts[g]}" for g in group_order))

    if sum(counts[g] for g in group_order if g != GROUP_UNKNOWN) == 0:
        raise RuntimeError(
            "Every test query landed in '" + GROUP_UNKNOWN + "', so no group-wise comparison is "
            "possible. The bias CSV's streamer ids do not match the ids this dataset groups by.\n"
            "  * Check that analyze_streamer_bias.py ran with the correct --etype_convention: with "
            "the wrong one it profiles ROOMS and calls them streamers.\n"
            "  * Check --group_role (currently '" + args.group_role + "') against the role "
            "detection printed above."
        )

    # ---- evaluate every model on every seed -------------------------------------------
    per_query_by_model: dict[str, list[list[dict]]] = {}
    per_seed_rows: list[dict] = []

    for model_name in args.models:
        per_query_by_model[model_name] = []
        for seed in args.run_seeds:
            folder, stem = resolve_checkpoint(model_name, args.dataset_name, seed, suffixes.get(model_name))
            model = build_model(model_name, args, node_raw_features=node_raw_features,
                                edge_raw_features=edge_raw_features, node_type_ids=node_type_ids,
                                neighbor_sampler=train_sampler, device=device)
            load_checkpoint(model, folder, stem, model_name, logger)
            model = convert_to_gpu(model, device=device)

            print(f"Evaluating {model_name} seed {seed}  ({stem}) ...")
            _loss, _agg, per_query = evaluate_tripartite_ranking(
                model=model, neighbor_sampler=full_sampler, data=test_data,
                idx_data_loader=test_loader, node_type_ids=node_type_ids, device=device,
                num_negatives=args.num_ranking_negatives,
                eval_rng=np.random.RandomState(seed=eval_rank_seed),
                return_per_query=True, eval_candidates=eval_candidates,
            )
            per_query = list(per_query)
            per_query_by_model[model_name].append(per_query)

            for g in group_order:
                if not idx_by_group[g]:
                    continue
                agg = mean_metric_dicts([per_query[i] for i in idx_by_group[g]])
                per_seed_rows.append({"group": g, "model": model_name, "seed": seed,
                                      "n_queries": counts[g],
                                      **{k: agg.get(k, float("nan")) for k in METRIC_KEYS}})

    # ---- aggregate mean +/- std over seeds --------------------------------------------
    results: dict[str, dict[str, dict[str, tuple[float, float]]]] = {
        g: {m: {} for m in args.models} for g in group_order
    }
    df_seed = pd.DataFrame(per_seed_rows)
    for g in group_order:
        for m in args.models:
            sub = df_seed[(df_seed["group"] == g) & (df_seed["model"] == m)]
            for k in METRIC_KEYS:
                if sub.empty:
                    results[g][m][k] = (float("nan"), float("nan"))
                else:
                    results[g][m][k] = (float(sub[k].mean()), float(sub[k].std(ddof=0)))

    print_group_table(results, group_order, args.models, counts)

    # ---- significance: HT vs each baseline, inside each group --------------------------
    sig_rows: list[dict] = []
    reference = args.models[0]
    baselines = [m for m in args.models[1:]]
    rng = np.random.RandomState(args.eval_seed)

    for metric in TEST_METRICS:
        ref_pq = seed_averaged_per_query(per_query_by_model[reference], metric)
        for base in baselines:
            base_pq = seed_averaged_per_query(per_query_by_model[base], metric)
            for g in group_order:
                idxs = idx_by_group[g]
                if not idxs:
                    continue
                stats = paired_test(ref_pq[idxs], base_pq[idxs], args.n_bootstrap, rng)
                sig_rows.append({"group": g, "reference": reference, "baseline": base,
                                 "metric": metric, **stats})

                # per-seed p-values, for readers who prefer seed-level evidence
                for s_i, seed in enumerate(args.run_seeds):
                    a = np.array([per_query_by_model[reference][s_i][i].get(metric, np.nan) for i in idxs])
                    b = np.array([per_query_by_model[base][s_i][i].get(metric, np.nan) for i in idxs])
                    st = paired_test(a, b, 0, rng)
                    sig_rows.append({"group": g, "reference": reference, "baseline": base,
                                     "metric": f"{metric}@seed{seed}", **st})

    headline = [r for r in sig_rows if "@seed" not in r["metric"]]
    print_significance(headline)

    out_csv = Path(args.out_csv)
    write_csvs(results, group_order, args.models, counts, per_seed_rows, sig_rows, out_csv)

    if args.no_plots:
        print("\nSkipping figure (--no_plots).")
    else:
        plot_groups(results, group_order, args.models, counts, out_csv.parent)


if __name__ == "__main__":
    main()
