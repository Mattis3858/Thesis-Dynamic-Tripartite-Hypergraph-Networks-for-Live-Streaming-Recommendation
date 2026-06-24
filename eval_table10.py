"""
Evaluate HT-Transformer (Full vs w/o bias gate) on the test set, stratified by user bias group (Table 10).

Reads experiment_results/user_bias_groups.csv, runs 1 positive + 99 negative ranking per test query,
and prints a Markdown table with strict top-K metrics (P@1, N@3, P@5, N@10).
"""

from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from models.HTTransformer import HTTransformer
from train_tripartite_link_prediction import evaluate_tripartite_ranking
from utils.DataLoader import get_idx_data_loader, get_tripartite_link_prediction_data
from utils.EarlyStopping import EarlyStopping
from utils.metrics import mean_metric_dicts
from utils.utils import convert_to_gpu, get_neighbor_sampler, get_tripartite_neighbor_sampler

ROOT = Path(__file__).resolve().parent
DEFAULT_BIAS_CSV = ROOT / "experiment_results" / "user_bias_groups.csv"
DEFAULT_FULL_DIR = (
    ROOT / "saved_models" / "HTTransformer" / "kuailive_tripartite" / "HTTransformer_full_seed0"
)
DEFAULT_WO_GATE_DIR = (
    ROOT
    / "saved_models"
    / "HTTransformer"
    / "kuailive_tripartite"
    / "HTTransformer_no_use_bias_gate_seed0"
)

BIAS_STREAMER = "Streamer-biased"
BIAS_ITEM = "Item-biased"
BIAS_MIXED = "Mixed"
BIAS_UNKNOWN = "New/Unknown"

GROUP_ORDER = [BIAS_STREAMER, BIAS_ITEM, BIAS_MIXED, BIAS_UNKNOWN]

MODEL_FULL = "full"
MODEL_WO_GATE = "wo_gate"
MODEL_DISPLAY = {MODEL_FULL: "Full", MODEL_WO_GATE: "w/o gate"}

# [MODIFIED] Using ultra-strict top-K metrics to break the ceiling effect
METRIC_KEYS = ("precision@1", "ndcg@3", "precision@5", "ndcg@5", "ndcg@10")
METRIC_HEADERS = ("P@1", "N@3", "P@5", "N@5", "N@10")


def load_user_group_map(csv_path: Path) -> dict[int, str]:
    df = pd.read_csv(csv_path)
    if "user_id" not in df.columns or "bias_group" not in df.columns:
        raise ValueError(f"{csv_path} must contain columns user_id and bias_group.")
    return {int(row.user_id): str(row.bias_group) for row in df.itertuples(index=False)}


def _resolve_checkpoint_dir(
    *,
    model_dir: str | None,
    dataset_name: str,
    save_model_name: str,
) -> tuple[Path, str]:
    if model_dir:
        folder = Path(model_dir).expanduser().resolve()
        if folder.suffix == ".pkl":
            return folder.parent, folder.stem
        return folder, save_model_name
    folder = ROOT / "saved_models" / "HTTransformer" / dataset_name / save_model_name
    return folder.resolve(), save_model_name


def _checkpoint_exists(folder: Path, save_model_name: str) -> bool:
    return (folder / f"{save_model_name}.pkl").is_file()


def _find_checkpoint(
    *,
    model_dir: str | None,
    dataset_name: str,
    save_model_name: str,
    fallback_dirs: list[Path],
) -> tuple[Path, str]:
    folder, stem = _resolve_checkpoint_dir(
        model_dir=model_dir,
        dataset_name=dataset_name,
        save_model_name=save_model_name,
    )
    if _checkpoint_exists(folder, stem):
        return folder, stem
    tried = [folder / f"{stem}.pkl"]
    for alt in fallback_dirs:
        if _checkpoint_exists(alt, save_model_name):
            return alt, save_model_name
        tried.append(alt / f"{save_model_name}.pkl")
    raise FileNotFoundError(
        "Checkpoint not found. Tried:\n  "
        + "\n  ".join(str(p) for p in tried)
    )


def build_ht_transformer(
    *,
    node_raw_features: np.ndarray,
    edge_raw_features: np.ndarray,
    node_type_ids: np.ndarray,
    neighbor_sampler,
    device: str,
    use_bias_gate: bool,
    use_type_init: bool,
    use_hetero_coocc: bool,
    time_feat_dim: int,
    channel_embedding_dim: int,
    cooccurrence_dim: int,
    patch_size: int,
    num_layers: int,
    num_heads: int,
    dropout: float,
    max_input_sequence_length: int,
) -> HTTransformer:
    return HTTransformer(
        node_raw_features=node_raw_features,
        edge_raw_features=edge_raw_features,
        node_type_ids=node_type_ids,
        neighbor_sampler=neighbor_sampler,
        time_feat_dim=time_feat_dim,
        hidden_dim=channel_embedding_dim,
        cooccurrence_dim=cooccurrence_dim,
        d_out=None,
        patch_size=patch_size,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
        max_input_sequence_length=max_input_sequence_length,
        device=device,
        use_bias_gate=use_bias_gate,
        use_type_init=use_type_init,
        use_hetero_coocc=use_hetero_coocc,
    )


def load_ht_checkpoint(
    model: HTTransformer,
    checkpoint_folder: Path,
    save_model_name: str,
    logger: logging.Logger,
) -> None:
    folder_str = str(checkpoint_folder)
    ckpt_path = checkpoint_folder / f"{save_model_name}.pkl"
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    early_stopping = EarlyStopping(
        patience=0,
        save_model_folder=folder_str,
        save_model_name=save_model_name,
        logger=logger,
        model_name="HTTransformer",
    )
    early_stopping.load_checkpoint(model, map_location="cpu")


def aggregate_metrics_by_bias_group(
    test_data,
    per_query_metrics: list[dict],
    user_group_map: dict[int, str],
) -> dict[str, dict[str, float]]:
    buckets: dict[str, list[dict]] = {g: [] for g in GROUP_ORDER}
    known_groups = set(GROUP_ORDER) - {BIAS_UNKNOWN}

    for i, m in enumerate(per_query_metrics):
        u = int(test_data.user_node_ids[i])
        group = user_group_map.get(u, BIAS_UNKNOWN)
        if group not in known_groups:
            group = BIAS_UNKNOWN
        buckets[group].append(m)

    out: dict[str, dict[str, float]] = {}
    for g in GROUP_ORDER:
        lst = buckets[g]
        out[g] = mean_metric_dicts(lst) if lst else {k: float("nan") for k in METRIC_KEYS}
    return out


def _fmt_metric(x: float) -> str:
    if isinstance(x, float) and np.isnan(x):
        return "—"
    return f"{x:.4f}"


def print_table10_markdown(
    results: dict[str, dict[str, dict[str, float]]],
    *,
    counts: dict[str, dict[str, int]] | None = None,
) -> None:
    rows: list[tuple[str, str, str, str, str, str, str]] = []
    for group in GROUP_ORDER:
        for model_key in (MODEL_FULL, MODEL_WO_GATE):
            m = results[model_key].get(group, {})
            rows.append(
                (
                    group,
                    MODEL_DISPLAY[model_key],
                    _fmt_metric(m.get("precision@1", float("nan"))),
                    _fmt_metric(m.get("ndcg@3", float("nan"))),
                    _fmt_metric(m.get("precision@5", float("nan"))),
                    _fmt_metric(m.get("ndcg@5", float("nan"))),
                    _fmt_metric(m.get("ndcg@10", float("nan"))),
                )
            )

    headers = ("Group", "Model", *METRIC_HEADERS)
    col_widths = [len(h) for h in headers]
    for row in rows:
        for j, cell in enumerate(row):
            col_widths[j] = max(col_widths[j], len(cell))

    def _pad(cells: tuple[str, ...]) -> str:
        return "| " + " | ".join(c.ljust(col_widths[i]) for i, c in enumerate(cells)) + " |"

    sep = "|" + "|".join("-" * (w + 2) for w in col_widths) + "|"

    print("\n### Table 10 — Strict Group-wise ranking (test, 1+99 negatives)\n")
    print(_pad(headers))
    print(sep)
    for row in rows:
        print(_pad(row))

    if counts:
        print("\n**Queries per (group, model)**:")
        for group in GROUP_ORDER:
            n = counts.get(MODEL_FULL, {}).get(group, 0)
            print(f"  - {group}: {n}")


def get_eval_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stratified test ranking for thesis Table 10.")
    parser.add_argument("--bias_groups_csv", type=str, default=str(DEFAULT_BIAS_CSV))
    parser.add_argument("--dataset_name", type=str, default="kuailive_tripartite")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    
    # [RESTORED] Back to batch_size 200 for fast inference
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
    
    # [RESTORED] Back to 99 negatives
    parser.add_argument("--num_ranking_negatives", type=int, default=99)
    parser.add_argument("--eval_seed", type=int, default=0)
    parser.add_argument(
        "--eval_candidates_path",
        type=str,
        default=None,
        help="Path to the shared fixed candidate .npz. Default: auto-resolve to "
        "eval_candidates/{dataset}_test_joint_vw_seed{eval_seed+1}.npz (same file the main "
        "comparison uses), so Table 10 is comparable to the main table.",
    )
    parser.add_argument(
        "--no_fixed_eval_candidates",
        action="store_true",
        help="Opt out of the fixed candidate set and use legacy on-the-fly UNIFORM negatives. "
        "NOT recommended: the result is not comparable to the main table.",
    )
    parser.add_argument("--run_seed", type=int, default=0)
    parser.add_argument("--full_model_dir", type=str, default=str(DEFAULT_FULL_DIR))
    parser.add_argument("--full_save_model_name", type=str, default=None)
    parser.add_argument("--wo_gate_model_dir", type=str, default=str(DEFAULT_WO_GATE_DIR))
    parser.add_argument("--wo_gate_save_model_name", type=str, default=None)
    parser.add_argument(
        "--out_csv",
        type=str,
        default=str(ROOT / "experiment_results" / "table10_group_metrics.csv"),
        help="Where to write the group-wise metrics CSV (one row per group x model).",
    )

    g_ti = parser.add_mutually_exclusive_group()
    g_ti.add_argument("--use_type_init", dest="use_type_init", action="store_true")
    g_ti.add_argument("--no-use_type_init", dest="use_type_init", action="store_false")
    parser.set_defaults(use_type_init=True)
    g_co = parser.add_mutually_exclusive_group()
    g_co.add_argument("--use_hetero_coocc", dest="use_hetero_coocc", action="store_true")
    g_co.add_argument("--no-use_hetero_coocc", dest="use_hetero_coocc", action="store_false")
    parser.set_defaults(use_hetero_coocc=True)

    return parser.parse_args()


def _ordered_metric_names(results: dict) -> list[str]:
    """Headline metrics first, then any remaining ones, for a stable CSV column order."""
    present: list[str] = []
    for mk in results:
        for g in results[mk]:
            for k in results[mk][g]:
                if k not in present:
                    present.append(k)
    headline = [
        "roc_auc", "average_precision", "precision@1", "ndcg@3",
        "precision@5", "ndcg@5", "ndcg@10", "recall@10",
    ]
    return [m for m in headline if m in present] + [m for m in sorted(present) if m not in headline]


def write_table10_csv(results: dict, query_counts: dict, path: str) -> None:
    """One row per (bias group x model) with all ranking metrics; for the thesis / records."""
    import csv as _csv

    metric_names = _ordered_metric_names(results)

    def _cell(v) -> str:
        if isinstance(v, float):
            return "nan" if np.isnan(v) else f"{v:.6f}"
        return "" if v is None else str(v)

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = _csv.writer(f)
        w.writerow(["group", "model", "n_queries"] + metric_names)
        for g in GROUP_ORDER:
            for mk in (MODEL_FULL, MODEL_WO_GATE):
                gm = results.get(mk, {}).get(g, {})
                n = query_counts.get(mk, {}).get(g, "")
                w.writerow([g, MODEL_DISPLAY.get(mk, mk), n] + [_cell(gm.get(m)) for m in metric_names])
    print(f"Wrote Table 10 metrics -> {path}")


def main() -> None:
    warnings.filterwarnings("ignore")
    args = get_eval_args()
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu"

    bias_csv = Path(args.bias_groups_csv).expanduser().resolve()
    user_group_map = load_user_group_map(bias_csv)

    default_stem = f"HTTransformer_seed{args.run_seed}"
    full_stem = args.full_save_model_name or default_stem
    wo_gate_stem = args.wo_gate_save_model_name or default_stem
    legacy_dir = (ROOT / "saved_models" / "HTTransformer" / args.dataset_name / default_stem).resolve()

    full_folder, full_stem = _find_checkpoint(
        model_dir=args.full_model_dir,
        dataset_name=args.dataset_name,
        save_model_name=full_stem,
        fallback_dirs=[legacy_dir],
    )
    wo_gate_folder, wo_gate_stem = _find_checkpoint(
        model_dir=args.wo_gate_model_dir,
        dataset_name=args.dataset_name,
        save_model_name=wo_gate_stem,
        fallback_dirs=[legacy_dir],
    )

    (
        node_raw_features, edge_raw_features, node_type_ids,
        full_data, train_data, _val_data, test_data,
        _new_node_val_data, _new_node_test_data,
        temporal_full_data, temporal_train_data,
    ) = get_tripartite_link_prediction_data(
        dataset_name=args.dataset_name, val_ratio=args.val_ratio, test_ratio=args.test_ratio, data_dir=args.data_dir,
    )

    num_nodes = len(node_type_ids)
    if temporal_full_data is not None and temporal_train_data is not None:
        train_neighbor_sampler = get_neighbor_sampler(data=temporal_train_data, sample_neighbor_strategy=args.sample_neighbor_strategy, time_scaling_factor=args.time_scaling_factor, seed=0, num_nodes=num_nodes)
        full_neighbor_sampler = get_neighbor_sampler(data=temporal_full_data, sample_neighbor_strategy=args.sample_neighbor_strategy, time_scaling_factor=args.time_scaling_factor, seed=1, num_nodes=num_nodes)
    else:
        train_neighbor_sampler = get_tripartite_neighbor_sampler(data=train_data, sample_neighbor_strategy=args.sample_neighbor_strategy, time_scaling_factor=args.time_scaling_factor, seed=0, num_nodes=num_nodes)
        full_neighbor_sampler = get_tripartite_neighbor_sampler(data=full_data, sample_neighbor_strategy=args.sample_neighbor_strategy, time_scaling_factor=args.time_scaling_factor, seed=1, num_nodes=num_nodes)

    test_loader = get_idx_data_loader(indices_list=list(range(len(test_data.user_node_ids))), batch_size=args.batch_size, shuffle=False)
    eval_rank_seed = args.eval_seed + 1
    logger = logging.getLogger("eval_table10")

    ht_kw = dict(
        node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, node_type_ids=node_type_ids,
        neighbor_sampler=train_neighbor_sampler, device=device, use_type_init=args.use_type_init,
        use_hetero_coocc=args.use_hetero_coocc, time_feat_dim=args.time_feat_dim, channel_embedding_dim=args.channel_embedding_dim,
        cooccurrence_dim=args.cooccurrence_dim, patch_size=args.patch_size, num_layers=args.num_layers,
        num_heads=args.num_heads, dropout=args.dropout, max_input_sequence_length=args.max_input_sequence_length,
    )

    # Fixed shared candidates by DEFAULT (consistent with the main comparison); legacy uniform is
    # opt-in only, else Table 10's group-wise numbers would be incomparable to the main protocol.
    eval_candidates = None
    if args.no_fixed_eval_candidates:
        logger.warning(
            "Using LEGACY on-the-fly UNIFORM negatives (--no_fixed_eval_candidates); not comparable "
            "to the main table."
        )
    else:
        from utils.eval_candidates import load_eval_candidates

        cand_path = args.eval_candidates_path or str(
            ROOT / "eval_candidates" / f"{args.dataset_name}_test_joint_vw_seed{args.eval_seed + 1}.npz"
        )
        if not Path(cand_path).exists():
            raise FileNotFoundError(
                f"Fixed eval candidates not found: {cand_path}\n"
                "Build it by running training once (auto-builds) or `python build_eval_candidates.py`, "
                "pass --eval_candidates_path, or use --no_fixed_eval_candidates for legacy uniform."
            )
        eval_candidates = load_eval_candidates(cand_path)
        logger.info("Table 10 ranking against fixed candidates: %s", cand_path)

    results: dict[str, dict[str, dict[str, float]]] = {}
    query_counts: dict[str, dict[str, int]] = {MODEL_FULL: {}, MODEL_WO_GATE: {}}

    for model_key, use_bias_gate, folder, stem in (
        (MODEL_FULL, True, full_folder, full_stem),
        (MODEL_WO_GATE, False, wo_gate_folder, wo_gate_stem),
    ):
        model = build_ht_transformer(use_bias_gate=use_bias_gate, **ht_kw)
        load_ht_checkpoint(model, folder, stem, logger)
        model = convert_to_gpu(model, device=device)

        test_eval_rng = np.random.RandomState(seed=eval_rank_seed)
        _loss, _agg, per_query = evaluate_tripartite_ranking(
            model=model, neighbor_sampler=full_neighbor_sampler, data=test_data, idx_data_loader=test_loader,
            node_type_ids=node_type_ids, device=device, num_negatives=args.num_ranking_negatives,
            eval_rng=test_eval_rng, return_per_query=True, eval_candidates=eval_candidates,
        )
        
        by_group = aggregate_metrics_by_bias_group(test_data, per_query, user_group_map)
        results[model_key] = by_group

        counts: dict[str, int] = {g: 0 for g in GROUP_ORDER}
        for i in range(test_data.num_interactions):
            g = user_group_map.get(int(test_data.user_node_ids[i]), BIAS_UNKNOWN)
            counts[g if g in set(GROUP_ORDER)-{BIAS_UNKNOWN} else BIAS_UNKNOWN] += 1
        query_counts[model_key] = counts

    print_table10_markdown(results, counts=query_counts)
    write_table10_csv(results, query_counts, args.out_csv)

if __name__ == "__main__":
    main()