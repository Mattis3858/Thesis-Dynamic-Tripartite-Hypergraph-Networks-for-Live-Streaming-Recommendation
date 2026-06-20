"""
Batch launcher for train_tripartite_link_prediction.py.

Modes:
  overall   — HTTransformer, HyperHawkes, LightGCN, HAN + baselines (DyGFormer, TGAT, GraphMixer, CAWN, TCL, TGN).
  ablation  — HTTransformer only; full grid of --no-use_* flags (2^3 runs).
  hyperparam — HTTransformer only; patch_size × channel_embedding_dim grid (paper spec).

After each successful training run, appends one row to:
  <results_dir>/results_overall.csv | results_ablation.csv | results_hyperparam.csv

Training writes --metrics_out JSON (mean test/val metrics over num_runs). Extra CLI tokens
are forwarded, e.g.:

  python run_experiments.py --mode overall --dataset_name kuailive_tripartite --num_runs 1
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TRAIN = ROOT / "train_tripartite_link_prediction.py"

# Must match utils.metrics.TRIPARTITE_RANKING_KS
# NOTE: 1 and 3 were added (P@1 / N@3 for the thesis tables). Changing this set changes the
# CSV columns -- start fresh results_*.csv files rather than appending to old ones.
TRIPARTITE_RANKING_KS = (1, 3, 5, 10, 20, 50, 100)

OVERALL_MODELS = [
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
]

# Paper hyperparameter search grid
HYPERPARAM_PATCH_SIZES = [1, 5, 10, 20]
HYPERPARAM_HIDDENS = [32, 64, 128, 256]

_CSV_META_COLUMNS = [
    "timestamp",
    "mode",
    "dataset_name",
    "num_runs",
    "model_name",
    "patch_size",
    "hidden_dim",
    "cooccurrence_dim",
    "use_bias_gate",
    "use_type_init",
    "use_hetero_coocc",
    # training-protocol provenance (filled from the metrics JSON by the generic loop below)
    "train_neg_sampling",
    "train_neg_ratio",
    "loss_type",
    "fusion_mode",
]


def _ranking_metric_columns(prefix: str) -> list[str]:
    """All ranking keys written by train_tripartite_link_prediction --metrics_out (test_* / val_*)."""
    cols = [f"{prefix}roc_auc", f"{prefix}average_precision"]
    for k in TRIPARTITE_RANKING_KS:
        cols.append(f"{prefix}precision@{k}")
        cols.append(f"{prefix}recall@{k}")
        cols.append(f"{prefix}ndcg@{k}")
    return cols


CSV_COLUMNS = _CSV_META_COLUMNS + _ranking_metric_columns("test_") + _ranking_metric_columns("val_")


def _fmt_cell(v) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        return repr(v)
    return str(v)


def _row_from_summary(mode: str, summary: dict) -> dict[str, str]:
    """Flatten metrics JSON into one CSV row (string cells)."""
    row = {col: "" for col in CSV_COLUMNS}
    row["timestamp"] = datetime.now(timezone.utc).isoformat()
    row["mode"] = mode
    row["dataset_name"] = _fmt_cell(summary.get("dataset_name"))
    row["num_runs"] = _fmt_cell(summary.get("num_runs"))
    row["model_name"] = _fmt_cell(summary.get("model_name"))
    row["patch_size"] = _fmt_cell(summary.get("patch_size"))
    row["hidden_dim"] = _fmt_cell(summary.get("channel_embedding_dim"))
    row["cooccurrence_dim"] = _fmt_cell(summary.get("cooccurrence_dim"))
    row["use_bias_gate"] = _fmt_cell(summary.get("use_bias_gate"))
    row["use_type_init"] = _fmt_cell(summary.get("use_type_init"))
    row["use_hetero_coocc"] = _fmt_cell(summary.get("use_hetero_coocc"))
    for col in CSV_COLUMNS:
        if row[col] != "":
            continue
        if col in summary:
            row[col] = _fmt_cell(summary[col])
    for k, v in summary.items():
        if k in CSV_COLUMNS and row[k] == "":
            row[k] = _fmt_cell(v)
    return row


def _append_csv(csv_path: Path, row: dict[str, str]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        if new_file:
            w.writeheader()
        w.writerow(row)


def _run_train(
    *,
    mode: str,
    results_dir: Path,
    train_args: list[str],
    dry_run: bool,
) -> None:
    metrics_path = results_dir / f"_metrics_{int(time.time() * 1000)}.json"
    cmd = [sys.executable, str(TRAIN), "--metrics_out", str(metrics_path)] + train_args
    print(" ".join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, check=True, cwd=str(ROOT))
    if not metrics_path.exists():
        print(f"[warn] missing metrics file: {metrics_path}", flush=True)
        return
    with open(metrics_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
    csv_name = f"results_{mode}.csv"
    _append_csv(results_dir / csv_name, _row_from_summary(mode, summary))
    try:
        metrics_path.unlink()
    except OSError:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["overall", "ablation", "hyperparam"],
    )
    parser.add_argument("--dry_run", action="store_true", help="Print commands without running.")
    parser.add_argument(
        "--results_dir",
        type=str,
        default=str(ROOT / "experiment_results"),
        help="Directory for results_*.csv and temporary metrics JSON.",
    )
    args, forwarded = parser.parse_known_args()
    results_dir = Path(args.results_dir).resolve()

    if args.mode == "overall":
        for m in OVERALL_MODELS:
            _run_train(
                mode="overall",
                results_dir=results_dir,
                train_args=["--model_name", m] + forwarded,
                dry_run=args.dry_run,
            )

    elif args.mode == "ablation":
        for bias, ti, co in itertools.product([True, False], repeat=3):
            bits: list[str] = []
            if not bias:
                bits.append("--no-use_bias_gate")
            if not ti:
                bits.append("--no-use_type_init")
            if not co:
                bits.append("--no-use_hetero_coocc")
            _run_train(
                mode="ablation",
                results_dir=results_dir,
                train_args=["--model_name", "HTTransformer"] + bits + forwarded,
                dry_run=args.dry_run,
            )

    elif args.mode == "hyperparam":
        for ps, hd in itertools.product(HYPERPARAM_PATCH_SIZES, HYPERPARAM_HIDDENS):
            _run_train(
                mode="hyperparam",
                results_dir=results_dir,
                train_args=[
                    "--model_name",
                    "HTTransformer",
                    "--patch_size",
                    str(ps),
                    "--channel_embedding_dim",
                    str(hd),
                ]
                + forwarded,
                dry_run=args.dry_run,
            )


if __name__ == "__main__":
    main()
