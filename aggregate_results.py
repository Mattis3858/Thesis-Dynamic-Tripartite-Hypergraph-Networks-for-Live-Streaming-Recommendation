"""
Aggregate per-seed test metrics into a mean±std main-comparison table.

Reads the per-seed result files written by train_tripartite_link_prediction.py:
    saved_results/<model_name>/<dataset>/<save_model_name>_seed<k>.json
(each holds ONE seed's "test metrics"), groups them by configuration (filename with the
``_seed<k>`` stripped), and reports mean ± sample-std (ddof=1) across seeds.

Why this exists: results_overall.csv stores only the *mean* over seeds (no std), and the per-seed
JSONs are scattered one-file-per-seed. The thesis main table needs mean ± std, so we recompute it
here from the raw per-seed numbers.

Outputs:
  * a markdown table to stdout (headline metrics), and
  * experiment_results/aggregated_mean_std.csv (every metric, mean & std columns).

The HTTransformer full-config group (no ablation suffix) is used as the reference; for each other
group the delta on a primary metric is shown with a rough "beyond 1 std" flag. That flag is a
sanity indicator, NOT a significance test -- for the paper, run a paired test across the shared
seeds if a gap is borderline.

Usage:
    python aggregate_results.py
    python aggregate_results.py --dataset_name kuailive_tripartite --primary_metric ndcg@10
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent

# Columns shown in the markdown table (must be keys in the JSON "test metrics").
HEADLINE_METRICS = [
    "precision@1",
    "ndcg@3",
    "ndcg@5",
    "ndcg@10",
    "recall@10",
    "roc_auc",
    "average_precision",
]

_SEED_RE = re.compile(r"_seed(\d+)$")
# Ablation/variant tokens -> a group carrying any of these is NOT the full config.
_ABLATION_TOKENS = ("no_bias_gate", "no_type_init", "no_hetero_coocc", "mean", "bpr")


def _to_float(v) -> float:
    """Parse a metric cell (string like '0.9908'/'nan' or a number) to float; '' -> nan."""
    if v is None or v == "":
        return float("nan")
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _config_key(stem: str) -> str | None:
    """Strip the trailing ``_seed<k>`` to get the configuration group key. None if no seed tag."""
    m = _SEED_RE.search(stem)
    if not m:
        return None
    return stem[: m.start()]


def collect(dataset_name: str) -> dict[str, dict]:
    """Return {config_key: {"model": str, "seeds": {seed: {metric: float}}}}."""
    groups: dict[str, dict] = {}
    pattern = f"saved_results/*/{dataset_name}/*_seed*.json"
    for path in sorted(ROOT.glob(pattern)):
        model_name = path.parent.parent.name
        stem = path.stem
        key = _config_key(stem)
        if key is None:
            continue
        seed_m = _SEED_RE.search(stem)
        seed = int(seed_m.group(1))
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            print(f"[warn] skipping unreadable {path}: {e}")
            continue
        metrics_raw = data.get("test metrics", {})
        metrics = {k: _to_float(v) for k, v in metrics_raw.items()}
        g = groups.setdefault(key, {"model": model_name, "seeds": {}})
        g["seeds"][seed] = metrics
    return groups


def _agg(values: list[float]) -> tuple[float, float, int]:
    """mean, sample-std (ddof=1; 0 if n<2), n -- ignoring NaNs."""
    xs = [v for v in values if not math.isnan(v)]
    if not xs:
        return float("nan"), float("nan"), 0
    mean = float(np.mean(xs))
    std = float(np.std(xs, ddof=1)) if len(xs) > 1 else 0.0
    return mean, std, len(xs)


def _all_metric_names(groups: dict[str, dict]) -> list[str]:
    names: set[str] = set()
    for g in groups.values():
        for m in g["seeds"].values():
            names.update(m.keys())
    # stable order: headline first, then the rest sorted
    rest = sorted(n for n in names if n not in HEADLINE_METRICS)
    return [n for n in HEADLINE_METRICS if n in names] + rest


def _aggregate_group(g: dict, metric_names: list[str]) -> dict[str, tuple[float, float, int]]:
    out: dict[str, tuple[float, float, int]] = {}
    for name in metric_names:
        vals = [seed_metrics.get(name, float("nan")) for seed_metrics in g["seeds"].values()]
        out[name] = _agg(vals)
    return out


def _find_reference_key(groups: dict[str, dict]) -> str | None:
    """The HTTransformer full-config group (model dir HTTransformer, no ablation token)."""
    candidates = [
        k for k, g in groups.items()
        if g["model"] == "HTTransformer" and not any(tok in k for tok in _ABLATION_TOKENS)
    ]
    if not candidates:
        return None
    # prefer the one with the most seeds, then the shortest key (fewest extra tokens)
    return sorted(candidates, key=lambda k: (-len(groups[k]["seeds"]), len(k)))[0]


def _fmt(stat: tuple[float, float, int]) -> str:
    mean, std, _n = stat
    if math.isnan(mean):
        return "—"
    return f"{mean:.4f}±{std:.4f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset_name", type=str, default="kuailive_tripartite")
    ap.add_argument("--primary_metric", type=str, default="ndcg@10",
                    help="Metric used for the HT-vs-baseline delta / beyond-1-std flag.")
    ap.add_argument("--results_dir", type=str, default=str(ROOT / "experiment_results"))
    args = ap.parse_args()

    # Windows consoles default to a non-UTF-8 codepage (e.g. cp950); force UTF-8 so the table's
    # ± and other glyphs never raise UnicodeEncodeError (redirect to a file to view cleanly).
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

    groups = collect(args.dataset_name)
    if not groups:
        print(f"No per-seed result JSONs found under saved_results/*/{args.dataset_name}/*_seed*.json")
        return

    metric_names = _all_metric_names(groups)
    agg = {key: _aggregate_group(g, metric_names) for key, g in groups.items()}
    ref_key = _find_reference_key(groups)

    # ---- markdown table (headline metrics) ----
    headline = [m for m in HEADLINE_METRICS if m in metric_names]
    header = ["config (model)", "n"] + headline
    print("\n### Test ranking — mean ± std over seeds\n")
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join(["---"] * len(header)) + "|")

    def _row_sort(item):
        key, g = item
        # reference first, then HTTransformer groups, then baselines alphabetically
        is_ref = 0 if key == ref_key else 1
        is_ht = 0 if g["model"] == "HTTransformer" else 1
        return (is_ref, is_ht, g["model"], key)

    for key, g in sorted(groups.items(), key=_row_sort):
        n = max((len([1 for s in g["seeds"].values() if not math.isnan(s.get(m, float("nan")))])
                 for m in headline), default=0)
        label = key + (" (ref)" if key == ref_key else "")
        cells = [_fmt(agg[key][m]) for m in headline]
        print(f"| {label} | {n} | " + " | ".join(cells) + " |")

    # ---- delta vs HT reference on the primary metric ----
    pm = args.primary_metric
    if ref_key is not None and pm in metric_names:
        ref_mean, ref_std, _ = agg[ref_key][pm]
        print(f"\n### Delta vs HT reference (ref = {ref_key}) on {pm}  "
              f"[HT = {ref_mean:.4f}+/-{ref_std:.4f}]\n")
        print("| config (model) | mean | Delta(HT-this) | beyond 1 std? |")
        print("|---|---|---|---|")
        for key, g in sorted(groups.items(), key=_row_sort):
            if key == ref_key:
                continue
            mean, std, _ = agg[key][pm]
            if math.isnan(mean) or math.isnan(ref_mean):
                continue
            delta = ref_mean - mean
            beyond = abs(delta) > max(ref_std, std, 1e-12)
            flag = "yes" if beyond else "no (within noise)"
            print(f"| {key} | {mean:.4f}±{std:.4f} | {delta:+.4f} | {flag} |")
        print("\n('beyond 1 std' is a rough sanity flag, not a significance test. For borderline "
              "gaps run a paired test across the shared seeds.)")

    # ---- full CSV dump (every metric: mean & std) ----
    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "aggregated_mean_std.csv"
    cols = ["config", "model", "n_seeds"]
    for m in metric_names:
        cols += [f"{m}_mean", f"{m}_std"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for key, g in sorted(groups.items(), key=_row_sort):
            n_seeds = len(g["seeds"])
            row = [key, g["model"], n_seeds]
            for m in metric_names:
                mean, std, _ = agg[key][m]
                row += [f"{mean:.6f}" if not math.isnan(mean) else "nan",
                        f"{std:.6f}" if not math.isnan(std) else "nan"]
            w.writerow(row)
    print(f"\nFull mean/std table -> {csv_path}")


if __name__ == "__main__":
    main()
