"""
Streamer-level behavioral bias analysis (committee follow-up #1).

Mirrors ``analyze_user_bias.py`` but on the STREAMER side, and runs on a large raw slice
(1k-10k streamers) instead of the small 10-streamer training subset, so the claim
"streamers also exhibit concentration bias" is backed by a population-scale sample.

Per streamer we compute
  * Audience HHI  -- concentration of interactions over USERS       (loyal-fanbase vs. broad-reach)
  * Item HHI      -- concentration of interactions over ROOMS/ITEMS (single-venue vs. multi-venue)
plus entropies, a repeat-viewer ratio, and volume statistics.

Small-sample caveat: HHI is upward-biased for low-volume streamers (a streamer with 3
interactions has HHI ~ 1 by construction). We therefore (1) drop streamers below
``--min_interactions``, (2) additionally report the size-corrected HHI
(H - 1/n) / (1 - 1/n), and (3) plot HHI against interaction volume (plus a Spearman
correlation) so the reader can verify the concentration is not a sample-size artifact.

Input is the pairwise edge table (``ml_kuailive_edges.csv``: src,dst,ts,label,etype,idx).
Two incompatible etype conventions exist in this project (see remap_kuailive_etypes.py);
select with ``--etype_convention``:
  source    (utils.preprocess_kuailive output):  0=user->room, 1=user->streamer, 2=streamer->room
  converter (preprocess_kuailive_tripartite in): 0=user->streamer, 1=user->room, 2=room->streamer

Usage:
  python analyze_streamer_bias.py --edges path/to/ml_kuailive_edges.csv \
      --etype_convention source --min_interactions 30
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DEFAULT_OUT_DIR = ROOT / "experiment_results"

GROUP_AUDIENCE = "Audience-concentrated"
GROUP_ITEM = "Item-concentrated"
GROUP_BROAD = "Broad"

# etype ids per convention; streamer_is_src refers to the streamer<->room edge
_CONVENTIONS = {
    "source": {"user_streamer": 1, "streamer_room": 2, "streamer_is_src": True},
    "converter": {"user_streamer": 0, "streamer_room": 2, "streamer_is_src": False},
}


def _hhi(counts: Counter) -> float:
    total = sum(counts.values())
    if total <= 0:
        return float("nan")
    p = np.array([c / total for c in counts.values()], dtype=np.float64)
    return float(np.sum(p**2))


def _normalized_hhi(counts: Counter) -> float:
    """(H - 1/n) / (1 - 1/n): removes the mechanical floor imposed by a small support size."""
    n = len(counts)
    if n <= 1:
        return float("nan")
    return float((_hhi(counts) - 1.0 / n) / (1.0 - 1.0 / n))


def _entropy(counts: Counter) -> float:
    total = sum(counts.values())
    if total <= 0:
        return float("nan")
    p = np.array([c / total for c in counts.values()], dtype=np.float64)
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def load_edges(path: Path, dedupe_labels: bool) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={c: c.strip().lower() for c in df.columns})
    for col in ("src", "dst", "ts", "etype"):
        if col not in df.columns:
            raise ValueError(f"{path} is missing column '{col}'; got {list(df.columns)}")
    df["src"] = df["src"].astype(np.int64)
    df["dst"] = df["dst"].astype(np.int64)
    df["etype"] = df["etype"].astype(np.int64)
    df["ts"] = df["ts"].astype(np.float64)

    if dedupe_labels:
        before = len(df)
        df = df.drop_duplicates(subset=["etype", "src", "dst", "ts"], keep="first")
        if before != len(df):
            print(f"Collapsed {before - len(df)} multi-label duplicate rows "
                  f"(same etype,src,dst,ts) -> {len(df)} rows.")
    return df.reset_index(drop=True)


def compute_streamer_stats(df: pd.DataFrame, convention: str) -> pd.DataFrame:
    """Aggregate per-streamer audience / item concentration from the pairwise edge table."""
    conv = _CONVENTIONS[convention]

    us = df[df["etype"] == conv["user_streamer"]]
    user_counts: dict[int, Counter] = defaultdict(Counter)
    for u, s in zip(us["src"].to_numpy(), us["dst"].to_numpy()):
        user_counts[int(s)][int(u)] += 1

    sr = df[df["etype"] == conv["streamer_room"]]
    s_col, r_col = ("src", "dst") if conv["streamer_is_src"] else ("dst", "src")
    item_counts: dict[int, Counter] = defaultdict(Counter)
    for s, r in zip(sr[s_col].to_numpy(), sr[r_col].to_numpy()):
        item_counts[int(s)][int(r)] += 1

    streamers = sorted(set(user_counts) | set(item_counts))
    if not streamers:
        raise RuntimeError(
            f"No streamer edges found under --etype_convention {convention}. "
            "The file is probably in the other convention; see remap_kuailive_etypes.py."
        )

    rows = []
    for s in streamers:
        uc, ic = user_counts.get(s, Counter()), item_counts.get(s, Counter())
        n_int = sum(uc.values())
        repeat = sum(c for c in uc.values() if c >= 2)
        rows.append(
            {
                "streamer_id": s,
                "interaction_count": n_int,
                "num_users": len(uc),
                "num_items": len(ic),
                "audience_hhi": _hhi(uc),
                "audience_hhi_norm": _normalized_hhi(uc),
                "audience_entropy": _entropy(uc),
                "item_hhi": _hhi(ic),
                "item_hhi_norm": _normalized_hhi(ic),
                "item_entropy": _entropy(ic),
                "repeat_viewer_ratio": (repeat / n_int) if n_int > 0 else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def compute_streamer_stats_from_dataset(train_data, mapping) -> pd.DataFrame:
    """
    Same per-streamer statistics, but read from an already-built tripartite dataset.

    Used to group the handful of streamers a trained model actually saw, so the group-wise
    evaluation needs no access to the original edge table. Roles come from utils.roles rather
    than from the column names, and only the TRAIN split is used (no leakage into the test
    period, matching analyze_user_bias.py).
    """
    from utils.roles import role_ids

    u_ids = train_data.user_node_ids
    s_ids = role_ids(train_data, mapping, "streamer")
    r_ids = role_ids(train_data, mapping, "room")

    user_counts: dict[int, Counter] = defaultdict(Counter)
    item_counts: dict[int, Counter] = defaultdict(Counter)
    for u, s, r in zip(u_ids, s_ids, r_ids):
        user_counts[int(s)][int(u)] += 1
        item_counts[int(s)][int(r)] += 1

    rows = []
    for s in sorted(user_counts):
        uc, ic = user_counts[s], item_counts[s]
        n_int = sum(uc.values())
        repeat = sum(c for c in uc.values() if c >= 2)
        rows.append(
            {
                "streamer_id": s,
                "interaction_count": n_int,
                "num_users": len(uc),
                "num_items": len(ic),
                "audience_hhi": _hhi(uc),
                "audience_hhi_norm": _normalized_hhi(uc),
                "audience_entropy": _entropy(uc),
                "item_hhi": _hhi(ic),
                "item_hhi_norm": _normalized_hhi(ic),
                "item_entropy": _entropy(ic),
                "repeat_viewer_ratio": (repeat / n_int) if n_int > 0 else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def select_k(features: np.ndarray, k_min: int, k_max: int, random_state: int) -> tuple[int, list[tuple[int, float]]]:
    """Pick k by silhouette score instead of hard-coding k=3."""
    k_max = min(k_max, len(features) - 1)
    scores: list[tuple[int, float]] = []
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(features)
        scores.append((k, float(silhouette_score(features, labels))))
    best_k = max(scores, key=lambda kv: kv[1])[0]
    return best_k, scores


def label_clusters(centroids: np.ndarray) -> dict[int, str]:
    """Name clusters by which concentration axis their centroid maximizes (audience = col 0)."""
    audience_c = int(np.argmax(centroids[:, 0]))
    item_order = np.argsort(centroids[:, 1])[::-1]
    item_c = int(item_order[0] if item_order[0] != audience_c else item_order[1])

    mapping = {audience_c: GROUP_AUDIENCE, item_c: GROUP_ITEM}
    rest = [c for c in range(centroids.shape[0]) if c not in mapping]
    for i, c in enumerate(rest):
        mapping[c] = GROUP_BROAD if len(rest) == 1 else f"{GROUP_BROAD}-{i + 1}"
    return mapping


def plot_all(df: pd.DataFrame, out_dir: Path, feat_cols: list[str]) -> None:
    # Imported lazily so the analysis/CSV path still runs on machines without a plotting stack.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid")

    # Fig A: concentration distributions (raw + size-corrected)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, col, title, color in (
        (axes[0][0], "audience_hhi", "Audience Concentration (HHI)", "#e74c3c"),
        (axes[0][1], "item_hhi", "Item Concentration (HHI)", "#3498db"),
        (axes[1][0], "audience_hhi_norm", "Audience HHI (size-corrected)", "#c0392b"),
        (axes[1][1], "item_hhi_norm", "Item HHI (size-corrected)", "#2980b9"),
    ):
        sns.histplot(data=df, x=col, kde=True, ax=ax, color=color, bins=30)
        ax.set_title(title, fontsize=13)
        ax.set_xlabel(col)
        ax.set_ylabel("Number of Streamers")
    plt.tight_layout()
    p = out_dir / "fig_streamer_bias_distributions.pdf"
    plt.savefig(p, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"Saved distribution plot  -> {p}")

    # Fig B: HHI vs. volume -- shows concentration is not a small-sample artifact
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, col, title in (
        (axes[0], "audience_hhi", "Audience HHI vs. interaction volume"),
        (axes[1], "item_hhi", "Item HHI vs. interaction volume"),
    ):
        ax.scatter(df["interaction_count"], df[col], s=14, alpha=0.5, edgecolor="none")
        ax.set_xscale("log")
        ax.set_xlabel("Interactions per streamer (log)")
        ax.set_ylabel(col)
        ax.set_title(title, fontsize=13)
    plt.tight_layout()
    p = out_dir / "fig_streamer_hhi_vs_volume.pdf"
    plt.savefig(p, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"Saved HHI-vs-volume plot -> {p}")

    # Fig C: clustering scatter (mirrors the user-side Figure 3)
    plt.figure(figsize=(7, 6))
    sns.scatterplot(
        data=df, x=feat_cols[0], y=feat_cols[1], hue="bias_group",
        palette="Set1", s=28, alpha=0.75, edgecolor="none",
    )
    plt.title("Streamer Clustering based on Interaction Bias", fontsize=14, fontweight="bold")
    plt.xlabel(f"Audience Concentration ({feat_cols[0]})", fontsize=12)
    plt.ylabel(f"Item Concentration ({feat_cols[1]})", fontsize=12)
    plt.legend(title="Streamer Bias Group", fontsize=10, title_fontsize=11)
    p = out_dir / "fig_streamer_clustering_scatter.pdf"
    plt.savefig(p, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"Saved clustering plot    -> {p}")


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Streamer-level HHI bias analysis and clustering.")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--edges", type=str,
                     help="Pairwise edge CSV (src,dst,ts,label,etype,idx), e.g. a large ml_kuailive_edges.csv slice.")
    src.add_argument("--from_dataset", type=str,
                     help="Instead of an edge CSV, profile the streamers of an already-built "
                          "tripartite dataset (e.g. kuailive_tripartite), using its TRAIN split. "
                          "Streamer ids are then global node ids, ready for eval_streamer_group.py.")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="--from_dataset only: dataset folder (default processed_data/<name>).")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="--from_dataset only.")
    parser.add_argument("--test_ratio", type=float, default=0.15, help="--from_dataset only.")
    parser.add_argument("--etype_convention", type=str, default="source", choices=sorted(_CONVENTIONS),
                        help="--edges only: which etype convention the file uses; see remap_kuailive_etypes.py.")
    parser.add_argument("--min_interactions", type=int, default=30,
                        help="Drop streamers below this interaction count (HHI is unreliable for tiny supports).")
    parser.add_argument("--sample_streamers", type=int, default=None,
                        help="Optionally subsample this many streamers (after filtering) for the analysis.")
    parser.add_argument("--split_mode", type=str, default="kmeans", choices=("kmeans", "median"),
                        help="How to form the groups. 'kmeans' for a population-scale slice; "
                             "'median' splits at the median of one concentration axis, which is the "
                             "honest choice for a handful of streamers, where k-means latches onto a "
                             "single outlier and produces a group of size one.")
    parser.add_argument("--median_axis", type=str, default="auto", choices=("auto", "audience", "item"),
                        help="--split_mode median: which axis to split on ('auto' picks the axis with "
                             "the larger relative spread).")
    parser.add_argument("--cluster_features", type=str, default="raw", choices=("raw", "norm"),
                        help="Which concentration features to cluster on: 'raw' HHI, or 'norm' "
                             "size-corrected HHI. Use 'norm' when raw HHI correlates strongly with "
                             "interaction volume (see the Spearman check), so the clusters reflect "
                             "behaviour rather than activity level.")
    parser.add_argument("--k_min", type=int, default=2)
    parser.add_argument("--k_max", type=int, default=5)
    parser.add_argument("--n_clusters", type=int, default=None,
                        help="Force k instead of selecting it by silhouette.")
    parser.add_argument("--random_state", type=int, default=2020)
    parser.add_argument("--keep_label_duplicates", action="store_true",
                        help="Keep multi-label rows (click/comment/like) as separate interactions.")
    parser.add_argument("--no_plots", action="store_true",
                        help="Skip figures (CSV + terminal output only); useful without matplotlib/seaborn.")
    parser.add_argument("--out_dir", type=str, default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--out_csv", type=str, default=None,
                        help="Default: <out_dir>/streamer_bias_groups.csv")
    return parser.parse_args()


def main() -> None:
    args = get_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = Path(args.out_csv) if args.out_csv else out_dir / "streamer_bias_groups.csv"

    if args.from_dataset:
        from utils.DataLoader import get_tripartite_link_prediction_data
        from utils.roles import detect_roles

        print(f"Loading tripartite dataset '{args.from_dataset}' (train split used for the stats) ...")
        loaded = get_tripartite_link_prediction_data(
            dataset_name=args.from_dataset, val_ratio=args.val_ratio,
            test_ratio=args.test_ratio, data_dir=args.data_dir,
        )
        full_data, train_data = loaded[3], loaded[4]
        mapping = detect_roles(full_data)
        stats = compute_streamer_stats_from_dataset(train_data, mapping)
        id_space = "global_node_id"
        print(f"Found {len(stats)} streamers in the dataset's train split.")
    else:
        print(f"Loading edges from {args.edges} (convention={args.etype_convention}) ...")
        df_edges = load_edges(Path(args.edges), not args.keep_label_duplicates)
        stats = compute_streamer_stats(df_edges, args.etype_convention)
        id_space = "raw_edge_id"
        print(f"Found {len(stats)} streamers in the slice.")

    kept = stats[stats["interaction_count"] >= args.min_interactions].copy()
    print(f"Kept {len(kept)} streamers with >= {args.min_interactions} interactions "
          f"(dropped {len(stats) - len(kept)} low-volume streamers).")
    if args.split_mode == "median":
        min_needed = 2
    else:
        min_needed = args.k_max + 1 if args.n_clusters is None else args.n_clusters + 1
    if len(kept) < min_needed:
        raise RuntimeError(
            f"Only {len(kept)} streamers survive --min_interactions {args.min_interactions}; "
            f"at least {min_needed} are needed for the requested k. Lower the threshold, lower "
            "--k_max, or use a larger slice."
        )
    if len(kept) < 30:
        detail = ("Silhouette-based k selection is unstable at this size"
                  if args.split_mode == "kmeans"
                  else "the median split is descriptive only at this size")
        print(f"!! WARNING: grouping only {len(kept)} streamers; {detail} -- treat the grouping as "
              "descriptive, and rely on the population-scale slice for the claim that streamer "
              "bias exists.")

    if args.sample_streamers is not None and args.sample_streamers < len(kept):
        kept = kept.sample(n=args.sample_streamers, random_state=args.random_state)
        print(f"Subsampled {len(kept)} streamers for the analysis.")

    feat_cols = (["audience_hhi", "item_hhi"] if args.cluster_features == "raw"
                 else ["audience_hhi_norm", "item_hhi_norm"])
    # The size-corrected HHI is undefined for a support of one (a streamer with a single
    # distinct viewer or item), so those rows cannot be clustered on it.
    n_before = len(kept)
    kept = kept.dropna(subset=feat_cols)
    if len(kept) < n_before:
        print(f"Dropped {n_before - len(kept)} streamers whose {args.cluster_features} features are "
              f"undefined (single distinct viewer or item).")

    kept = kept.sort_values("streamer_id").reset_index(drop=True)
    print(f"Clustering on {feat_cols}.")
    features_raw = kept[feat_cols].to_numpy(dtype=np.float64)

    # A degenerate axis silently reduces the 2D clustering to 1D, so say so loudly.
    for j, name in enumerate(feat_cols):
        if np.nanstd(features_raw[:, j]) < 1e-12:
            print(
                f"\n!! WARNING: '{name}' is constant across all streamers "
                f"(value {features_raw[0, j]:.4f}). Clustering therefore uses the other axis only.\n"
                f"   Common causes: a wrong --etype_convention, or a 1:1 streamer-item mapping in this slice."
            )

    features = StandardScaler().fit_transform(features_raw)

    if args.split_mode == "median":
        if args.median_axis == "auto":
            spread = np.nanstd(features_raw, axis=0) / np.maximum(np.nanmean(features_raw, axis=0), 1e-12)
            axis = int(np.argmax(spread))
            print(f"\nMedian split on '{feat_cols[axis]}' (larger relative spread: "
                  f"{spread[axis]:.3f} vs {spread[1 - axis]:.3f}).")
        else:
            axis = 0 if args.median_axis == "audience" else 1
            print(f"\nMedian split on '{feat_cols[axis]}'.")

        thr = float(np.median(features_raw[:, axis]))
        high = features_raw[:, axis] > thr
        high_label = GROUP_AUDIENCE if axis == 0 else GROUP_ITEM
        kept["cluster_id"] = high.astype(int)
        kept["bias_group"] = np.where(high, high_label, GROUP_BROAD)
        centroids_raw = np.array([
            features_raw[~high].mean(axis=0) if (~high).any() else [np.nan, np.nan],
            features_raw[high].mean(axis=0) if high.any() else [np.nan, np.nan],
        ])
        cluster_to_label = {0: GROUP_BROAD, 1: high_label}
        print(f"  threshold = {thr:.4f}  ->  {int(high.sum())} above / {int((~high).sum())} below")

        print("\n=== Group centroids (unstandardized) ===")
        for c in (0, 1):
            print(f"  {cluster_to_label[c]}: "
                  f"{feat_cols[0]}={centroids_raw[c][0]:.4f}, {feat_cols[1]}={centroids_raw[c][1]:.4f}")
        _finish(kept, args, out_csv, out_dir, feat_cols, id_space)
        return

    if args.n_clusters is not None:
        k = args.n_clusters
        print(f"\nUsing forced k={k}.")
    else:
        k, scores = select_k(features, args.k_min, args.k_max, args.random_state)
        print("\n=== Silhouette scores by k ===")
        for kk, sc in scores:
            print(f"  k={kk}: {sc:.4f}" + ("   <-- selected" if kk == k else ""))

    km = KMeans(n_clusters=k, random_state=args.random_state, n_init=10)
    kept["cluster_id"] = km.fit_predict(features)
    centroids_raw = np.array(
        [features_raw[kept["cluster_id"].to_numpy() == c].mean(axis=0) for c in range(k)]
    )
    cluster_to_label = label_clusters(centroids_raw)
    kept["bias_group"] = kept["cluster_id"].map(cluster_to_label)

    print("\n=== Cluster centroids (unstandardized) ===")
    for c in range(k):
        print(f"  cluster {c} ({cluster_to_label[c]}): "
              f"{feat_cols[0]}={centroids_raw[c][0]:.4f}, {feat_cols[1]}={centroids_raw[c][1]:.4f}")

    _finish(kept, args, out_csv, out_dir, feat_cols, id_space)


def _finish(kept: pd.DataFrame, args, out_csv: Path, out_dir: Path,
            feat_cols: list[str], id_space: str) -> None:
    """Shared reporting tail for both split modes: CSV, group summary, sanity check, figures."""
    # Tells eval_streamer_group.py whether these ids are already dataset node ids.
    kept = kept.copy()
    kept["id_space"] = id_space

    sizes = kept["bias_group"].value_counts()
    if (sizes < 2).any():
        singletons = ", ".join(f"{g} (n={n})" for g, n in sizes.items() if n < 2)
        print(f"\n!! WARNING: group(s) of size one: {singletons}. Such a group's metrics describe a "
              "single streamer, not a group -- prefer --split_mode median at this sample size.")

    kept.to_csv(out_csv, index=False)
    print(f"\nWrote {len(kept)} streamers -> {out_csv}")

    print("\n=== Summary by streamer bias group ===")
    summary = (
        kept.groupby("bias_group")
        .agg(
            n_streamers=("streamer_id", "count"),
            mean_interactions=("interaction_count", "mean"),
            median_interactions=("interaction_count", "median"),
            mean_num_users=("num_users", "mean"),
            mean_num_items=("num_items", "mean"),
            mean_audience_hhi=("audience_hhi", "mean"),
            mean_audience_hhi_norm=("audience_hhi_norm", "mean"),
            mean_item_hhi=("item_hhi", "mean"),
            mean_item_hhi_norm=("item_hhi_norm", "mean"),
            mean_repeat_ratio=("repeat_viewer_ratio", "mean"),
        )
        .round(4)
    )
    print(summary.to_string())
    summary_path = out_dir / "streamer_bias_group_summary.csv"
    summary.to_csv(summary_path)
    print(f"\nWrote group summary -> {summary_path}")

    # Small-sample sanity check: is the concentration merely a volume artifact?
    log_vol = np.log10(kept["interaction_count"].to_numpy(dtype=np.float64))
    print("\n=== Small-sample sanity check (Spearman vs. log10 volume) ===")
    for col in ("audience_hhi", "item_hhi"):
        vals = kept[col].to_numpy(dtype=np.float64)
        if np.nanstd(vals) < 1e-12:
            print(f"  {col:12s}: undefined (constant column)")
            continue
        rho, pval = spearmanr(vals, log_vol)
        print(f"  {col:12s}: rho = {rho:+.4f} (p = {pval:.3g})")
    print("  A strongly negative rho would mean the concentration is a low-volume artifact.")

    if args.no_plots:
        print("\nSkipping figures (--no_plots).")
    else:
        print("\nGenerating figures ...")
        plot_all(kept, out_dir, feat_cols)


if __name__ == "__main__":
    main()
