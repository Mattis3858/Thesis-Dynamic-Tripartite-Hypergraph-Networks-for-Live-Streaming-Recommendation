"""
User bias grouping for thesis Table 10 (group-wise comparison) and Figure 2/3.

Uses **train split only** to compute per-user Streamer / Room HHI, 
then KMeans (k=3) and centroid-based labels.
Outputs CSV and visualization plots (PDF).
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

from utils.DataLoader import get_tripartite_link_prediction_data
from utils.roles import RoleMapping, detect_roles, role_ids

ROOT = Path(__file__).resolve().parent
DEFAULT_OUT_DIR = ROOT / "experiment_results"
DEFAULT_OUT_CSV = DEFAULT_OUT_DIR / "user_bias_groups.csv"

BIAS_STREAMER = "Streamer-biased"
BIAS_ITEM = "Item-biased"
BIAS_MIXED = "Mixed"

# Grayscale-safe encoding: shape + fill lightness + ellipse line style, so the clusters remain
# distinguishable when the paper is printed in black and white (colour carries no information).
GROUP_STYLE = {
    BIAS_STREAMER: {"marker": "o", "face": "#e0e0e0", "line": "-"},
    BIAS_MIXED: {"marker": "^", "face": "#8c8c8c", "line": "--"},
    BIAS_ITEM: {"marker": "s", "face": "#1a1a1a", "line": ":"},
}


def _herfindahl(counts: Counter) -> float:
    """HHI = sum_s (p_s)^2 over interaction counts."""
    total = sum(counts.values())
    if total <= 0:
        return float("nan")
    proportions = np.array([c / total for c in counts.values()], dtype=np.float64)
    return float(np.sum(proportions**2))


def compute_user_hhi_from_train(train_data, mapping: RoleMapping) -> dict[int, dict]:
    """
    Aggregate train hyperedges (u, s, r) per user; return stats per user_id.

    Roles come from ``mapping`` rather than from the column names: on datasets built without
    remap_kuailive_etypes.py the column called "streamer" actually holds rooms, which would
    otherwise make streamer_hhi measure room concentration (see utils/roles.py).
    """
    streamer_counts: dict[int, Counter] = defaultdict(Counter)
    room_counts: dict[int, Counter] = defaultdict(Counter)
    interaction_count: dict[int, int] = defaultdict(int)

    u_ids = train_data.user_node_ids
    s_ids = role_ids(train_data, mapping, "streamer")
    r_ids = role_ids(train_data, mapping, "room")
    for i in range(train_data.num_interactions):
        u = int(u_ids[i])
        s = int(s_ids[i])
        r = int(r_ids[i])
        interaction_count[u] += 1
        streamer_counts[u][s] += 1
        room_counts[u][r] += 1

    out: dict[int, dict] = {}
    for u in interaction_count:
        out[u] = {
            "interaction_count": interaction_count[u],
            "streamer_hhi": _herfindahl(streamer_counts[u]),
            "room_hhi": _herfindahl(room_counts[u]),
        }
    return out


def _assign_cluster_labels(centroids: np.ndarray) -> dict[int, str]:
    """
    Map cluster_id -> bias_group using centroid comparison (k=3).

    Labels come from each cluster's RELATIVE position on the two axes, not from a per-axis
    argmax. The streamer axis saturates (most users concentrate on a single streamer, so their
    HHI is ~1.0), and on a saturated axis an argmax picks an arbitrary cluster: the group that
    is high on streamer HHI *and* high on room HHI would win the streamer label even though what
    actually distinguishes it is its room concentration. Z-scoring each axis across the
    centroids and ranking by the difference makes the label reflect which axis a cluster leans
    towards relative to the others.
    """
    if centroids.shape != (3, 2):
        raise ValueError(f"Expected 3x2 centroids, got {centroids.shape}")

    std = centroids.std(axis=0)
    std[std < 1e-12] = 1.0
    z = (centroids - centroids.mean(axis=0)) / std
    lean = z[:, 0] - z[:, 1]  # > 0: leans streamer, < 0: leans room/item

    streamer_cluster = int(np.argmax(lean))
    room_cluster = int(np.argmin(lean))
    mixed_cluster = ({0, 1, 2} - {streamer_cluster, room_cluster}).pop()

    return {
        streamer_cluster: BIAS_STREAMER,
        room_cluster: BIAS_ITEM,
        mixed_cluster: BIAS_MIXED,
    }


def run_clustering(
    user_stats: dict[int, dict],
    *,
    n_clusters: int = 3,
    random_state: int = 2020,
) -> tuple[np.ndarray, np.ndarray, dict[int, str], np.ndarray]:
    users = sorted(user_stats.keys())
    features = np.array(
        [[user_stats[u]["streamer_hhi"], user_stats[u]["room_hhi"]] for u in users],
        dtype=np.float64,
    )

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_ids = kmeans.fit_predict(features)
    centroids = kmeans.cluster_centers_
    cluster_to_label = _assign_cluster_labels(centroids)
    return np.asarray(users), cluster_ids, cluster_to_label, centroids


def _cluster_ellipse(ax, xy: np.ndarray, n_std: float = 2.0,
                     min_extent: tuple[float, float] = (0.0, 0.0), **kwargs) -> None:
    """
    Outline a cluster with a 2-sigma covariance ellipse.

    Drawn instead of a convex hull because a cluster whose points share an identical coordinate
    (e.g. every member at streamer HHI = 1.0) is degenerate, and a hull would raise; a covariance
    ellipse with a small regularizer still renders as a thin, informative sliver.
    """
    from matplotlib.patches import Ellipse

    if len(xy) < 3:
        return
    mean = xy.mean(axis=0)
    cov = np.cov(xy, rowvar=False) + np.eye(2) * 1e-9
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = np.maximum(vals[order], 0.0), vecs[:, order]
    angle = float(np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0])))
    width, height = 2.0 * n_std * np.sqrt(vals)
    # A cluster collapsed onto one axis would otherwise draw a zero-width sliver that no printer
    # can render; floor each extent at a fraction of the plotted range so it stays visible.
    ax.add_patch(
        Ellipse(xy=tuple(mean), width=max(width, min_extent[0]), height=max(height, min_extent[1]),
                angle=angle, fill=False, **kwargs)
    )


def plot_and_save_visualizations(df: pd.DataFrame, out_dir: Path):
    """Generate and save academic plots for Figure 2 and Figure 3."""
    sns.set_theme(style="whitegrid")

    # --- Plot 1: Distributions (For Figure 2 in Thesis) ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    sns.histplot(data=df, x="streamer_hhi", kde=True, ax=axes[0], color="#95a5a6", bins=20)
    axes[0].set_title("Streamer Concentration (HHI) Distribution", fontsize=14)
    axes[0].set_xlabel("Streamer HHI")
    axes[0].set_ylabel("Number of Users")

    sns.histplot(data=df, x="room_hhi", kde=True, ax=axes[1], color="#9b59b6", bins=20)
    axes[1].set_title("Item Concentration (HHI) Distribution", fontsize=14)
    axes[1].set_xlabel("Item HHI")
    axes[1].set_ylabel("Number of Users")

    plt.tight_layout()
    dist_path = out_dir / "fig2_user_bias_distributions.pdf"
    plt.savefig(dist_path, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved distribution plot to: {dist_path}")

    # --- Plot 2: Scatter & Clustering (For Figure 3 in Thesis) ---
    # Encoded so the groups stay separable when the paper is printed in black and white:
    # marker SHAPE, fill LIGHTNESS, an enclosing ellipse with its own LINE STYLE, and a label
    # written next to each centroid. Colour alone is not used to carry any information.
    fig, ax = plt.subplots(figsize=(7.2, 6))
    ax.grid(True, alpha=0.3, linewidth=0.6)
    ax.set_axisbelow(True)

    x_all = df["streamer_hhi"].to_numpy(dtype=float)
    y_all = df["room_hhi"].to_numpy(dtype=float)
    x_span = max(float(x_all.max() - x_all.min()), 1e-6)
    y_span = max(float(y_all.max() - y_all.min()), 1e-6)
    x_mid = float(x_all.min()) + x_span / 2
    min_extent = (0.03 * x_span, 0.03 * y_span)

    for group in (BIAS_STREAMER, BIAS_MIXED, BIAS_ITEM):
        sub = df[df["bias_group"] == group]
        if sub.empty:
            continue
        style = GROUP_STYLE[group]
        xy = sub[["streamer_hhi", "room_hhi"]].to_numpy(dtype=float)

        ax.scatter(
            xy[:, 0], xy[:, 1],
            marker=style["marker"], s=52,
            facecolor=style["face"], edgecolor="black", linewidth=0.5, alpha=0.85,
            label=f"{group} (n={len(sub)})", zorder=3,
        )
        _cluster_ellipse(ax, xy, min_extent=min_extent, edgecolor="black",
                         linestyle=style["line"], linewidth=1.4, zorder=4)

        centroid = xy.mean(axis=0)
        ax.scatter(*centroid, marker="X", s=150, facecolor="white", edgecolor="black",
                   linewidth=1.4, zorder=5)
        # Clusters sitting against the right edge (HHI saturated at 1.0) get their label placed to
        # the left, into empty space, instead of on top of the neighbouring cluster's points.
        to_left = centroid[0] > x_mid
        ax.annotate(
            group, xy=centroid,
            xytext=(-12, 10) if to_left else (12, 10), textcoords="offset points",
            ha="right" if to_left else "left", fontsize=10, fontweight="bold", zorder=6,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="black", linewidth=0.6),
        )

    ax.set_title("User Clustering based on Interaction Bias", fontsize=14, fontweight="bold")
    ax.set_xlabel("Streamer Concentration (HHI)", fontsize=12)
    ax.set_ylabel("Item Concentration (HHI)", fontsize=12)
    ax.legend(title="User Bias Group", fontsize=10, title_fontsize=11, loc="upper left",
              framealpha=0.95)

    scatter_path = out_dir / "fig3_user_clustering_scatter.pdf"
    plt.savefig(scatter_path, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved scatter plot to: {scatter_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cluster users by Streamer/Room HHI and generate plots."
    )
    parser.add_argument("--dataset_name", type=str, default="kuailive_tripartite")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Dataset folder; default <repo>/processed_data/{dataset_name}",
    )
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--random_state", type=int, default=2020, help="KMeans random seed.")
    args = parser.parse_args()

    print("Loading tripartite data (train split used for HHI / clustering only) ...")
    (
        _node_raw_features,
        _edge_raw_features,
        _node_type_ids,
        full_data,
        train_data,
        *_rest,
    ) = get_tripartite_link_prediction_data(
        dataset_name=args.dataset_name,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        data_dir=args.data_dir,
    )

    print(
        f"Train hyperedges (deduped): {train_data.num_interactions} "
        f"(users with >=1 train interaction will be clustered)"
    )

    mapping = detect_roles(full_data)
    if mapping.swapped:
        print(
            "NOTE: this dataset's columns are swapped, so streamer_hhi/room_hhi below are computed\n"
            "      from the DETECTED roles, not from the column names. Group labels and figures are\n"
            "      therefore in terms of the real streamers and rooms."
        )

    user_stats = compute_user_hhi_from_train(train_data, mapping)
    if len(user_stats) < 3:
        raise RuntimeError(f"Need at least 3 users for KMeans; found {len(user_stats)}.")

    users, cluster_ids, cluster_to_label, centroids = run_clustering(
        user_stats, random_state=args.random_state
    )

    rows = []
    for u, cid in zip(users, cluster_ids):
        cid = int(cid)
        st = user_stats[int(u)]
        rows.append(
            {
                "user_id": int(u),
                "interaction_count": st["interaction_count"],
                "streamer_hhi": st["streamer_hhi"],
                "room_hhi": st["room_hhi"],
                "cluster_id": cid,
                "bias_group": cluster_to_label[cid],
            }
        )

    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Save CSV
    df = pd.DataFrame(rows)
    df = df.sort_values("user_id").reset_index(drop=True)
    df.to_csv(DEFAULT_OUT_CSV, index=False)
    print(f"\nWrote {len(df)} users -> {DEFAULT_OUT_CSV}")

    # Generate Plots
    print("\nGenerating academic visualizations...")
    plot_and_save_visualizations(df, out_dir)

    print("\n=== Summary stats by bias_group ===")
    summary = df.groupby("bias_group")[["interaction_count", "streamer_hhi", "room_hhi"]].agg(
        ["mean", "median", "count"]
    )
    print(summary.to_string())


if __name__ == "__main__":
    main()