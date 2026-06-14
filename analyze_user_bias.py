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

ROOT = Path(__file__).resolve().parent
DEFAULT_OUT_DIR = ROOT / "experiment_results"
DEFAULT_OUT_CSV = DEFAULT_OUT_DIR / "user_bias_groups.csv"

BIAS_STREAMER = "Streamer-biased"
BIAS_ITEM = "Item-biased"
BIAS_MIXED = "Mixed"


def _herfindahl(counts: Counter) -> float:
    """HHI = sum_s (p_s)^2 over interaction counts."""
    total = sum(counts.values())
    if total <= 0:
        return float("nan")
    proportions = np.array([c / total for c in counts.values()], dtype=np.float64)
    return float(np.sum(proportions**2))


def compute_user_hhi_from_train(train_data) -> dict[int, dict]:
    """
    Aggregate train hyperedges (u, s, r) per user; return stats per user_id.
    """
    streamer_counts: dict[int, Counter] = defaultdict(Counter)
    room_counts: dict[int, Counter] = defaultdict(Counter)
    interaction_count: dict[int, int] = defaultdict(int)

    u_ids = train_data.user_node_ids
    s_ids = train_data.streamer_node_ids
    r_ids = train_data.item_node_ids
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
    """
    if centroids.shape != (3, 2):
        raise ValueError(f"Expected 3x2 centroids, got {centroids.shape}")

    streamer_cluster = int(np.argmax(centroids[:, 0]))
    room_cluster = int(np.argmax(centroids[:, 1]))

    if streamer_cluster == room_cluster:
        order_room = np.argsort(centroids[:, 1])[::-1]
        room_cluster = int(order_room[1] if order_room[0] == streamer_cluster else order_room[0])

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


def plot_and_save_visualizations(df: pd.DataFrame, out_dir: Path):
    """Generate and save academic plots for Figure 2 and Figure 3."""
    sns.set_theme(style="whitegrid")
    colors = {BIAS_STREAMER: "#e74c3c", BIAS_ITEM: "#3498db", BIAS_MIXED: "#2ecc71"}

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
    plt.figure(figsize=(7, 6))
    sns.scatterplot(
        data=df, 
        x="streamer_hhi", 
        y="room_hhi", 
        hue="bias_group", 
        palette=colors,
        s=60, 
        alpha=0.8,
        edgecolor="w"
    )
    plt.title("User Clustering based on Interaction Bias", fontsize=14, fontweight="bold")
    plt.xlabel("Streamer Concentration (HHI)", fontsize=12)
    plt.ylabel("Item Concentration (HHI)", fontsize=12)
    plt.legend(title="User Bias Group", fontsize=10, title_fontsize=11)
    
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
        _full_data,
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

    user_stats = compute_user_hhi_from_train(train_data)
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