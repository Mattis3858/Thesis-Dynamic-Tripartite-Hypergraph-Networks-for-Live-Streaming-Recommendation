"""
Draw a small, readable schematic of the tripartite (User–Streamer–Room) graph using REAL data
points sampled from the processed dataset — for a "this is what the graph looks like" paper figure.

The full graph is a hairball (thousands of nodes); this samples a few users and the streamers/rooms
they interact with, lays the three node types at the corners of a triangle, and draws each
(user, streamer, room) interaction as a light triangle (a tripartite hyperedge). Colors are a
validated colorblind-safe categorical palette (blue / aqua / yellow); nodes carry dark outlines and
a legend so identity never rests on fill color alone.

NOTE on the streamer/room label swap: on the swap-affected data (see remap), node type 2 is really
the room and type 3 the streamer. Pass --swap_streamer_room to relabel the legend/clusters correctly.

Usage:
    python visualize_tripartite_graph.py --dataset_name kuailive_tripartite --n_users 5 --max_hyperedges 22
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent

# validated categorical palette (dataviz skill): slot1 blue, slot2 aqua, slot3 yellow
COL_USER = "#2a78d6"
COL_STREAMER = "#1baf7a"
COL_ROOM = "#eda100"
EDGE_INK = "#33322f"     # dark node outline (gives low-contrast fills relief vs surface)
LINE_INK = "#9a998f"     # recessive hyperedge lines
SURFACE = "#fcfcfb"


def load_hyperedges(dataset_name: str, data_dir: str | None):
    base = Path(data_dir) if data_dir else ROOT / "processed_data" / dataset_name
    df = pd.read_csv(base / f"ml_{dataset_name}.csv")
    df = df.rename(columns={c: c.strip().lower() for c in df.columns})
    node_types = np.load(base / f"ml_{dataset_name}_node_types.npy")
    return df, node_types


def sample_subgraph(df: pd.DataFrame, n_users: int, max_hyperedges: int, seed: int):
    """Pick a few moderate-degree users, keep their (u,streamer,room) hyperedges up to a cap."""
    rng = np.random.RandomState(seed)
    deg = df.groupby("u").size()
    # moderate-degree users make a legible (connected but not dense) figure
    cand = deg[(deg >= 2) & (deg <= 10)].index.to_numpy()
    if len(cand) == 0:
        cand = deg.index.to_numpy()
    users = rng.choice(cand, size=min(n_users, len(cand)), replace=False)
    sub = df[df["u"].isin(users)].drop_duplicates(subset=["u", "streamer", "room"])
    if len(sub) > max_hyperedges:
        sub = sub.sample(n=max_hyperedges, random_state=seed)
    return sub.reset_index(drop=True)


def _cluster_positions(ids, center, base_radius):
    """Arrange a type's nodes on a circle around its corner (radius grows with count)."""
    ids = sorted(ids)
    n = len(ids)
    r = base_radius * (0.6 + 0.12 * math.sqrt(max(n, 1)))
    pos = {}
    for i, nid in enumerate(ids):
        if n == 1:
            pos[nid] = center
        else:
            a = 2 * math.pi * i / n - math.pi / 2
            pos[nid] = (center[0] + r * math.cos(a), center[1] + r * math.sin(a))
    return pos


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset_name", type=str, default="kuailive_tripartite")
    p.add_argument("--data_dir", type=str, default=None)
    p.add_argument("--n_users", type=int, default=5)
    p.add_argument("--max_hyperedges", type=int, default=22)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--swap_streamer_room", action="store_true",
                   help="Relabel type2<->type3 (use on the swap-affected data so labels are correct).")
    p.add_argument("--out", type=str, default=str(ROOT / "experiment_results" / "tripartite_graph_schematic.pdf"))
    args = p.parse_args()

    df, _node_types = load_hyperedges(args.dataset_name, args.data_dir)
    sub = sample_subgraph(df, args.n_users, args.max_hyperedges, args.seed)

    users = set(int(x) for x in sub["u"])
    streamers = set(int(x) for x in sub["streamer"])
    rooms = set(int(x) for x in sub["room"])

    # triangle corners: users bottom-left, streamers top, rooms bottom-right
    pos = {}
    pos.update(_cluster_positions(users, (-3.2, -1.9), 1.6))
    pos.update(_cluster_positions(streamers, (0.0, 3.4), 1.6))
    pos.update(_cluster_positions(rooms, (3.2, -1.9), 1.6))

    deg = defaultdict(int)
    for _, r in sub.iterrows():
        for c in ("u", "streamer", "room"):
            deg[int(r[c])] += 1

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError:
        print("matplotlib not installed — cannot draw. `pip install matplotlib`.")
        return

    lab_s = "Room" if args.swap_streamer_room else "Streamer"
    lab_r = "Streamer" if args.swap_streamer_room else "Room"

    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    ax.set_facecolor(SURFACE); fig.patch.set_facecolor(SURFACE)

    # hyperedges as light triangles (u–streamer, streamer–room, u–room)
    for _, r in sub.iterrows():
        u, s, w = int(r["u"]), int(r["streamer"]), int(r["room"])
        tri = [pos[u], pos[s], pos[w], pos[u]]
        xs, ys = zip(*tri)
        ax.plot(xs, ys, color=LINE_INK, alpha=0.28, linewidth=1.0, zorder=1)

    def draw_nodes(ids, color, label):
        xs = [pos[i][0] for i in ids]; ys = [pos[i][1] for i in ids]
        sizes = [90 + 45 * deg[i] for i in ids]
        ax.scatter(xs, ys, s=sizes, c=color, edgecolors=EDGE_INK, linewidths=1.3, zorder=3, label=label)

    draw_nodes(users, COL_USER, "User")
    draw_nodes(streamers, COL_STREAMER, lab_s)
    draw_nodes(rooms, COL_ROOM, lab_r)

    # cluster captions near corners
    ax.text(-3.2, -3.9, "Users", ha="center", fontsize=12, fontweight="bold", color=COL_USER)
    ax.text(0.0, 5.0, lab_s + "s", ha="center", fontsize=12, fontweight="bold", color=COL_STREAMER)
    ax.text(3.2, -3.9, lab_r + "s", ha="center", fontsize=12, fontweight="bold", color=COL_ROOM)

    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COL_USER, markeredgecolor=EDGE_INK, markersize=11, label="User"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COL_STREAMER, markeredgecolor=EDGE_INK, markersize=11, label=lab_s),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COL_ROOM, markeredgecolor=EDGE_INK, markersize=11, label=lab_r),
        Line2D([0], [0], color=LINE_INK, alpha=0.5, linewidth=1.2, label="Interaction (hyperedge)"),
    ]
    ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=9)
    ax.set_title(
        f"Tripartite interaction graph (sampled: {len(users)} users, "
        f"{len(streamers)} {lab_s.lower()}s, {len(rooms)} {lab_r.lower()}s, {len(sub)} hyperedges)",
        fontsize=11,
    )
    ax.set_xlim(-6, 6); ax.set_ylim(-4.5, 5.5)
    ax.set_aspect("equal"); ax.axis("off")
    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Wrote {out} and {out.with_suffix('.png')}")
    print(f"Sampled {len(users)} users, {len(streamers)} {lab_s.lower()}s, {len(rooms)} {lab_r.lower()}s, {len(sub)} hyperedges.")


if __name__ == "__main__":
    main()
