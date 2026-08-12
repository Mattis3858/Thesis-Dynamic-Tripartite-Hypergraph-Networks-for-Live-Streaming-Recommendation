"""
Temporal dynamics of predictions: does the model assign TIME-VARYING probabilities to the same
(u, v, w) triples, so that the top-ranked combo changes as the query time t advances?

Two outputs:
  1. Case study (a figure): for one user and a few (streamer, room) combos, sweep the query time t
     over the timeline and plot P(u,v,w,t) per combo. Markers show each combo's real interaction
     times. Shows the ranking cross-overs the advisor asked for.
  2. Aggregate dynamism stat: over a sample of users (each with >=2 observed combos), the fraction
     whose model-argmax combo CHANGES across the time grid -> "for X% of users the top prediction
     shifts over time." This is the quantitative backbone so the figure is illustration, not cherry-pick.

Mechanism (state honestly in the thesis): the variation comes mainly from the evolving neighbour
context (which combos are recently active, sampled with time < t), consistent with the no_time
finding that recency -- not fine-grained time deltas -- drives predictions. It is NOT leakage: at
query time t only interactions strictly before t are visible.

Usage:
    python eval_temporal_dynamics.py --gpu 0                 # auto-pick an illustrative user + aggregate
    python eval_temporal_dynamics.py --gpu 0 --user_id 123   # a specific user for the figure
"""

from __future__ import annotations

import argparse
import csv as _csv
import logging
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from eval_table10 import _checkpoint_exists, _resolve_checkpoint_dir, build_ht_transformer, load_ht_checkpoint
from utils.DataLoader import get_tripartite_link_prediction_data
from utils.utils import convert_to_gpu, get_neighbor_sampler, get_tripartite_neighbor_sampler

ROOT = Path(__file__).resolve().parent


@torch.no_grad()
def score(model, u_arr, v_arr, w_arr, t_arr, device, batch_size=2048) -> np.ndarray:
    """P(u,v,w,t) for aligned arrays; batched. Returns float array [N]."""
    out = []
    n = len(u_arr)
    for s in range(0, n, batch_size):
        e = min(s + batch_size, n)
        y, _, _, _ = model(
            np.asarray(u_arr[s:e], dtype=np.int64), np.asarray(v_arr[s:e], dtype=np.int64),
            np.asarray(w_arr[s:e], dtype=np.int64), np.asarray(t_arr[s:e], dtype=np.float64),
            edge_ids=None, edges_are_positive=False,
        )
        out.append(y.view(-1).float().cpu().numpy())
    return np.concatenate(out) if out else np.array([])


def user_combos(full_data) -> dict[int, dict[tuple[int, int], list[float]]]:
    """user -> {(v,w): sorted list of interaction times}."""
    d: dict[int, dict[tuple[int, int], list[float]]] = defaultdict(lambda: defaultdict(list))
    for u, v, w, t in zip(full_data.user_node_ids, full_data.streamer_node_ids,
                          full_data.item_node_ids, full_data.node_interact_times):
        d[int(u)][(int(v), int(w))].append(float(t))
    for u in d:
        for c in d[u]:
            d[u][c].sort()
    return d


def pick_user(uc: dict, n_combos: int) -> int:
    """Auto-pick: among users with >= n_combos distinct combos, the one whose interactions span the
    widest time range (most likely to show a temporal shift)."""
    best_u, best_span = None, -1.0
    for u, combos in uc.items():
        if len(combos) < n_combos:
            continue
        all_t = [t for times in combos.values() for t in times]
        span = max(all_t) - min(all_t)
        if span > best_span:
            best_u, best_span = u, span
    if best_u is None:  # fallback: user with the most combos
        best_u = max(uc, key=lambda u: len(uc[u]))
    return best_u


def choose_combos(combos: dict[tuple[int, int], list[float]], n: int) -> list[tuple[int, int]]:
    """Pick n combos with the most-separated median interaction times (to expose temporal shifts)."""
    by_median = sorted(combos, key=lambda c: float(np.median(combos[c])))
    if len(by_median) <= n:
        return by_median
    idx = np.linspace(0, len(by_median) - 1, n).round().astype(int)
    return [by_median[i] for i in idx]


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset_name", type=str, default="kuailive_tripartite")
    p.add_argument("--data_dir", type=str, default=None)
    p.add_argument("--val_ratio", type=float, default=0.15)
    p.add_argument("--test_ratio", type=float, default=0.15)
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
    p.add_argument("--run_seed", type=int, default=0)
    p.add_argument("--full_save_model_name", type=str, default="HTTransformer_neg5_popneg_seed0")
    p.add_argument("--user_id", type=int, default=None, help="User for the case-study figure (auto if unset).")
    p.add_argument("--n_combos", type=int, default=3, help="Number of (streamer,room) combos to plot.")
    p.add_argument("--n_time_points", type=int, default=60, help="Time-grid resolution for the figure.")
    p.add_argument("--agg_users", type=int, default=300, help="Users sampled for the aggregate dynamism stat.")
    p.add_argument("--agg_time_points", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_prefix", type=str, default=str(ROOT / "experiment_results" / "temporal_dynamics"))
    return p.parse_args()


def main() -> None:
    warnings.filterwarnings("ignore")
    args = get_args()
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu"
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("eval_temporal_dynamics")

    (
        node_raw_features, edge_raw_features, node_type_ids, full_data, train_data,
        _val, test_data, _nv, _nt, temporal_full_data, temporal_train_data,
    ) = get_tripartite_link_prediction_data(
        dataset_name=args.dataset_name, val_ratio=args.val_ratio, test_ratio=args.test_ratio, data_dir=args.data_dir,
    )
    num_nodes = len(node_type_ids)
    if temporal_full_data is not None:
        full_ns = get_neighbor_sampler(data=temporal_full_data, sample_neighbor_strategy=args.sample_neighbor_strategy,
                                       time_scaling_factor=args.time_scaling_factor, seed=1, num_nodes=num_nodes)
    else:
        full_ns = get_tripartite_neighbor_sampler(data=full_data, sample_neighbor_strategy=args.sample_neighbor_strategy,
                                                  time_scaling_factor=args.time_scaling_factor, seed=1, num_nodes=num_nodes)

    stem = args.full_save_model_name
    folder, stem = _resolve_checkpoint_dir(
        model_dir=str(ROOT / "saved_models" / "HTTransformer" / args.dataset_name / stem),
        dataset_name=args.dataset_name, save_model_name=stem)
    if not _checkpoint_exists(folder, stem):
        raise FileNotFoundError(f"Checkpoint not found: {folder / (stem + '.pkl')}")
    model = build_ht_transformer(
        node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, node_type_ids=node_type_ids,
        neighbor_sampler=full_ns, device=device, use_bias_gate=True, use_type_init=True, use_hetero_coocc=True,
        time_feat_dim=args.time_feat_dim, channel_embedding_dim=args.channel_embedding_dim,
        cooccurrence_dim=args.cooccurrence_dim, patch_size=args.patch_size, num_layers=args.num_layers,
        num_heads=args.num_heads, dropout=args.dropout, max_input_sequence_length=args.max_input_sequence_length)
    load_ht_checkpoint(model, folder, stem, logger)
    model = convert_to_gpu(model, device=device)
    model.eval()
    model.set_neighbor_sampler(full_ns)

    uc = user_combos(full_data)
    t_max_global = float(np.max(full_data.node_interact_times))

    # ---------- 1) case study ----------
    u0 = args.user_id if args.user_id is not None else pick_user(uc, args.n_combos)
    if u0 not in uc:
        raise ValueError(f"user {u0} has no interactions.")
    combos = choose_combos(uc[u0], args.n_combos)
    t_start = min(t for c in combos for t in uc[u0][c])
    grid = np.linspace(t_start, t_max_global, args.n_time_points)

    flat_u, flat_v, flat_w, flat_t = [], [], [], []
    for (v, w) in combos:
        for t in grid:
            flat_u.append(u0); flat_v.append(v); flat_w.append(w); flat_t.append(t)
    probs = score(model, flat_u, flat_v, flat_w, flat_t, device).reshape(len(combos), len(grid))

    csv_path = f"{args.out_prefix}_user{u0}.csv"
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w_ = _csv.writer(f)
        w_.writerow(["combo_streamer", "combo_room", "t", "prob"])
        for ci, (v, w) in enumerate(combos):
            for ti, t in enumerate(grid):
                w_.writerow([v, w, t, f"{probs[ci, ti]:.6f}"])
    print(f"Wrote trajectory CSV -> {csv_path}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Grayscale-safe: the curves are told apart by LINE STYLE, MARKER and gray level rather
        # than by hue, because the point of this figure is which curve overtakes which -- a
        # distinction that vanishes in black-and-white print if only colour separates them.
        styles = [
            {"ls": "-", "marker": "o", "gray": "0.0"},
            {"ls": "--", "marker": "s", "gray": "0.45"},
            {"ls": ":", "marker": "^", "gray": "0.0"},
            {"ls": "-.", "marker": "D", "gray": "0.45"},
            {"ls": (0, (3, 1, 1, 1)), "marker": "v", "gray": "0.7"},
        ]
        plt.figure(figsize=(8, 4.5))
        n_mark = 12  # markers thin out so they annotate the curve instead of hiding it
        for ci, (v, w) in enumerate(combos):
            st = styles[ci % len(styles)]
            plt.plot(grid, probs[ci], color=st["gray"], linestyle=st["ls"], linewidth=1.8,
                     marker=st["marker"], markersize=5, markevery=max(1, len(grid) // n_mark),
                     markerfacecolor="white", markeredgecolor=st["gray"],
                     label=f"(s={v}, r={w})")
            for t in uc[u0][(v, w)]:  # real interaction times of this combo
                plt.axvline(t, color=st["gray"], alpha=0.35, linestyle=st["ls"], linewidth=0.8)
        plt.xlabel("query time t"); plt.ylabel("predicted probability")
        plt.title(f"Time-varying predictions for user {u0} (vertical lines = real interactions)")
        plt.grid(True, alpha=0.25, linewidth=0.6)
        plt.legend(fontsize=8); plt.tight_layout()
        fig_path = f"{args.out_prefix}_user{u0}.pdf"
        plt.savefig(fig_path, bbox_inches="tight"); plt.close()
        print(f"Wrote trajectory figure -> {fig_path}")
    except ImportError:
        print("matplotlib not available -> skipped the figure (trajectory CSV still written).")

    # ---------- 2) aggregate dynamism ----------
    rng = np.random.RandomState(args.seed)
    eligible = [u for u, cs in uc.items() if len(cs) >= 2]
    sample = eligible if len(eligible) <= args.agg_users else list(rng.choice(eligible, args.agg_users, replace=False))
    changed = 0
    counted = 0
    fu, fv, fw, ft, meta = [], [], [], [], []  # meta: (user_idx, combo_idx, t_idx)
    user_combo_list = {}
    for ui, u in enumerate(sample):
        cs = list(uc[u].keys())[:5]
        user_combo_list[ui] = cs
        t0 = min(t for c in cs for t in uc[u][c])
        g = np.linspace(t0, t_max_global, args.agg_time_points)
        for ci, (v, w) in enumerate(cs):
            for ti, t in enumerate(g):
                fu.append(u); fv.append(v); fw.append(w); ft.append(t); meta.append((ui, ci, ti))
    if fu:
        p = score(model, fu, fv, fw, ft, device)
        # reassemble: per user, [n_combos, n_t]; check if argmax-combo changes across t
        by_user: dict[int, dict] = defaultdict(lambda: defaultdict(dict))
        for (ui, ci, ti), val in zip(meta, p):
            by_user[ui][ti][ci] = val
        for ui, tmap in by_user.items():
            n_c = len(user_combo_list[ui])
            if n_c < 2:
                continue
            argmaxes = []
            for ti, cmap in sorted(tmap.items()):
                argmaxes.append(max(cmap, key=cmap.get))
            counted += 1
            if len(set(argmaxes)) > 1:
                changed += 1
    frac = changed / counted if counted else float("nan")
    print("\n### Temporal dynamism (aggregate) ###")
    print(f"users evaluated (>=2 combos): {counted}")
    print(f"users whose top-1 combo CHANGES over the time grid: {changed} ({100*frac:.1f}%)")
    agg_path = f"{args.out_prefix}_aggregate.csv"
    with open(agg_path, "w", newline="", encoding="utf-8") as f:
        w_ = _csv.writer(f)
        w_.writerow(["users_evaluated", "users_top1_changes", "fraction_changes", "agg_time_points"])
        w_.writerow([counted, changed, f"{frac:.6f}", args.agg_time_points])
    print(f"Wrote aggregate stat -> {agg_path}")
    print(f"\nCase-study user = {u0}; combos = {combos}")


if __name__ == "__main__":
    main()
