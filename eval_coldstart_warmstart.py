"""
Cold-start remedies for new streamers (committee follow-up #3).

``eval_coldstart.py`` showed that masking a role's history at inference collapses ranking
quality when the cold node is the low-cardinality hub. This script asks the follow-up
question: can a cold node be *linked to existing warm nodes* to recover part of that loss?

The cold node keeps its own identity token (so it stays cold); only its historical CONTEXT is
borrowed from a warm proxy node. Four strategies are compared under the identical zero-history
simulation and the same fixed shared candidate set:

  zero            no history at all -- reproduces the current thesis table (control)
  popular         borrow the globally most-active warm node's history (naive control: does ANY
                  history help, regardless of relevance?)
  proxy_<partner> borrow the history of the warm node most frequently co-occurring with the
                  query's partner entity (e.g. for a cold streamer: the streamer that most often
                  serves this ITEM, or that this USER most often watches)

Validity guards
  * Proxy tables are built from the TRAIN split only -- no test-period leakage.
  * The masked node itself is never selectable as its own proxy, otherwise its identity would
    leak back in and the "cold" simulation would be void.
  * A query whose partner has no train history falls back to an empty sequence, i.e. to `zero`,
    never to something better-informed.

Reported over ``--run_seeds`` (5 by default) as mean +/- std, with a paired Wilcoxon signed-rank
test of each strategy against `zero` on seed-averaged per-query metrics.

Usage:
  python eval_coldstart_warmstart.py --dataset_name kuailive_tripartite --cold_role streamer \
      --run_seeds 0 1 2 3 4
"""

from __future__ import annotations

import argparse
import logging
import types
import warnings
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from eval_streamer_group import (
    build_model,
    load_checkpoint,
    paired_test,
    resolve_checkpoint,
    seed_averaged_per_query,
)
from train_tripartite_link_prediction import evaluate_tripartite_ranking
from utils.DataLoader import get_idx_data_loader, get_tripartite_link_prediction_data
from utils.roles import detect_roles
from utils.utils import convert_to_gpu, get_neighbor_sampler, get_tripartite_neighbor_sampler

ROOT = Path(__file__).resolve().parent

ROLE_ATTR = {
    "user": "user_node_ids",
    "streamer": "streamer_node_ids",
    "item": "item_node_ids",
}
PARTNERS = {
    "streamer": ("item", "user"),
    "item": ("streamer", "user"),
    "user": ("streamer", "item"),
}

METRIC_KEYS = ("roc_auc", "average_precision", "precision@1", "ndcg@3", "ndcg@10", "recall@10")
METRIC_HEADERS = ("AUC", "AP", "P@1", "N@3", "N@10", "R@10")
TEST_METRICS = ("precision@1", "ndcg@10")

STRATEGY_ZERO = "zero"
STRATEGY_POPULAR = "popular"


# --------------------------------------------------------------------------------------
# proxy tables (train split only)
# --------------------------------------------------------------------------------------
def build_proxy_tables(train_data, cold_role: str) -> tuple[dict[str, dict[int, list[int]]], list[int]]:
    """partner id -> cold-role ids ranked by train co-occurrence, plus a global popularity ranking."""
    cold_ids = getattr(train_data, ROLE_ATTR[cold_role])
    tables: dict[str, dict[int, list[int]]] = {}

    for partner in PARTNERS[cold_role]:
        partner_ids = getattr(train_data, ROLE_ATTR[partner])
        counts: dict[int, Counter] = defaultdict(Counter)
        for c, p in zip(cold_ids, partner_ids):
            counts[int(p)][int(c)] += 1
        tables[partner] = {
            p: [cid for cid, _ in cnt.most_common()] for p, cnt in counts.items()
        }

    global_counts = Counter(int(c) for c in cold_ids)
    popular = [cid for cid, _ in global_counts.most_common()]
    return tables, popular


def pick_proxies(masked_ids: np.ndarray, partner_ids: np.ndarray | None,
                 table: dict[int, list[int]] | None, popular: list[int]) -> np.ndarray:
    """Proxy per row; 0 means 'no proxy available' (falls back to an empty history)."""
    out = np.zeros(len(masked_ids), dtype=np.int64)
    for i, masked in enumerate(masked_ids):
        masked = int(masked)
        ranked = popular if table is None else table.get(int(partner_ids[i]), [])
        for cand in ranked:
            if cand != masked:
                out[i] = cand
                break
    return out


# --------------------------------------------------------------------------------------
# inference-time simulation
# --------------------------------------------------------------------------------------
@contextmanager
def coldstart_context(model, cold_role: str, proxy_fn):
    """Mask the cold role's history; optionally substitute a warm proxy's history instead."""
    _orig = model.compute_tripartite_temporal_embeddings

    def simulated(self, user_node_ids, streamer_node_ids, item_node_ids, interact_times):
        ids = {"user": user_node_ids, "streamer": streamer_node_ids, "item": item_node_ids}
        hist = {
            role: [list(x) for x in self.neighbor_sampler.get_all_first_hop_neighbors(arr, interact_times)]
            for role, arr in ids.items()
        }

        n = len(ids[cold_role])
        if proxy_fn is None:
            proxy_ids = np.zeros(n, dtype=np.int64)
        else:
            proxy_ids = proxy_fn(ids)

        has_proxy = proxy_ids > 0
        if has_proxy.any():
            # The sampler needs valid ids everywhere; rows without a proxy are blanked below.
            lookup_ids = np.where(has_proxy, proxy_ids, ids[cold_role])
            p_nbr, p_edge, p_time = self.neighbor_sampler.get_all_first_hop_neighbors(lookup_ids, interact_times)
            p_nbr, p_edge, p_time = list(p_nbr), list(p_edge), list(p_time)
        else:
            p_nbr = p_edge = p_time = [None] * n

        empty_i = np.array([], dtype=np.int64)
        empty_f = np.array([], dtype=np.float32)
        cold_nbr, cold_edge, cold_time = [], [], []
        for i in range(n):
            if has_proxy[i]:
                cold_nbr.append(p_nbr[i])
                cold_edge.append(p_edge[i])
                cold_time.append(p_time[i])
            else:
                cold_nbr.append(empty_i.copy())
                cold_edge.append(empty_i.copy())
                cold_time.append(empty_f.copy())
        hist[cold_role] = [cold_nbr, cold_edge, cold_time]

        padded = {}
        for role, arr in ids.items():
            nbr, edge, tim = hist[role]
            padded[role] = self.pad_sequences(
                arr, interact_times, nbr, edge, tim, self.patch_size, self.max_input_sequence_length
            )

        u_pad_ids, u_pad_e, u_pad_t = padded["user"]
        v_pad_ids, v_pad_e, v_pad_t = padded["streamer"]
        w_pad_ids, w_pad_e, w_pad_t = padded["item"]

        u_co, v_co, w_co = self.cooccurrence_encoder(u_pad_ids, v_pad_ids, w_pad_ids)

        u_n, u_e, u_tf = self.get_features(interact_times, u_pad_ids, u_pad_e, u_pad_t)
        v_n, v_e, v_tf = self.get_features(interact_times, v_pad_ids, v_pad_e, v_pad_t)
        w_n, w_e, w_tf = self.get_features(interact_times, w_pad_ids, w_pad_e, w_pad_t)

        u_pn, u_pe, u_pt, u_pc = self.get_patches(u_n, u_e, u_tf, u_co, self.patch_size)
        v_pn, v_pe, v_pt, v_pc = self.get_patches(v_n, v_e, v_tf, v_co, self.patch_size)
        w_pn, w_pe, w_pt, w_pc = self.get_patches(w_n, w_e, w_tf, w_co, self.patch_size)

        zu = self._project_and_fuse_patches(u_pn, u_pe, u_pt, u_pc)
        zv = self._project_and_fuse_patches(v_pn, v_pe, v_pt, v_pc)
        zw = self._project_and_fuse_patches(w_pn, w_pe, w_pt, w_pc)

        z = torch.cat([zu, zv, zw], dim=1)
        for transformer in self.transformers:
            z = transformer(z)

        l_u, l_v = zu.shape[1], zv.shape[1]
        h_u = self.output_proj_u(torch.mean(z[:, :l_u, :], dim=1))
        h_v = self.output_proj_v(torch.mean(z[:, l_u : l_u + l_v, :], dim=1))
        h_w = self.output_proj_w(torch.mean(z[:, l_u + l_v :, :], dim=1))
        return h_u, h_v, h_w

    model.compute_tripartite_temporal_embeddings = types.MethodType(simulated, model)
    try:
        yield
    finally:
        model.compute_tripartite_temporal_embeddings = _orig


def make_proxy_fn(strategy: str, cold_role: str, tables: dict, popular: list[int]):
    if strategy == STRATEGY_ZERO:
        return None
    if strategy == STRATEGY_POPULAR:
        return lambda ids: pick_proxies(ids[cold_role], None, None, popular)

    partner = strategy.replace("proxy_", "")
    table = tables[partner]
    return lambda ids: pick_proxies(ids[cold_role], ids[partner], table, popular)


# --------------------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------------------
def print_table(results: dict, strategies: list[str]) -> None:
    headers = ("Strategy", *METRIC_HEADERS)
    rows = []
    for s in strategies:
        cells = []
        for k in METRIC_KEYS:
            mean, std = results[s][k]
            cells.append("n/a" if np.isnan(mean) else f"{mean:.4f}+/-{std:.4f}")
        rows.append((s, *cells))

    widths = [len(h) for h in headers]
    for row in rows:
        for j, c in enumerate(row):
            widths[j] = max(widths[j], len(c))

    def _pad(cells) -> str:
        return "| " + " | ".join(str(c).ljust(widths[i]) for i, c in enumerate(cells)) + " |"

    print("\n### Cold-node warm-start strategies (mean +/- std over seeds)\n")
    print(_pad(headers))
    print("|" + "|".join("-" * (w + 2) for w in widths) + "|")
    for row in rows:
        print(_pad(row))


def print_significance(sig_rows: list[dict]) -> None:
    print("\n### Paired test vs. `zero` (seed-averaged per-query)\n")
    print("| Strategy | Metric | n | mean diff | 95% CI | wins/losses | p |")
    print("|----------|--------|---|-----------|--------|-------------|---|")
    for r in sig_rows:
        print(f"| {r['strategy']} | {r['metric']} | {r['n']} | {r['mean_diff']:+.4f} "
              f"| [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] | {r['wins']}/{r['losses']} "
              f"| {r['p_value']:.3g} |")


def plot_strategies(results: dict, strategies: list[str], cold_role: str, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(strategies))
    for ax, metric, title in ((axes[0], "precision@1", f"HR@1 under cold {cold_role}"),
                              (axes[1], "ndcg@10", f"NDCG@10 under cold {cold_role}")):
        means = [results[s][metric][0] for s in strategies]
        stds = [results[s][metric][1] for s in strategies]
        ax.bar(x, means, 0.6, yerr=stds, capsize=4, color="#3498db")
        ax.set_xticks(x)
        ax.set_xticklabels(strategies, rotation=15, ha="right")
        ax.set_ylabel(metric)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    p = out_dir / f"fig_coldstart_warmstart_{cold_role}.pdf"
    plt.savefig(p, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"Wrote figure          -> {p}")


# --------------------------------------------------------------------------------------
def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Warm-start remedies for cold nodes (zero-history simulation).")
    p.add_argument("--dataset_name", type=str, default="kuailive_tripartite")
    p.add_argument("--data_dir", type=str, default=None)
    p.add_argument("--cold_role", type=str, default="streamer", choices=sorted(ROLE_ATTR),
                   help="Which role is treated as brand new. NOTE: pick the role that actually "
                        "collapses in the cold-start table for this dataset.")
    p.add_argument("--model_name", type=str, default="HTTransformer")
    p.add_argument("--run_seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--suffix", type=str, default=None,
                   help="Checkpoint suffix, e.g. _neg5_popneg. Auto-discovered when omitted.")
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
                   help="Legacy uniform negatives; NOT comparable to the main table.")
    p.add_argument("--n_bootstrap", type=int, default=2000)
    p.add_argument("--no_plots", action="store_true")
    p.add_argument("--out_csv", type=str, default=None,
                   help="Default: experiment_results/coldstart_warmstart_<cold_role>.csv")
    return p.parse_args()


def main() -> None:
    warnings.filterwarnings("ignore")
    args = get_args()
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu"
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("eval_coldstart_warmstart")

    out_csv = Path(args.out_csv) if args.out_csv else (
        ROOT / "experiment_results" / f"coldstart_warmstart_{args.cold_role}.csv"
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)

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

    # The column named "streamer" is not always the streamer (see utils/roles.py), so state the
    # real identity of the chosen --cold_role instead of letting the user assume it.
    mapping = detect_roles(full_data)
    if args.cold_role in ("streamer", "item"):
        attr = ROLE_ATTR[args.cold_role]
        real_role = "streamer" if attr == mapping.streamer_attr else "room/item"
        print(f"--cold_role {args.cold_role} -> data.{attr}, which this dataset's structure "
              f"identifies as the real {real_role.upper()}.")
        if args.cold_role == "streamer" and mapping.swapped:
            print("!! WARNING: on this dataset the real streamers live in the OTHER column. "
                  "For a genuine new-streamer scenario use --cold_role item.")
        elif args.cold_role == "item" and not mapping.swapped:
            print("!! WARNING: on this dataset --cold_role item masks rooms/items, not streamers. "
                  "For a new-streamer scenario use --cold_role streamer.")

    tables, popular = build_proxy_tables(train_data, args.cold_role)
    partners = PARTNERS[args.cold_role]
    strategies = [STRATEGY_ZERO, STRATEGY_POPULAR] + [f"proxy_{p}" for p in partners]
    print(f"\nCold role: {args.cold_role} | strategies: {strategies}")
    for p in partners:
        print(f"  proxy_{p}: {len(tables[p])} distinct {p} keys with train history")

    per_query_by_strategy: dict[str, list[list[dict]]] = {s: [] for s in strategies}
    per_seed_rows: list[dict] = []

    for seed in args.run_seeds:
        folder, stem = resolve_checkpoint(args.model_name, args.dataset_name, seed, args.suffix)
        model = build_model(args.model_name, args, node_raw_features=node_raw_features,
                            edge_raw_features=edge_raw_features, node_type_ids=node_type_ids,
                            neighbor_sampler=train_sampler, device=device)
        load_checkpoint(model, folder, stem, args.model_name, logger)
        model = convert_to_gpu(model, device=device)

        for strategy in strategies:
            proxy_fn = make_proxy_fn(strategy, args.cold_role, tables, popular)
            print(f"Evaluating seed {seed} | strategy {strategy} ...")
            with coldstart_context(model, args.cold_role, proxy_fn):
                _loss, agg, per_query = evaluate_tripartite_ranking(
                    model=model, neighbor_sampler=full_sampler, data=test_data,
                    idx_data_loader=test_loader, node_type_ids=node_type_ids, device=device,
                    num_negatives=args.num_ranking_negatives,
                    eval_rng=np.random.RandomState(seed=eval_rank_seed),
                    return_per_query=True, eval_candidates=eval_candidates,
                )
            per_query_by_strategy[strategy].append(list(per_query))
            per_seed_rows.append({"strategy": strategy, "seed": seed,
                                  **{k: agg.get(k, float("nan")) for k in METRIC_KEYS}})

    df_seed = pd.DataFrame(per_seed_rows)
    results = {
        s: {k: (float(df_seed[df_seed.strategy == s][k].mean()),
                float(df_seed[df_seed.strategy == s][k].std(ddof=0)))
            for k in METRIC_KEYS}
        for s in strategies
    }
    print_table(results, strategies)

    rng = np.random.RandomState(args.eval_seed)
    sig_rows: list[dict] = []
    for metric in TEST_METRICS:
        base_pq = seed_averaged_per_query(per_query_by_strategy[STRATEGY_ZERO], metric)
        for s in strategies:
            if s == STRATEGY_ZERO:
                continue
            pq = seed_averaged_per_query(per_query_by_strategy[s], metric)
            sig_rows.append({"strategy": s, "metric": metric,
                             **paired_test(pq, base_pq, args.n_bootstrap, rng)})
    print_significance(sig_rows)

    rows = []
    for s in strategies:
        row = {"cold_role": args.cold_role, "strategy": s, "n_seeds": len(args.run_seeds)}
        for k in METRIC_KEYS:
            row[f"{k}_mean"], row[f"{k}_std"] = results[s][k]
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\nWrote strategy metrics -> {out_csv}")

    df_seed.to_csv(out_csv.with_name(out_csv.stem + "_per_seed.csv"), index=False)
    print(f"Wrote per-seed metrics -> {out_csv.with_name(out_csv.stem + '_per_seed.csv')}")
    pd.DataFrame(sig_rows).to_csv(out_csv.with_name(out_csv.stem + "_significance.csv"), index=False)
    print(f"Wrote significance     -> {out_csv.with_name(out_csv.stem + '_significance.csv')}")

    if args.no_plots:
        print("\nSkipping figure (--no_plots).")
    else:
        plot_strategies(results, strategies, args.cold_role, out_csv.parent)


if __name__ == "__main__":
    main()
