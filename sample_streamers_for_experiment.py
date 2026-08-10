"""
Sample a balanced streamer subset for the group-wise model experiment (committee follow-up #2).

Takes the bias groups produced by ``analyze_streamer_bias.py`` on a large slice, picks
``--per_group`` streamers from each group, and writes a row-subset of the pairwise edge CSV
containing only those streamers (plus their rooms and audiences). Feeding that subset through
the usual converter yields ONE mixed dataset holding all groups, so a single training round
supports a per-group stratified evaluation and group differences are not confounded by
dataset-specific training.

Selection is stratified by interaction volume inside each group (``--n_strata`` quantile bins),
so the picked streamers span the volume range instead of clumping at the head or the tail --
otherwise "group X performs worse" could just mean "group X's sample had less data". The
per-group volume balance is printed and written out for the record.

The output CSV keeps the original ``idx`` values untouched, so it stays aligned with the
original ``ml_kuailive_features.npy`` / ``ml_kuailive_node.npy``; reuse those unchanged.

Usage:
  python sample_streamers_for_experiment.py \
      --streamer_groups experiment_results/streamer_bias_groups.csv \
      --edges path/to/ml_kuailive_edges.csv --etype_convention source \
      --per_group 10 --out_edges path/to/ml_kuailive_edges_grouped30.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_streamer_bias import _CONVENTIONS, load_edges

ROOT = Path(__file__).resolve().parent
DEFAULT_OUT_DIR = ROOT / "experiment_results"


def stratified_pick(
    group_df: pd.DataFrame, n_pick: int, n_strata: int, rng: np.random.RandomState
) -> pd.DataFrame:
    """Pick n_pick streamers spanning the group's interaction-volume range."""
    if len(group_df) <= n_pick:
        return group_df.copy()

    ordered = group_df.sort_values("interaction_count").reset_index(drop=True)
    n_strata = max(1, min(n_strata, n_pick, len(ordered)))
    bins = np.array_split(np.arange(len(ordered)), n_strata)

    base, extra = divmod(n_pick, n_strata)
    quotas = [base + (1 if i < extra else 0) for i in range(n_strata)]

    picked_pos: list[int] = []
    for positions, quota in zip(bins, quotas):
        take = min(quota, len(positions))
        if take > 0:
            picked_pos.extend(rng.choice(positions, size=take, replace=False).tolist())

    # Top up from whatever is left if a thin stratum could not fill its quota.
    if len(picked_pos) < n_pick:
        remaining = [i for i in range(len(ordered)) if i not in set(picked_pos)]
        need = min(n_pick - len(picked_pos), len(remaining))
        picked_pos.extend(rng.choice(remaining, size=need, replace=False).tolist())

    return ordered.iloc[sorted(picked_pos)].copy()


def filter_edges_to_streamers(
    df: pd.DataFrame, streamers: set[int], convention: str
) -> tuple[pd.DataFrame, set[int], int]:
    """Keep every edge of the selected streamers, their rooms, and their audiences."""
    conv = _CONVENTIONS[convention]
    e_us, e_sr = conv["user_streamer"], conv["streamer_room"]
    e_ur = ({0, 1, 2} - {e_us, e_sr}).pop()
    s_col, r_col = ("src", "dst") if conv["streamer_is_src"] else ("dst", "src")

    sr_all = df[df["etype"] == e_sr]
    sr_keep = sr_all[sr_all[s_col].isin(streamers)]
    rooms = set(sr_keep[r_col].astype(int).tolist())

    # A room owned by several streamers would drag foreign interactions in through the
    # user->room layer; report it rather than silently widening the slice.
    room_owners = sr_all.groupby(r_col)[s_col].nunique()
    shared_rooms = int((room_owners.loc[list(rooms)] > 1).sum()) if rooms else 0

    keep = pd.concat(
        [
            df[(df["etype"] == e_us) & (df["dst"].isin(streamers))],
            sr_keep,
            df[(df["etype"] == e_ur) & (df["dst"].isin(rooms))],
        ]
    )
    keep = keep.sort_values("idx" if "idx" in keep.columns else "ts").reset_index(drop=True)
    return keep, rooms, shared_rooms


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample streamers per bias group and slice the edge table.")
    parser.add_argument("--streamer_groups", type=str,
                        default=str(DEFAULT_OUT_DIR / "streamer_bias_groups.csv"),
                        help="CSV from analyze_streamer_bias.py (needs streamer_id, interaction_count, bias_group).")
    parser.add_argument("--edges", type=str, required=True, help="The same large edge CSV analyzed in S1.")
    parser.add_argument("--etype_convention", type=str, default="source", choices=sorted(_CONVENTIONS))
    parser.add_argument("--per_group", type=int, default=10, help="Streamers to pick from each bias group.")
    parser.add_argument("--n_strata", type=int, default=3, help="Volume strata per group for balanced picking.")
    parser.add_argument("--seed", type=int, default=2020)
    parser.add_argument("--keep_label_duplicates", action="store_true",
                        help="Keep multi-label rows; must match the setting used in analyze_streamer_bias.py.")
    parser.add_argument("--out_edges", type=str, required=True, help="Where to write the sliced edge CSV.")
    parser.add_argument("--out_dir", type=str, default=str(DEFAULT_OUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = get_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(args.seed)

    groups = pd.read_csv(args.streamer_groups)
    for col in ("streamer_id", "interaction_count", "bias_group"):
        if col not in groups.columns:
            raise ValueError(f"{args.streamer_groups} is missing column '{col}'.")

    print(f"Loaded {len(groups)} clustered streamers from {args.streamer_groups}")
    print(f"Groups: {dict(groups['bias_group'].value_counts())}\n")

    picks = []
    for name, gdf in groups.groupby("bias_group"):
        sel = stratified_pick(gdf, args.per_group, args.n_strata, rng)
        if len(sel) < args.per_group:
            print(f"!! WARNING: group '{name}' has only {len(gdf)} streamers; picked all {len(sel)}.")
        picks.append(sel)
    selected = pd.concat(picks).sort_values(["bias_group", "streamer_id"]).reset_index(drop=True)

    print("=== Selected streamers per group (volume balance check) ===")
    balance = (
        selected.groupby("bias_group")
        .agg(
            n_selected=("streamer_id", "count"),
            total_interactions=("interaction_count", "sum"),
            mean_interactions=("interaction_count", "mean"),
            min_interactions=("interaction_count", "min"),
            max_interactions=("interaction_count", "max"),
            mean_audience_hhi=("audience_hhi", "mean"),
            mean_item_hhi=("item_hhi", "mean"),
        )
        .round(4)
    )
    print(balance.to_string())
    spread = balance["total_interactions"].max() / max(balance["total_interactions"].min(), 1)
    print(f"\nVolume spread across groups (max/min total interactions): {spread:.2f}x")
    if spread > 3:
        print("!! WARNING: groups are strongly imbalanced in volume; per-group metric gaps may reflect "
              "data volume rather than streamer type. Consider re-running with a different --seed or "
              "capping per-streamer interactions.")

    sel_csv = out_dir / "selected_streamers.csv"
    selected.to_csv(sel_csv, index=False)
    balance.to_csv(out_dir / "selected_streamers_balance.csv")
    print(f"\nWrote selection -> {sel_csv}")
    print(f"Wrote balance   -> {out_dir / 'selected_streamers_balance.csv'}")

    print(f"\nLoading edges from {args.edges} ...")
    df_edges = load_edges(Path(args.edges), not args.keep_label_duplicates)
    streamer_set = set(selected["streamer_id"].astype(int).tolist())
    sliced, rooms, shared_rooms = filter_edges_to_streamers(df_edges, streamer_set, args.etype_convention)

    if shared_rooms:
        print(f"!! WARNING: {shared_rooms} of the {len(rooms)} kept rooms are shared by more than one "
              "streamer, so the slice may include interactions of non-selected streamers.")

    out_edges = Path(args.out_edges)
    out_edges.parent.mkdir(parents=True, exist_ok=True)
    sliced.to_csv(out_edges, index=False)

    print(f"\n=== Slice summary ===")
    print(f"  streamers : {len(streamer_set)}")
    print(f"  rooms     : {len(rooms)}")
    print(f"  edge rows : {len(sliced)} (of {len(df_edges)})")
    print(f"  written   -> {out_edges}")

    meta = {
        "source_edges": str(Path(args.edges)),
        "etype_convention": args.etype_convention,
        "per_group": args.per_group,
        "n_strata": args.n_strata,
        "seed": args.seed,
        "num_streamers": len(streamer_set),
        "num_rooms": len(rooms),
        "num_edge_rows": int(len(sliced)),
        "shared_rooms": shared_rooms,
        "groups": {str(k): int(v) for k, v in selected["bias_group"].value_counts().items()},
    }
    meta_path = out_dir / "selected_streamers_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"  meta      -> {meta_path}")

    print("\nNext steps:")
    if args.etype_convention == "source":
        print(f"  1) python remap_kuailive_etypes.py --in_edges {out_edges} --out_edges {out_edges.with_name(out_edges.stem + '_fixed.csv')}")
        conv_in = out_edges.with_name(out_edges.stem + "_fixed.csv")
    else:
        print("  1) (already in converter convention; no remap needed)")
        conv_in = out_edges
    print(f"  2) python preprocess_data/preprocess_kuailive_tripartite.py --edges {conv_in} \\")
    print("       --node_features <ml_kuailive_node.npy> --edge_features <ml_kuailive_features.npy> \\")
    print("       --out_dir processed_data/kuailive_streamer_groups --dataset_name kuailive_streamer_groups")
    print("     (verify the meta shows the TRUE streamer/room counts)")


if __name__ == "__main__":
    main()
