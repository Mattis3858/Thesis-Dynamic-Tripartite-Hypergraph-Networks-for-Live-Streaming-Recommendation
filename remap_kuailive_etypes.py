"""
Fix the streamer/room role swap before feeding utils.preprocess_kuailive output into the thesis
converter (preprocess_data/preprocess_kuailive_tripartite.py).

The two scripts use INCOMPATIBLE etype conventions:
  utils.preprocess_kuailive (source):   0=user->room, 1=user->streamer, 2=streamer->room
  preprocess_kuailive_tripartite (dst): 0=user->streamer, 1=user->room, 2=room->streamer

Feeding the source directly makes the converter label rooms as streamers (verified: a 50-streamer /
1613-room slice came out as num_streamers=1613, num_rooms=50). This script rewrites the edge CSV into
the converter's convention so streamer/room end up correct:
  source etype 0 (u->room)      -> etype 1 (u->room)           [src,dst unchanged]
  source etype 1 (u->streamer)  -> etype 0 (u->streamer)       [src,dst unchanged]
  source etype 2 (streamer->room) -> etype 2 (room->streamer)  [swap src<->dst]

Edge-feature .npy and node .npy are unchanged (idx is preserved; the converter re-sorts internally).

Usage:
    python remap_kuailive_etypes.py --in_edges kuailive_data/ml_kuailive_edges.csv --out_edges kuailive_data/ml_kuailive_edges_fixed.csv
Then run the converter with --edges kuailive_data/ml_kuailive_edges_fixed.csv and verify the meta shows
the TRUE streamer/room counts.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd


def remap(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={c: c.strip().lower() for c in df.columns}).copy()
    for col in ("src", "dst", "etype"):
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}'; got {list(df.columns)}")
    orig = df["etype"].to_numpy()
    # swap src<->dst where the source etype is 2 (streamer->room  =>  room->streamer)
    mask2 = orig == 2
    src2 = df.loc[mask2, "src"].to_numpy().copy()
    df.loc[mask2, "src"] = df.loc[mask2, "dst"].to_numpy()
    df.loc[mask2, "dst"] = src2
    # remap etype: 0->1, 1->0, 2->2
    df["etype"] = np.select([orig == 0, orig == 1, orig == 2], [1, 0, 2], default=orig)
    return df


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in_edges", required=True)
    p.add_argument("--out_edges", required=True)
    args = p.parse_args()
    df = pd.read_csv(args.in_edges)
    out = remap(df)
    out.to_csv(args.out_edges, index=False)
    print(f"Remapped {len(out)} rows -> {args.out_edges}")
    print("etype counts (after):", out["etype"].value_counts().to_dict())


if __name__ == "__main__":
    main()
