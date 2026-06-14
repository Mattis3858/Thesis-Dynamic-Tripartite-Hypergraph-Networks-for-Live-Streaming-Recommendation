"""
Build the fixed, model-independent ranking-candidate set ONCE, persist it, and (optionally) verify
that any two consumers read byte-identical negatives.

Run this before evaluating any model under the new protocol. Every model / ablation / baseline
must then point ``--eval_candidates_path`` at the file produced here (the train script auto-builds
it at the default path on first run, but running this explicitly is the documented, auditable step).

Examples
--------
    # build the main test candidate set (joint (streamer, room) replacement) and verify
    python build_eval_candidates.py --verify

    # full-data GPU machine, custom mixture / window
    python build_eval_candidates.py --num_negatives 99 --neg_ratio_popularity 0.5 \
        --neg_ratio_hard 0.3 --neg_ratio_uniform 0.2 --seed 1

The output path mirrors the train script's default:
    eval_candidates/{dataset}_{split}_{setting}_seed{seed}.npz
"""

from __future__ import annotations

import argparse
import hashlib
import logging
from pathlib import Path

import numpy as np

from utils.DataLoader import get_tripartite_link_prediction_data
from utils.eval_candidates import (
    EvalCandidateConfig,
    build_eval_candidates,
    load_eval_candidates,
    save_eval_candidates,
)

ROOT = Path(__file__).resolve().parent


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _default_path(dataset: str, split: str, setting: str, seed: int) -> Path:
    return ROOT / "eval_candidates" / f"{dataset}_{split}_{setting}_seed{seed}.npz"


def _print_query_negatives(cand, qi: int, k: int = 10) -> None:
    if qi >= cand.num_queries:
        print(f"  (query {qi} out of range; only {cand.num_queries} queries)")
        return
    m = int(cand.neg_count[qi])
    negs = list(zip(cand.neg_v[qi, : min(k, m)].tolist(), cand.neg_w[qi, : min(k, m)].tolist()))
    print(
        f"  query {qi}: u={int(cand.u[qi])} positive=({int(cand.v_pos[qi])},{int(cand.w_pos[qi])}) "
        f"t={cand.t[qi]:.4g} | {m} negatives, first {len(negs)}: {negs}"
    )


def get_build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Build fixed tripartite ranking candidates")
    p.add_argument("--dataset_name", type=str, default="kuailive_tripartite")
    p.add_argument("--data_dir", type=str, default=None)
    p.add_argument("--val_ratio", type=float, default=0.15)
    p.add_argument("--test_ratio", type=float, default=0.15)
    p.add_argument("--splits", type=str, nargs="+", default=["test"], choices=["test", "val"],
                   help="Which splits to build candidates for (default: test only).")
    p.add_argument("--num_negatives", type=int, default=99)
    p.add_argument("--neg_ratio_popularity", type=float, default=0.5)
    p.add_argument("--neg_ratio_hard", type=float, default=0.3)
    p.add_argument("--neg_ratio_uniform", type=float, default=0.2)
    p.add_argument("--neg_popularity_window", type=float, default=None)
    p.add_argument("--neg_candidate_universe", type=str, default="observed", choices=["observed", "product"])
    p.add_argument("--setting", type=str, default="joint_vw", choices=["joint_vw", "streamer_only", "item_only"])
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out_path", type=str, default=None, help="Override output path (single split only).")
    p.add_argument("--verify", action="store_true", help="Print sample negatives + sha256 and confirm two reads match.")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = get_build_args()

    (
        _node_feat, _edge_feat, node_type_ids, full_data, train_data, val_data, test_data,
        _nnv, _nnt, _tf, _tt,
    ) = get_tripartite_link_prediction_data(
        dataset_name=args.dataset_name, val_ratio=args.val_ratio, test_ratio=args.test_ratio, data_dir=args.data_dir,
    )

    split_data = {"test": test_data, "val": val_data}
    for split in args.splits:
        cfg = EvalCandidateConfig(
            num_negatives=args.num_negatives,
            ratio_popularity=args.neg_ratio_popularity,
            ratio_hard=args.neg_ratio_hard,
            ratio_uniform=args.neg_ratio_uniform,
            popularity_window=args.neg_popularity_window,
            candidate_universe=args.neg_candidate_universe,
            setting=args.setting,
            seed=args.seed,
        )
        cand = build_eval_candidates(
            split_data[split], full_data, node_type_ids, cfg, train_data=train_data
        )
        if args.out_path and len(args.splits) == 1:
            out = Path(args.out_path)
        else:
            out = _default_path(args.dataset_name, split, args.setting, args.seed)
        save_eval_candidates(out, cand)
        digest = _sha256(out)
        print(f"\n[{split}] saved -> {out}")
        print(f"[{split}] sha256 = {digest}")
        print(f"[{split}] summary = {cand.summary()}")

        if args.verify:
            print(f"\n--- verification ({split}) ---")
            print("(a) sample negatives:")
            _print_query_negatives(cand, 0)
            _print_query_negatives(cand, 100)
            # (b) two independent reads must be byte-identical -> any two models read the same negatives
            r1 = load_eval_candidates(out)
            r2 = load_eval_candidates(out)
            same = (
                np.array_equal(r1.neg_v, r2.neg_v)
                and np.array_equal(r1.neg_w, r2.neg_w)
                and np.array_equal(r1.neg_v, cand.neg_v)
            )
            print(f"(b) two independent reads identical (model A == model B): {same}")
            print(f"    read#1 q0 negs: {list(zip(r1.neg_v[0,:5].tolist(), r1.neg_w[0,:5].tolist()))}")
            print(f"    read#2 q0 negs: {list(zip(r2.neg_v[0,:5].tolist(), r2.neg_w[0,:5].tolist()))}")


if __name__ == "__main__":
    main()
