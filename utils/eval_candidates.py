"""
Fixed, pre-generated test/val candidate sets for tripartite ranking evaluation.

Why this exists
---------------
The legacy protocol sampled the N ranking negatives *inside* ``evaluate_tripartite_ranking``
at scoring time, so different models / runs could see different negatives -- cross-model
comparison was not guaranteed to be fair, and the uniform ``(v', w')`` draws were trivially
easy (the (streamer, room) combo space barely overlaps the positives), saturating HR@K.

This module builds the candidate set **once**, persists it to disk, and lets every model read
the *same* file. It also replaces uniform sampling with a harder, structured mixture.

Hard rules honoured here (see MODIFY_EVAL_PLAN.md):
  * Negatives are model-independent: built from data + a fixed seed, never from model scores.
  * No time leakage: negatives reuse the positive's timestamp ``t``; popularity is measured
    strictly over interactions with ``ts < t``. Neighbour sampling (strictly < t) is the
    forward's job and is untouched.
  * Reproducible: one ``np.random.RandomState(seed)`` advanced in a fixed query order; the
    result is written to an ``.npz`` so all models load byte-identical candidates.
  * False-negative protection: a candidate ``(v', w')`` is never one the user actually
    interacts with anywhere in the full timeline (past *or* future).

Everything scales off the data: pool sizes and the candidate universe are read from
``node_type_ids`` / the interaction tables. Nothing about 78/3 or 294/10 is hard-coded.
On a small dev subset a query may not be able to fill N negatives after exclusions -- that is
logged and the per-query count is recorded; on the full-data GPU machine N=99 is reachable.

Negative-sampling mixture (``setting='joint_vw'``, the thesis main protocol):
  * popularity (default 50%): most-interacted ``(v, w)`` combos as of ``t`` (history < t).
  * user-history hard (default 30%): combos that behaviourally-similar users engage with but
    the current user never has. Similarity (shared-streamer overlap) and combo frequency are
    computed from the TRAIN split only -- no val/test-period leakage into negative difficulty.
  * uniform (default 20%): uniform over the remaining combo universe.
Shortfalls in any source are back-filled from the uniform pool; a fully-exhausted query keeps
fewer than N negatives.

``setting='streamer_only'`` / ``'item_only'`` replace a single role (uniform only for now --
see TODO); ``item_only`` is intended to stay disabled when the room pool is tiny.
"""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

# node_type_ids encoding (see processed_data/.../*_meta.json): 1=user, 2=streamer, 3=room.
STREAMER_TYPE = 2
ROOM_TYPE = 3

VALID_SETTINGS = ("joint_vw", "streamer_only", "item_only")


@dataclass
class EvalCandidateConfig:
    """All knobs for candidate generation. Persisted alongside the candidates for provenance."""

    num_negatives: int = 99
    ratio_popularity: float = 0.5
    ratio_hard: float = 0.3
    ratio_uniform: float = 0.2
    # Popularity time window ending at t, in the dataset's own ts units. None = use all history
    # before t (cumulative). TODO(GPU): try a finite window (e.g. last D days) on full data and
    # pick a default that best separates popular-vs-cold; small dev data has too few events to tune.
    popularity_window: Optional[float] = None
    # How many most-similar users feed the hard-negative pool. TODO(GPU): tune on full data.
    num_similar_users: int = 50
    # "observed" = only (streamer, room) combos that actually occur in the data (realistic, harder);
    # "product" = full streamer x room grid (legacy-style, includes combos that never happen).
    candidate_universe: str = "observed"
    setting: str = "joint_vw"
    seed: int = 1

    def validate(self) -> None:
        if self.num_negatives < 1:
            raise ValueError(f"num_negatives must be >= 1, got {self.num_negatives}")
        if self.setting not in VALID_SETTINGS:
            raise ValueError(f"setting must be one of {VALID_SETTINGS}, got {self.setting!r}")
        if self.candidate_universe not in ("observed", "product"):
            raise ValueError(f"candidate_universe must be 'observed' or 'product', got {self.candidate_universe!r}")
        s = self.ratio_popularity + self.ratio_hard + self.ratio_uniform
        if not np.isclose(s, 1.0):
            raise ValueError(f"ratios must sum to 1.0, got {s} ({self.ratio_popularity}/{self.ratio_hard}/{self.ratio_uniform})")
        for r in (self.ratio_popularity, self.ratio_hard, self.ratio_uniform):
            if r < 0:
                raise ValueError("ratios must be non-negative")


@dataclass
class EvalCandidates:
    """
    Per-query ranking candidates aligned to ``eval_data`` row order (query i == data row i).

    ``neg_v`` / ``neg_w`` are ``[Q, N]`` int64, right-padded with ``-1`` where a query could not
    fill N negatives; ``neg_count[i]`` is the number of valid (non-``-1``) negatives for query i.
    The positive ``(v_pos[i], w_pos[i])`` is always the rank candidate at index 0 downstream.
    """

    query_ids: np.ndarray  # [Q]
    u: np.ndarray  # [Q]
    v_pos: np.ndarray  # [Q]
    w_pos: np.ndarray  # [Q]
    t: np.ndarray  # [Q]
    neg_v: np.ndarray  # [Q, N]
    neg_w: np.ndarray  # [Q, N]
    neg_count: np.ndarray  # [Q]
    config: dict = field(default_factory=dict)

    @property
    def num_queries(self) -> int:
        return int(self.query_ids.shape[0])

    @property
    def num_negatives(self) -> int:
        return int(self.neg_v.shape[1])

    def summary(self) -> dict:
        cnt = self.neg_count
        return {
            "num_queries": self.num_queries,
            "num_negatives_target": self.num_negatives,
            "neg_count_min": int(cnt.min()) if cnt.size else 0,
            "neg_count_max": int(cnt.max()) if cnt.size else 0,
            "neg_count_mean": float(cnt.mean()) if cnt.size else 0.0,
            "queries_underfilled": int(np.sum(cnt < self.num_negatives)),
        }


def _pools(node_type_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    streamer_pool = np.where(node_type_ids == STREAMER_TYPE)[0].astype(np.int64)
    room_pool = np.where(node_type_ids == ROOM_TYPE)[0].astype(np.int64)
    if len(streamer_pool) == 0 or len(room_pool) == 0:
        raise ValueError("node_type_ids must contain at least one streamer (type 2) and one room (type 3).")
    return streamer_pool, room_pool


def _combo_universe(full_data, streamer_pool: np.ndarray, room_pool: np.ndarray, universe: str) -> np.ndarray:
    """Return [C, 2] array of candidate (streamer, room) global-id pairs, sorted for determinism."""
    if universe == "observed":
        combos = sorted(
            set(zip(full_data.streamer_node_ids.tolist(), full_data.item_node_ids.tolist()))
        )
    else:  # product grid
        combos = [(int(s), int(r)) for s in streamer_pool.tolist() for r in room_pool.tolist()]
        combos.sort()
    return np.asarray(combos, dtype=np.int64).reshape(-1, 2)


def _user_alltime_combos(full_data) -> dict[int, set[int]]:
    """user global id -> set of combo *indices is built later*; here we return (v,w) tuples."""
    out: dict[int, set] = defaultdict(set)
    for u, v, w in zip(
        full_data.user_node_ids.tolist(),
        full_data.streamer_node_ids.tolist(),
        full_data.item_node_ids.tolist(),
    ):
        out[int(u)].add((int(v), int(w)))
    return out


def _precompute_hard_negative_lists(
    train_data,
    combo_to_idx: dict[tuple[int, int], int],
    num_similar_users: int,
    user_alltime_combos: dict[int, set],
) -> dict[int, list[int]]:
    """
    For each user, an ordered list of "hard" combo indices: combos that behaviourally-similar
    users engage with, ranked by frequency among those similar users, excluding combos the user
    has ever touched.

    Time range (no test-period leakage):
      * Similarity (shared streamers) and hard-combo frequency are computed from ``train_data``
        ONLY -- val/test-period behaviour never influences which negatives are "hard". This keeps
        the candidate-difficulty distribution constructible from training-time information.
      * The exclusion ``combo in own`` uses ``user_alltime_combos`` (the user's full-timeline
        interaction set, incl. future): this is the deliberate false-negative protection -- a
        candidate the user truly interacts with at any time must never be labelled a negative.

    TODO(GPU): O(user * neighbour) per user. Fine for hundreds of users; if the full dataset has
    many more, cache this to disk or switch to a sparse co-occurrence matmul.
    """
    user_streamers: dict[int, set] = defaultdict(set)
    train_user_combos: dict[int, set] = defaultdict(set)
    streamer_users: dict[int, set] = defaultdict(set)
    for u, v, w in zip(
        train_data.user_node_ids.tolist(),
        train_data.streamer_node_ids.tolist(),
        train_data.item_node_ids.tolist(),
    ):
        u, v, w = int(u), int(v), int(w)
        user_streamers[u].add(v)
        train_user_combos[u].add((v, w))
        streamer_users[v].add(u)

    hard_lists: dict[int, list[int]] = {}
    for u, streamers in user_streamers.items():
        # candidate similar users: anyone sharing >= 1 streamer with u in TRAIN
        shared = Counter()
        for s in streamers:
            for other in streamer_users[s]:
                if other != u:
                    shared[other] += 1
        if not shared:
            hard_lists[u] = []
            continue
        top_users = [usr for usr, _ in shared.most_common(num_similar_users)]
        own = user_alltime_combos.get(u, set())  # full-timeline false-negative protection (kept)
        combo_freq = Counter()
        for other in top_users:
            for combo in train_user_combos[other]:  # similar users' TRAIN-period combos only
                if combo in own:
                    continue  # never propose a combo the user truly interacts with (any time)
                ci = combo_to_idx.get(combo)
                if ci is not None:
                    combo_freq[ci] += 1
        # most frequent first; ties broken by combo index for determinism
        hard_lists[u] = [ci for ci, _ in sorted(combo_freq.items(), key=lambda kv: (-kv[1], kv[0]))]
    return hard_lists


def build_eval_candidates(
    eval_data,
    full_data,
    node_type_ids: np.ndarray,
    config: EvalCandidateConfig,
    logger: Optional[logging.Logger] = None,
    train_data=None,
) -> EvalCandidates:
    """
    Build the fixed candidate set for ``eval_data`` (e.g. test_data).

    :param eval_data: TripartiteData split to evaluate (queries = its positive rows).
    :param full_data: TripartiteData over all interactions -- used for the false-negative filter
        (user's all-time combos) and for the combo universe.
    :param node_type_ids: global node type array (1=user, 2=streamer, 3=room).
    :param config: EvalCandidateConfig.
    :param train_data: TripartiteData of the TRAIN split -- used (train-only) for hard-negative
        similarity and frequency, so val/test-period behaviour never shapes negative difficulty.
        Required for ``setting='joint_vw'``; if None, the hard-negative source is disabled and
        those slots fall back to uniform (a warning is logged).
    :return: EvalCandidates aligned to eval_data row order.

    Time ranges (no test-period leakage): popularity is strictly as-of-t (history < t); hard
    negatives are train-only; only the deliberate false-negative filter uses the full timeline.
    """
    config.validate()
    log = logger or logging.getLogger(__name__)
    rng = np.random.RandomState(config.seed)

    streamer_pool, room_pool = _pools(node_type_ids)

    if config.setting != "joint_vw":
        return _build_single_role_candidates(eval_data, full_data, streamer_pool, room_pool, config, rng, log)

    combos = _combo_universe(full_data, streamer_pool, room_pool, config.candidate_universe)
    num_combos = combos.shape[0]
    combo_to_idx = {(int(v), int(w)): i for i, (v, w) in enumerate(combos)}
    all_combo_indices = np.arange(num_combos, dtype=np.int64)

    user_pos_combos = _user_alltime_combos(full_data)  # full-timeline false-negative filter
    if train_data is not None:
        hard_lists = _precompute_hard_negative_lists(
            train_data, combo_to_idx, config.num_similar_users, user_pos_combos
        )
    else:
        log.warning(
            "train_data not provided: hard-negative source disabled (its slots fall back to "
            "uniform). Pass train_data to enable train-only hard negatives."
        )
        hard_lists = {}

    # --- popularity as-of t via a sliding window over the time-sorted full timeline ---
    hist_ts = full_data.node_interact_times.astype(np.float64)
    hist_combo_idx = np.array(
        [combo_to_idx.get((int(v), int(w)), -1) for v, w in zip(full_data.streamer_node_ids, full_data.item_node_ids)],
        dtype=np.int64,
    )
    hist_order = np.argsort(hist_ts, kind="stable")
    hist_ts_sorted = hist_ts[hist_order]
    hist_combo_sorted = hist_combo_idx[hist_order]
    pop_counts = np.zeros(num_combos, dtype=np.int64)
    right = 0
    left = 0

    Q = eval_data.num_interactions
    N = config.num_negatives
    n_pop = int(round(config.ratio_popularity * N))
    n_hard = int(round(config.ratio_hard * N))
    n_uni = N - n_pop - n_hard  # absorbs rounding; >= 0 because ratios sum to 1

    eval_u = eval_data.user_node_ids.astype(np.int64)
    eval_v = eval_data.streamer_node_ids.astype(np.int64)
    eval_w = eval_data.item_node_ids.astype(np.int64)
    eval_t = eval_data.node_interact_times.astype(np.float64)

    neg_v = np.full((Q, N), -1, dtype=np.int64)
    neg_w = np.full((Q, N), -1, dtype=np.int64)
    neg_count = np.zeros(Q, dtype=np.int64)

    # process queries in time order so the popularity window can advance monotonically
    eval_order = np.argsort(eval_t, kind="stable")
    window = config.popularity_window

    for qi in eval_order:
        tq = eval_t[qi]
        while right < len(hist_ts_sorted) and hist_ts_sorted[right] < tq:  # strictly before t
            c = hist_combo_sorted[right]
            if c >= 0:
                pop_counts[c] += 1
            right += 1
        if window is not None:
            while left < right and hist_ts_sorted[left] < tq - window:
                c = hist_combo_sorted[left]
                if c >= 0:
                    pop_counts[c] -= 1
                left += 1

        u = int(eval_u[qi])
        v_p, w_p = int(eval_v[qi]), int(eval_w[qi])
        forbidden = set(user_pos_combos.get(u, set()))
        forbidden.add((v_p, w_p))  # never propose the positive itself

        picked: list[int] = []
        picked_set: set[int] = set()

        def _take(source_indices, target_len):
            """Append unseen, non-forbidden combo indices until len(picked) reaches target_len."""
            for ci in source_indices:
                if len(picked) >= target_len:
                    break
                ci = int(ci)
                if ci in picked_set:
                    continue
                if (int(combos[ci, 0]), int(combos[ci, 1])) in forbidden:
                    continue
                picked.append(ci)
                picked_set.add(ci)

        # popularity: most popular combos as-of t (deterministic; ties broken by combo index)
        pop_order = np.lexsort((all_combo_indices, -pop_counts))  # popularity desc, index asc
        _take(pop_order, n_pop)

        # user-history hard negatives (brings the running total up to n_pop + n_hard)
        _take(hard_lists.get(u, []), n_pop + n_hard)

        # uniform fill (also back-fills any shortfall the structured sources left)
        target_total = min(N, num_combos - sum(1 for c in forbidden if c in combo_to_idx))
        remaining = [ci for ci in all_combo_indices if ci not in picked_set
                     and (int(combos[ci, 0]), int(combos[ci, 1])) not in forbidden]
        rng.shuffle(remaining)
        for ci in remaining:
            if len(picked) >= target_total:
                break
            picked.append(ci)
            picked_set.add(ci)

        m = len(picked)
        neg_count[qi] = m
        if m > 0:
            sel = combos[np.asarray(picked, dtype=np.int64)]
            neg_v[qi, :m] = sel[:, 0]
            neg_w[qi, :m] = sel[:, 1]

    cand = EvalCandidates(
        query_ids=np.arange(Q, dtype=np.int64),
        u=eval_u,
        v_pos=eval_v,
        w_pos=eval_w,
        t=eval_t,
        neg_v=neg_v,
        neg_w=neg_w,
        neg_count=neg_count,
        config=asdict(config),
    )
    s = cand.summary()
    log.info("Built eval candidates (setting=joint_vw, universe=%s): %s", config.candidate_universe, s)
    if s["queries_underfilled"] > 0:
        log.warning(
            "%d/%d queries could not reach N=%d negatives (combo universe has only %d combos "
            "minus the user's own interactions). Expected on a small dev subset; N=%d is reachable "
            "on the full dataset.",
            s["queries_underfilled"], Q, N, num_combos, N,
        )
    return cand


def _build_single_role_candidates(
    eval_data, full_data, streamer_pool, room_pool, config, rng, log
) -> EvalCandidates:
    """
    streamer_only: replace v, keep w fixed.  item_only: replace w, keep v fixed.
    Uniform sampling over the role pool with the same false-negative filter.

    TODO(GPU): apply the popularity/hard mixture here too if these settings are reported in the
    paper. Right now only the main joint_vw setting uses the structured mixture.
    """
    if config.setting == "item_only" and len(room_pool) <= config.num_negatives:
        log.warning(
            "item_only setting requested but the room pool has only %d rooms -- cannot form %d "
            "distinct negatives; this setting is expected to be DISABLED (its numbers duplicate the "
            "main setting). Producing what the pool allows for logic-verification only.",
            len(room_pool), config.num_negatives,
        )

    user_pos = _user_alltime_combos(full_data)
    Q = eval_data.num_interactions
    N = config.num_negatives
    eval_u = eval_data.user_node_ids.astype(np.int64)
    eval_v = eval_data.streamer_node_ids.astype(np.int64)
    eval_w = eval_data.item_node_ids.astype(np.int64)
    eval_t = eval_data.node_interact_times.astype(np.float64)

    neg_v = np.full((Q, N), -1, dtype=np.int64)
    neg_w = np.full((Q, N), -1, dtype=np.int64)
    neg_count = np.zeros(Q, dtype=np.int64)

    for qi in range(Q):
        u, v_p, w_p = int(eval_u[qi]), int(eval_v[qi]), int(eval_w[qi])
        forbidden = set(user_pos.get(u, set()))
        forbidden.add((v_p, w_p))
        if config.setting == "streamer_only":
            pool = [int(s) for s in streamer_pool if (int(s), w_p) not in forbidden]
            rng.shuffle(pool)
            pool = pool[:N]
            m = len(pool)
            neg_v[qi, :m] = pool
            neg_w[qi, :m] = w_p
        else:  # item_only
            pool = [int(r) for r in room_pool if (v_p, int(r)) not in forbidden]
            rng.shuffle(pool)
            pool = pool[:N]
            m = len(pool)
            neg_v[qi, :m] = v_p
            neg_w[qi, :m] = pool
        neg_count[qi] = m

    cand = EvalCandidates(
        query_ids=np.arange(Q, dtype=np.int64),
        u=eval_u, v_pos=eval_v, w_pos=eval_w, t=eval_t,
        neg_v=neg_v, neg_w=neg_w, neg_count=neg_count, config=asdict(config),
    )
    log.info("Built eval candidates (setting=%s): %s", config.setting, cand.summary())
    return cand


def save_eval_candidates(path: str | Path, cand: EvalCandidates) -> Path:
    """Persist candidates to a .npz (config stored as a JSON string for provenance)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        query_ids=cand.query_ids,
        u=cand.u,
        v_pos=cand.v_pos,
        w_pos=cand.w_pos,
        t=cand.t,
        neg_v=cand.neg_v,
        neg_w=cand.neg_w,
        neg_count=cand.neg_count,
        config_json=np.array(json.dumps(cand.config), dtype=object),
    )
    return path


def load_eval_candidates(path: str | Path) -> EvalCandidates:
    path = Path(path)
    with np.load(path, allow_pickle=True) as data:
        cfg = json.loads(str(data["config_json"].item())) if "config_json" in data else {}
        return EvalCandidates(
            query_ids=data["query_ids"],
            u=data["u"],
            v_pos=data["v_pos"],
            w_pos=data["w_pos"],
            t=data["t"],
            neg_v=data["neg_v"],
            neg_w=data["neg_w"],
            neg_count=data["neg_count"],
            config=cfg,
        )
