"""
Metrics for DyGLib-style tasks and tripartite ranking evaluation.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

# 1 ground-truth + 99 negatives => 100 candidates; K must be <= num_candidates for meaningful @K.
# K=1,3 added so the strict top-K metrics the thesis tables report (P@1, N@3) are actually
# computed (eval_table10.py reads precision@1 / ndcg@3). If you change this set, regenerate the
# results_*.csv files from scratch -- run_experiments.py mirrors this tuple to build CSV columns.
TRIPARTITE_RANKING_KS: tuple[int, ...] = (1, 3, 5, 10, 20, 50, 100)


def get_link_prediction_metrics(predicts: torch.Tensor, labels: torch.Tensor):
    """
    get metrics for the link prediction task
    :param predicts: Tensor, shape (num_samples, )
    :param labels: Tensor, shape (num_samples, )
    :return:
        dictionary of metrics {'metric_name_1': metric_1, ...}
    """
    predicts = predicts.cpu().detach().numpy()
    labels = labels.cpu().numpy()

    average_precision = average_precision_score(y_true=labels, y_score=predicts)
    roc_auc = roc_auc_score(y_true=labels, y_score=predicts)

    return {"average_precision": average_precision, "roc_auc": roc_auc}


def get_node_classification_metrics(predicts: torch.Tensor, labels: torch.Tensor):
    """
    get metrics for the node classification task
    :param predicts: Tensor, shape (num_samples, )
    :param labels: Tensor, shape (num_samples, )
    :return:
        dictionary of metrics {'metric_name_1': metric_1, ...}
    """
    predicts = predicts.cpu().detach().numpy()
    labels = labels.cpu().numpy()

    roc_auc = roc_auc_score(y_true=labels, y_score=predicts)

    return {"roc_auc": roc_auc}


def tripartite_ranking_metrics_per_query(
    scores: np.ndarray,
    positive_index: int = 0,
    ks: tuple[int, ...] | None = None,
) -> dict[str, float]:
    """
    Ranking metrics for one query: exactly one relevant item at ``positive_index``,
    binary relevance, higher score = better.

    Returns ROC-AUC, Average Precision (AP), and for each K in ``ks``:
    Precision@K, Recall@K, NDCG@K (single-item IDCG = 1/log2(2)).

    Tie handling: HR/Precision/NDCG are the **expected values under a uniform random
    tie-break** -- the positive is equally likely to occupy any of the ``num_tied`` positions
    its score-block spans. This is the analytic equivalent of shuffling tied candidates with a
    random seed (but with no seed variance), and is identical to plain integer-rank metrics
    when there are no ties (the usual case for a continuous model). The old ``argsort`` rank
    always placed the positive (at index 0) ahead of equal-scored candidates, so a model
    emitting constant scores scored HR/NDCG = 1.0 while its AUC was 0.5 (e.g. degenerate HAN
    outputs). Under this fix a constant-score model reads as exactly random: AUC = 0.5 and
    HR@K = K / num_candidates, consistent with AP.

    :param scores: shape (num_candidates,)
    :param positive_index: index of the ground-truth among candidates (default 0)
    :param ks: K values for @K metrics; default ``TRIPARTITE_RANKING_KS``
    """
    if ks is None:
        ks = TRIPARTITE_RANKING_KS

    scores = np.asarray(scores, dtype=np.float64)
    num_cand = scores.shape[0]
    if num_cand < 2:
        raise ValueError("Need at least 2 candidates for ranking metrics.")

    pos_score = scores[positive_index]
    num_strictly_better = int(np.sum(scores > pos_score))
    num_tied = int(np.sum(scores == pos_score))  # includes the positive itself, so >= 1
    # expected 0-indexed rank (diagnostic): ties share the middle of the block they occupy
    rank = num_strictly_better + (num_tied - 1) / 2.0

    labels = np.zeros(num_cand, dtype=np.int64)
    labels[positive_index] = 1

    try:
        roc = float(roc_auc_score(y_true=labels, y_score=scores))
    except ValueError:
        roc = float("nan")
    try:
        ap = float(average_precision_score(y_true=labels, y_score=scores))
    except ValueError:
        ap = float("nan")

    metrics: dict[str, float] = {"roc_auc": roc, "average_precision": ap}

    idcg1 = 1.0 / np.log2(2.0)  # single relevant item

    # Expected metrics under a uniform random tie-break: the positive is equally likely to
    # occupy any 0-indexed position in [lo, hi). With no ties (num_tied == 1) this collapses
    # to the usual integer-rank metrics.
    lo = num_strictly_better
    hi = num_strictly_better + num_tied  # exclusive
    for k in ks:
        if k <= 0:
            raise ValueError(f"K must be positive, got {k}")
        # fraction of the tie block that falls within top-k = P(positive in top-k)
        in_topk = max(0, min(hi, k) - lo)
        hit = in_topk / num_tied
        metrics[f"recall@{k}"] = float(hit)
        metrics[f"precision@{k}"] = float(hit / k)
        # expected NDCG: average DCG over the tie-block positions that land within top-k
        p_end = min(hi, k)  # exclusive
        if p_end > lo:
            positions = np.arange(lo, p_end)
            dcg_k = float(np.sum(1.0 / np.log2(positions + 2.0)) / num_tied)
        else:
            dcg_k = 0.0
        metrics[f"ndcg@{k}"] = float(dcg_k / idcg1) if idcg1 > 0 else 0.0

    return metrics


def mean_metric_dicts(metric_list: list[dict[str, float]]) -> dict[str, float]:
    """Element-wise mean over dicts with identical keys; skips NaN values per key."""
    if not metric_list:
        return {}
    keys = metric_list[0].keys()
    out: dict[str, float] = {}
    for k in keys:
        vals = [m[k] for m in metric_list if not np.isnan(m[k])]
        out[k] = float(np.mean(vals)) if vals else float("nan")
    return out
