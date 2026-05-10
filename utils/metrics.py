"""
Metrics for DyGLib-style tasks and tripartite ranking evaluation.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

# 1 ground-truth + 99 negatives => 100 candidates; K must be <= num_candidates for meaningful @K
TRIPARTITE_RANKING_KS: tuple[int, ...] = (5, 10, 20, 50, 100)


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

    order = np.argsort(-scores)
    rank = int(np.where(order == positive_index)[0][0])

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

    idcg1 = 1.0 / np.log2(2.0)

    for k in ks:
        if k <= 0:
            raise ValueError(f"K must be positive, got {k}")
        metrics[f"recall@{k}"] = 1.0 if rank < k else 0.0
        metrics[f"precision@{k}"] = (1.0 / k) if rank < k else 0.0
        if rank < k:
            dcg_k = 1.0 / np.log2(rank + 2.0)
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
