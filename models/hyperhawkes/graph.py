"""Hypergraph construction from user–item interaction timelines (HyperHawkes)."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)


def get_sub_sequences(bipartite_graph, sub_time_delta: float):
    bi_edge_index, _bi_edge_weight, bi_edge_time = bipartite_graph
    offset = 0
    current_u_length = 0
    current_uid = int(bi_edge_index[0, 0].item())
    all_subseqs = []
    n_edges = bi_edge_index.shape[1]
    for col in range(n_edges):
        uid = int(bi_edge_index[0, col].item())
        if uid != current_uid:
            items = bi_edge_index[1, offset : offset + current_u_length]
            user_edge_times = bi_edge_time[offset : offset + current_u_length]
            edge_time_shifted = torch.roll(user_edge_times, 1)
            edge_time_shifted[0] = user_edge_times[0]
            time_delta = user_edge_times - edge_time_shifted
            sub_seqs = (time_delta > sub_time_delta).nonzero(as_tuple=True)[0]
            sub_seqs = torch.tensor_split(items, sub_seqs)
            all_subseqs += [s.tolist() for s in sub_seqs]
            current_uid = uid
            offset += current_u_length
            current_u_length = 1
        else:
            current_u_length += 1
    if current_u_length > 0:
        items = bi_edge_index[1, offset : offset + current_u_length]
        user_edge_times = bi_edge_time[offset : offset + current_u_length]
        edge_time_shifted = torch.roll(user_edge_times, 1)
        edge_time_shifted[0] = user_edge_times[0]
        time_delta = user_edge_times - edge_time_shifted
        sub_seqs = (time_delta > sub_time_delta).nonzero(as_tuple=True)[0]
        sub_seqs = torch.tensor_split(items, sub_seqs)
        all_subseqs += [s.tolist() for s in sub_seqs]
    return all_subseqs


def construct_global_hyper_graph(bipartite_graph, sub_time_delta: float, min_support: float, n_nodes: int):
    try:
        from mlxtend.frequent_patterns import fpmax
        from mlxtend.preprocessing import TransactionEncoder
    except ImportError as e:
        raise ImportError(
            "HyperHawkes hypergraph mining requires mlxtend (pip install mlxtend)."
        ) from e

    all_subseqs = get_sub_sequences(bipartite_graph, sub_time_delta)
    te = TransactionEncoder()
    te_ary = te.fit(all_subseqs).transform(all_subseqs)
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
    logger.info("HyperHawkes: %s user sub-sequences for intent mining", len(all_subseqs))

    frequent_itemsets = fpmax(df_encoded, min_support=min_support, use_colnames=True)
    frequent_itemsets["length"] = frequent_itemsets["itemsets"].apply(lambda x: len(x))
    frequent_itemsets = frequent_itemsets[(frequent_itemsets["length"] >= 2)].reset_index(drop=True)
    logger.info("HyperHawkes: %s frequent itemsets (len>=2)", len(frequent_itemsets))

    nodes, edges, edge_weights = [], [], []
    if len(frequent_itemsets) == 0:
        logger.warning("HyperHawkes: no frequent itemsets; using empty hypergraph stub")
        return torch.tensor([[0, 0], [0, 0]], dtype=torch.long), torch.tensor([1.0], dtype=torch.float)

    max_edge_weight = frequent_itemsets["support"].max()
    min_edge_weight = frequent_itemsets["support"].min()
    for edge_id, row in frequent_itemsets.iterrows():
        items = row["itemsets"]
        edge_support = row["support"]
        edge_weight = 1 + 99 * (edge_support - min_edge_weight) / (max_edge_weight - min_edge_weight + 1e-12)
        nodes.append(torch.LongTensor(list(items)))
        edges.append(torch.LongTensor([edge_id] * len(items)))
        edge_weights.append(torch.FloatTensor([edge_weight] * len(items)))

    edge_index = torch.stack((torch.cat(nodes), torch.cat(edges)))
    edge_weights = torch.cat(edge_weights)
    logger.info("HyperHawkes: %s hyperedges", int(torch.unique(edge_index[1]).numel()))
    return edge_index, edge_weights


def build_bipartite_from_arrays(
    user_locals: np.ndarray,
    item_locals: np.ndarray,
    times: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sort by (user, time) and return (edge_index[2,E], edge_weight[E], edge_time[E])."""
    order = np.lexsort((times, user_locals))
    u = torch.from_numpy(user_locals[order].astype(np.int64))
    it = torch.from_numpy(item_locals[order].astype(np.int64))
    t = torch.from_numpy(times[order].astype(np.float32))
    edge_index = torch.stack([u, it])
    deg = torch.bincount(u, minlength=int(u.max().item()) + 1 if u.numel() else 1).float()
    norm_deg = 1.0 / torch.where(deg == 0, torch.ones(1), deg)
    edge_weight = norm_deg[u]
    return edge_index, edge_weight, t
