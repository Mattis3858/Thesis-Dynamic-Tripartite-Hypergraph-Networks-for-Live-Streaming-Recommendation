"""Meta-path adjacency masks for tripartite HAN (U–S, U–R, S–R views)."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch

from utils.DataLoader import TripartiteData

META_PATH_NAMES = ("user_streamer", "user_room", "streamer_room")


def _meta_path_edges(
    train_data: TripartiteData,
    num_nodes: int,
) -> dict[str, tuple[list[int], list[int]]]:
    """Undirected edges per meta-path type on the global node id space."""
    edges = {name: ([], []) for name in META_PATH_NAMES}
    for u, v, w in zip(
        train_data.user_node_ids,
        train_data.streamer_node_ids,
        train_data.item_node_ids,
    ):
        u, v, w = int(u), int(v), int(w)
        if u <= 0 or v <= 0 or w <= 0:
            continue
        for a, b in ((u, v), (v, u)):
            edges["user_streamer"][0].extend([a, b])
            edges["user_streamer"][1].extend([b, a])
        for a, b in ((u, w), (w, u)):
            edges["user_room"][0].extend([a, b])
            edges["user_room"][1].extend([b, a])
        for a, b in ((v, w), (w, v)):
            edges["streamer_room"][0].extend([a, b])
            edges["streamer_room"][1].extend([b, a])
    return edges


def _adj_to_bias(
    rows: list[int],
    cols: list[int],
    num_nodes: int,
    nhood: int = 1,
) -> torch.Tensor:
    """HAN-style attention bias: 0 = allowed, -1e9 = masked (see HAN utils/process.py)."""
    if not rows:
        mt = np.eye(num_nodes, dtype=np.float32)
    else:
        data = np.ones(len(rows), dtype=np.float32)
        adj = sp.coo_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes)).tocsr()
        mt = np.eye(num_nodes, dtype=np.float32)
        a = adj + sp.eye(num_nodes)
        for _ in range(nhood):
            mt = (mt @ a.toarray()).astype(np.float32)
        mt = (mt > 0).astype(np.float32)

    bias = -1e9 * (1.0 - mt)
    return torch.from_numpy(bias)


def build_meta_path_biases(
    train_data: TripartiteData,
    num_nodes: int,
    nhood: int = 1,
) -> list[torch.Tensor]:
    """One [N, N] bias matrix per meta-path (U–S, U–R, S–R)."""
    edge_dict = _meta_path_edges(train_data, num_nodes)
    return [_adj_to_bias(edge_dict[name][0], edge_dict[name][1], num_nodes, nhood=nhood) for name in META_PATH_NAMES]
