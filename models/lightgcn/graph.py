"""Symmetric normalized adjacency for LightGCN on tripartite hyperedges."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch

from utils.DataLoader import TripartiteData


def build_normalized_adj_from_tripartite(
    train_data: TripartiteData,
    num_nodes: int,
) -> torch.Tensor:
    """
    Build D^{-1/2} A D^{-1/2} from train hyperedges (u, v, w).

    Undirected edges: (u,v), (u,w), (v,w). Node 0 (padding) is excluded.
    """
    rows: list[int] = []
    cols: list[int] = []
    for u, v, w in zip(
        train_data.user_node_ids,
        train_data.streamer_node_ids,
        train_data.item_node_ids,
    ):
        u, v, w = int(u), int(v), int(w)
        if u <= 0 or v <= 0 or w <= 0:
            continue
        for a, b in ((u, v), (v, u), (u, w), (w, u), (v, w), (w, v)):
            rows.append(a)
            cols.append(b)

    if not rows:
        idx = torch.zeros((2, 1), dtype=torch.long)
        val = torch.ones(1, dtype=torch.float32)
        return torch.sparse_coo_tensor(idx, val, (num_nodes, num_nodes)).coalesce()

    data = np.ones(len(rows), dtype=np.float32)
    mat = sp.coo_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes))
    mat = mat.tocsr()
    rowsum = np.array(mat.sum(1)).flatten()
    d_inv_sqrt = np.power(rowsum, -0.5, where=rowsum > 0)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat = sp.diags(d_inv_sqrt)
    norm = d_mat @ mat @ d_mat
    norm = norm.tocoo()
    indices = torch.from_numpy(np.vstack((norm.row, norm.col)).astype(np.int64))
    values = torch.from_numpy(norm.data.astype(np.float32))
    return torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes)).coalesce()
