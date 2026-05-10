from torch.utils.data import Dataset, DataLoader
import importlib.util
import numpy as np
import random
import pandas as pd
from pathlib import Path
from typing import Optional


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _load_preprocess_kuailive_tripartite():
    path = _project_root() / "preprocess_data" / "preprocess_kuailive_tripartite.py"
    spec = importlib.util.spec_from_file_location("preprocess_kuailive_tripartite", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load preprocess module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.preprocess_kuailive_tripartite


def _ensure_tripartite_files(base: Path, dataset_name: str) -> None:
    """
    If ml_{dataset_name}.csv is missing but raw KuaiLive triplets + npy exist in base,
    run preprocess_kuailive_tripartite in-place (writes standard ml_* names).
    """
    main_csv = base / f"ml_{dataset_name}.csv"
    if main_csv.is_file():
        return

    edges = base / "ml_kuailive_edges.csv"
    node_npy = base / "ml_kuailive_node.npy"
    edge_npy = base / "ml_kuailive_features.npy"

    if edges.is_file() and node_npy.is_file() and edge_npy.is_file():
        print(
            f"[TripartiteData] {main_csv.name} not found; building from "
            f"{edges.name}, {node_npy.name}, {edge_npy.name} ..."
        )
        preprocess = _load_preprocess_kuailive_tripartite()
        preprocess(
            edges_path=edges,
            node_features_path=node_npy,
            edge_features_path=edge_npy,
            out_dir=base,
            dataset_name=dataset_name,
        )
        return

    raise FileNotFoundError(
        f"Cannot load tripartite data under {base}.\n"
        f"  Expected: {main_csv.name} (and matching ml_{dataset_name}_node.npy, "
        f"ml_{dataset_name}_node_types.npy, ml_{dataset_name}.npy)\n"
        f"  Or raw bundle in the same folder: ml_kuailive_edges.csv, "
        f"ml_kuailive_node.npy, ml_kuailive_features.npy"
    )


class CustomizedDataset(Dataset):
    def __init__(self, indices_list: list):
        """
        Customized dataset.
        :param indices_list: list, list of indices
        """
        super(CustomizedDataset, self).__init__()

        self.indices_list = indices_list

    def __getitem__(self, idx: int):
        """
        get item at the index in self.indices_list
        :param idx: int, the index
        :return:
        """
        return self.indices_list[idx]

    def __len__(self):
        return len(self.indices_list)


def get_idx_data_loader(indices_list: list, batch_size: int, shuffle: bool):
    """
    get data loader that iterates over indices
    :param indices_list: list, list of indices
    :param batch_size: int, batch size
    :param shuffle: boolean, whether to shuffle the data
    :return: data_loader, DataLoader
    """
    dataset = CustomizedDataset(indices_list=indices_list)

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             drop_last=False)
    return data_loader


class Data:

    def __init__(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray, node_interact_times: np.ndarray, edge_ids: np.ndarray, labels: np.ndarray):
        """
        Data object to store the nodes interaction information.
        :param src_node_ids: ndarray
        :param dst_node_ids: ndarray
        :param node_interact_times: ndarray
        :param edge_ids: ndarray
        :param labels: ndarray
        """
        self.src_node_ids = src_node_ids
        self.dst_node_ids = dst_node_ids
        self.node_interact_times = node_interact_times
        self.edge_ids = edge_ids
        self.labels = labels
        self.num_interactions = len(src_node_ids)
        self.unique_node_ids = set(src_node_ids) | set(dst_node_ids)
        self.num_unique_nodes = len(self.unique_node_ids)


class TripartiteData:
    """
    One row per tripartite hyperedge (User, Streamer, Room) at time ts.
    Node ids use the unified global space (0 = padding unused here).
    """

    def __init__(
        self,
        user_node_ids: np.ndarray,
        streamer_node_ids: np.ndarray,
        item_node_ids: np.ndarray,
        node_interact_times: np.ndarray,
        edge_ids: np.ndarray,
        labels: np.ndarray,
    ):
        self.user_node_ids = user_node_ids
        self.streamer_node_ids = streamer_node_ids
        self.item_node_ids = item_node_ids
        self.node_interact_times = node_interact_times
        self.edge_ids = edge_ids
        self.labels = labels
        self.num_interactions = len(user_node_ids)
        self.unique_node_ids = (
            set(user_node_ids) | set(streamer_node_ids) | set(item_node_ids)
        )
        self.num_unique_nodes = len(self.unique_node_ids)


def _dedup_tripartite_link_targets(data: TripartiteData, split_name: str = "") -> TripartiteData:
    """
    Link-prediction / ranking: one evaluation query per unique (u, streamer, room, t).
    Drops later rows that repeat the same four-tuple (ignores label / edge_idx), preserving
    the first occurrence order. Does NOT affect temporal graph Data used by NeighborSampler.
    """
    u = data.user_node_ids
    s = data.streamer_node_ids
    r = data.item_node_ids
    t = data.node_interact_times
    n_before = len(u)
    seen = set()
    keep: list[int] = []
    for i in range(n_before):
        key = (int(u[i]), int(s[i]), int(r[i]), float(t[i]))
        if key in seen:
            continue
        seen.add(key)
        keep.append(i)
    keep_a = np.asarray(keep, dtype=np.int64)
    n_after = len(keep_a)
    if n_before > n_after and split_name:
        print(
            "Tripartite: link-target dedup [{}] {} -> {} rows (duplicate u,s,r,ts)".format(
                split_name, n_before, n_after
            )
        )
    return TripartiteData(
        user_node_ids=u[keep_a],
        streamer_node_ids=s[keep_a],
        item_node_ids=r[keep_a],
        node_interact_times=t[keep_a],
        edge_ids=data.edge_ids[keep_a],
        labels=data.labels[keep_a],
    )


def get_link_prediction_data(dataset_name: str, val_ratio: float, test_ratio: float):
    """
    generate data for link prediction task (inductive & transductive settings)
    :param dataset_name: str, dataset name
    :param val_ratio: float, validation data ratio
    :param test_ratio: float, test data ratio
    :return: node_raw_features, edge_raw_features, (np.ndarray),
            full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data, (Data object)
    """
    # Load data and train val test split
    graph_df = pd.read_csv('./processed_data/{}/ml_{}.csv'.format(dataset_name, dataset_name))
    edge_raw_features = np.load('./processed_data/{}/ml_{}.npy'.format(dataset_name, dataset_name))
    node_raw_features = np.load('./processed_data/{}/ml_{}_node.npy'.format(dataset_name, dataset_name))

    NODE_FEAT_DIM = EDGE_FEAT_DIM = 172
    assert NODE_FEAT_DIM >= node_raw_features.shape[1], f'Node feature dimension in dataset {dataset_name} is bigger than {NODE_FEAT_DIM}!'
    assert EDGE_FEAT_DIM >= edge_raw_features.shape[1], f'Edge feature dimension in dataset {dataset_name} is bigger than {EDGE_FEAT_DIM}!'
    # padding the features of edges and nodes to the same dimension (172 for all the datasets)
    if node_raw_features.shape[1] < NODE_FEAT_DIM:
        node_zero_padding = np.zeros((node_raw_features.shape[0], NODE_FEAT_DIM - node_raw_features.shape[1]))
        node_raw_features = np.concatenate([node_raw_features, node_zero_padding], axis=1)
    if edge_raw_features.shape[1] < EDGE_FEAT_DIM:
        edge_zero_padding = np.zeros((edge_raw_features.shape[0], EDGE_FEAT_DIM - edge_raw_features.shape[1]))
        edge_raw_features = np.concatenate([edge_raw_features, edge_zero_padding], axis=1)

    assert NODE_FEAT_DIM == node_raw_features.shape[1] and EDGE_FEAT_DIM == edge_raw_features.shape[1], 'Unaligned feature dimensions after feature padding!'

    # get the timestamp of validate and test set
    val_time, test_time = list(np.quantile(graph_df.ts, [(1 - val_ratio - test_ratio), (1 - test_ratio)]))

    src_node_ids = graph_df.u.values.astype(np.longlong)
    dst_node_ids = graph_df.i.values.astype(np.longlong)
    node_interact_times = graph_df.ts.values.astype(np.float64)
    edge_ids = graph_df.idx.values.astype(np.longlong)
    labels = graph_df.label.values

    full_data = Data(src_node_ids=src_node_ids, dst_node_ids=dst_node_ids, node_interact_times=node_interact_times, edge_ids=edge_ids, labels=labels)

    # the setting of seed follows previous works
    random.seed(2020)

    # union to get node set
    node_set = set(src_node_ids) | set(dst_node_ids)
    num_total_unique_node_ids = len(node_set)

    # compute nodes which appear at test time
    test_node_set = set(src_node_ids[node_interact_times > val_time]).union(set(dst_node_ids[node_interact_times > val_time]))
    # sample nodes which we keep as new nodes (to test inductiveness), so then we have to remove all their edges from training
    new_test_node_set = set(random.sample(test_node_set, int(0.1 * num_total_unique_node_ids)))

    # mask for each source and destination to denote whether they are new test nodes
    new_test_source_mask = graph_df.u.map(lambda x: x in new_test_node_set).values
    new_test_destination_mask = graph_df.i.map(lambda x: x in new_test_node_set).values

    # mask, which is true for edges with both destination and source not being new test nodes (because we want to remove all edges involving any new test node)
    observed_edges_mask = np.logical_and(~new_test_source_mask, ~new_test_destination_mask)

    # for train data, we keep edges happening before the validation time which do not involve any new node, used for inductiveness
    train_mask = np.logical_and(node_interact_times <= val_time, observed_edges_mask)

    train_data = Data(src_node_ids=src_node_ids[train_mask], dst_node_ids=dst_node_ids[train_mask],
                      node_interact_times=node_interact_times[train_mask],
                      edge_ids=edge_ids[train_mask], labels=labels[train_mask])

    # define the new nodes sets for testing inductiveness of the model
    train_node_set = set(train_data.src_node_ids).union(train_data.dst_node_ids)
    assert len(train_node_set & new_test_node_set) == 0
    # new nodes that are not in the training set
    new_node_set = node_set - train_node_set

    val_mask = np.logical_and(node_interact_times <= test_time, node_interact_times > val_time)
    test_mask = node_interact_times > test_time

    # new edges with new nodes in the val and test set (for inductive evaluation)
    edge_contains_new_node_mask = np.array([(src_node_id in new_node_set or dst_node_id in new_node_set)
                                            for src_node_id, dst_node_id in zip(src_node_ids, dst_node_ids)])
    new_node_val_mask = np.logical_and(val_mask, edge_contains_new_node_mask)
    new_node_test_mask = np.logical_and(test_mask, edge_contains_new_node_mask)

    # validation and test data
    val_data = Data(src_node_ids=src_node_ids[val_mask], dst_node_ids=dst_node_ids[val_mask],
                    node_interact_times=node_interact_times[val_mask], edge_ids=edge_ids[val_mask], labels=labels[val_mask])

    test_data = Data(src_node_ids=src_node_ids[test_mask], dst_node_ids=dst_node_ids[test_mask],
                     node_interact_times=node_interact_times[test_mask], edge_ids=edge_ids[test_mask], labels=labels[test_mask])

    # validation and test with edges that at least has one new node (not in training set)
    new_node_val_data = Data(src_node_ids=src_node_ids[new_node_val_mask], dst_node_ids=dst_node_ids[new_node_val_mask],
                             node_interact_times=node_interact_times[new_node_val_mask],
                             edge_ids=edge_ids[new_node_val_mask], labels=labels[new_node_val_mask])

    new_node_test_data = Data(src_node_ids=src_node_ids[new_node_test_mask], dst_node_ids=dst_node_ids[new_node_test_mask],
                              node_interact_times=node_interact_times[new_node_test_mask],
                              edge_ids=edge_ids[new_node_test_mask], labels=labels[new_node_test_mask])

    print("The dataset has {} interactions, involving {} different nodes".format(full_data.num_interactions, full_data.num_unique_nodes))
    print("The training dataset has {} interactions, involving {} different nodes".format(
        train_data.num_interactions, train_data.num_unique_nodes))
    print("The validation dataset has {} interactions, involving {} different nodes".format(
        val_data.num_interactions, val_data.num_unique_nodes))
    print("The test dataset has {} interactions, involving {} different nodes".format(
        test_data.num_interactions, test_data.num_unique_nodes))
    print("The new node validation dataset has {} interactions, involving {} different nodes".format(
        new_node_val_data.num_interactions, new_node_val_data.num_unique_nodes))
    print("The new node test dataset has {} interactions, involving {} different nodes".format(
        new_node_test_data.num_interactions, new_node_test_data.num_unique_nodes))
    print("{} nodes were used for the inductive testing, i.e. are never seen during training".format(len(new_test_node_set)))

    return node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data


def get_tripartite_link_prediction_data(
    dataset_name: str,
    val_ratio: float,
    test_ratio: float,
    data_dir: Optional[str] = None,
):
    """
    Load tripartite hyperedges produced by preprocess_kuailive_tripartite.py.

    Paths resolve from the DyGLib repo root (parent of ``utils/``), not from the
    process working directory, unless ``data_dir`` is set.

    Default folder: ``<repo>/processed_data/{dataset_name}/`` with:
      - ml_{dataset_name}.csv  columns: u, streamer, room, ts, label, idx
      - ml_{dataset_name}_node.npy
      - ml_{dataset_name}_node_types.npy
      - ml_{dataset_name}.npy  (edge features; row idx from CSV)
      - ml_{dataset_name}_temporal_edges.npz  (optional; all pairwise edges for sampler)

    If ``ml_{dataset_name}.csv`` is missing but ``ml_kuailive_edges.csv``,
    ``ml_kuailive_node.npy``, and ``ml_kuailive_features.npy`` exist in that folder,
    preprocessing is run once to generate the standard files (id_maps are optional).

    :param data_dir: optional absolute or relative path to the dataset directory
    :return: node_raw_features, edge_raw_features, node_type_ids, full_data, train_data,
        val_data, test_data, new_node_val_data, new_node_test_data, temporal_full_data,
        temporal_train_data. The last two are ``Data`` (pairwise edges in global ids) if
        ``ml_{dataset_name}_temporal_edges.npz`` exists; else ``None``.
    """
    if data_dir is not None:
        base = Path(data_dir).expanduser().resolve()
    else:
        base = _project_root() / "processed_data" / dataset_name

    _ensure_tripartite_files(base, dataset_name)

    graph_df = pd.read_csv(base / f"ml_{dataset_name}.csv")
    edge_raw_features = np.load(str(base / f"ml_{dataset_name}.npy"))
    node_raw_features = np.load(str(base / f"ml_{dataset_name}_node.npy"))
    node_type_ids = np.load(str(base / f"ml_{dataset_name}_node_types.npy"))

    NODE_FEAT_DIM = EDGE_FEAT_DIM = 172
    assert NODE_FEAT_DIM >= node_raw_features.shape[1], (
        f"Node feature dimension in dataset {dataset_name} is bigger than {NODE_FEAT_DIM}!"
    )
    assert EDGE_FEAT_DIM >= edge_raw_features.shape[1], (
        f"Edge feature dimension in dataset {dataset_name} is bigger than {EDGE_FEAT_DIM}!"
    )
    if node_raw_features.shape[1] < NODE_FEAT_DIM:
        node_zero_padding = np.zeros(
            (node_raw_features.shape[0], NODE_FEAT_DIM - node_raw_features.shape[1])
        )
        node_raw_features = np.concatenate([node_raw_features, node_zero_padding], axis=1)
    if edge_raw_features.shape[1] < EDGE_FEAT_DIM:
        edge_zero_padding = np.zeros(
            (edge_raw_features.shape[0], EDGE_FEAT_DIM - edge_raw_features.shape[1])
        )
        edge_raw_features = np.concatenate([edge_raw_features, edge_zero_padding], axis=1)

    assert NODE_FEAT_DIM == node_raw_features.shape[1] and EDGE_FEAT_DIM == edge_raw_features.shape[1], (
        "Unaligned feature dimensions after feature padding!"
    )

    val_time, test_time = list(
        np.quantile(graph_df.ts, [(1 - val_ratio - test_ratio), (1 - test_ratio)])
    )

    user_node_ids = graph_df.u.values.astype(np.longlong)
    streamer_node_ids = graph_df.streamer.values.astype(np.longlong)
    item_node_ids = graph_df.room.values.astype(np.longlong)
    node_interact_times = graph_df.ts.values.astype(np.float64)
    edge_ids = graph_df.idx.values.astype(np.longlong)
    labels = graph_df.label.values

    full_data = TripartiteData(
        user_node_ids=user_node_ids,
        streamer_node_ids=streamer_node_ids,
        item_node_ids=item_node_ids,
        node_interact_times=node_interact_times,
        edge_ids=edge_ids,
        labels=labels,
    )

    random.seed(2020)

    node_set = full_data.unique_node_ids
    num_total_unique_node_ids = len(node_set)

    past_val_mask = node_interact_times > val_time
    test_node_set = (
        set(user_node_ids[past_val_mask])
        | set(streamer_node_ids[past_val_mask])
        | set(item_node_ids[past_val_mask])
    )
    test_node_list = list(test_node_set)
    k_new = int(0.1 * num_total_unique_node_ids)
    if k_new > len(test_node_list):
        k_new = len(test_node_list)
    new_test_node_set = set(random.sample(test_node_list, k_new)) if k_new > 0 else set()

    def hyperedge_touches_holdout(u_arr, s_arr, r_arr) -> np.ndarray:
        return np.array(
            [
                (int(u) in new_test_node_set
                 or int(s) in new_test_node_set
                 or int(r) in new_test_node_set)
                for u, s, r in zip(u_arr, s_arr, r_arr)
            ]
        )

    new_test_hyper_mask = hyperedge_touches_holdout(user_node_ids, streamer_node_ids, item_node_ids)
    observed_edges_mask = ~new_test_hyper_mask

    train_mask = np.logical_and(node_interact_times <= val_time, observed_edges_mask)

    train_data = TripartiteData(
        user_node_ids=user_node_ids[train_mask],
        streamer_node_ids=streamer_node_ids[train_mask],
        item_node_ids=item_node_ids[train_mask],
        node_interact_times=node_interact_times[train_mask],
        edge_ids=edge_ids[train_mask],
        labels=labels[train_mask],
    )

    train_node_set = train_data.unique_node_ids
    assert len(train_node_set & new_test_node_set) == 0
    new_node_set = node_set - train_node_set

    val_mask = np.logical_and(node_interact_times <= test_time, node_interact_times > val_time)
    test_mask = node_interact_times > test_time

    edge_contains_new_node_mask = np.array(
        [
            (u in new_node_set or s in new_node_set or r in new_node_set)
            for u, s, r in zip(user_node_ids, streamer_node_ids, item_node_ids)
        ]
    )
    new_node_val_mask = np.logical_and(val_mask, edge_contains_new_node_mask)
    new_node_test_mask = np.logical_and(test_mask, edge_contains_new_node_mask)

    val_data = TripartiteData(
        user_node_ids=user_node_ids[val_mask],
        streamer_node_ids=streamer_node_ids[val_mask],
        item_node_ids=item_node_ids[val_mask],
        node_interact_times=node_interact_times[val_mask],
        edge_ids=edge_ids[val_mask],
        labels=labels[val_mask],
    )
    test_data = TripartiteData(
        user_node_ids=user_node_ids[test_mask],
        streamer_node_ids=streamer_node_ids[test_mask],
        item_node_ids=item_node_ids[test_mask],
        node_interact_times=node_interact_times[test_mask],
        edge_ids=edge_ids[test_mask],
        labels=labels[test_mask],
    )
    new_node_val_data = TripartiteData(
        user_node_ids=user_node_ids[new_node_val_mask],
        streamer_node_ids=streamer_node_ids[new_node_val_mask],
        item_node_ids=item_node_ids[new_node_val_mask],
        node_interact_times=node_interact_times[new_node_val_mask],
        edge_ids=edge_ids[new_node_val_mask],
        labels=labels[new_node_val_mask],
    )
    new_node_test_data = TripartiteData(
        user_node_ids=user_node_ids[new_node_test_mask],
        streamer_node_ids=streamer_node_ids[new_node_test_mask],
        item_node_ids=item_node_ids[new_node_test_mask],
        node_interact_times=node_interact_times[new_node_test_mask],
        edge_ids=edge_ids[new_node_test_mask],
        labels=labels[new_node_test_mask],
    )

    train_data = _dedup_tripartite_link_targets(train_data, "train")
    val_data = _dedup_tripartite_link_targets(val_data, "val")
    test_data = _dedup_tripartite_link_targets(test_data, "test")
    new_node_val_data = _dedup_tripartite_link_targets(new_node_val_data, "new_node_val")
    new_node_test_data = _dedup_tripartite_link_targets(new_node_test_data, "new_node_test")

    print(
        "Tripartite: The dataset has {} hyperedges, involving {} different nodes".format(
            full_data.num_interactions, full_data.num_unique_nodes
        )
    )
    print(
        "Tripartite: training {} hyperedges, {} nodes; val {}; test {}".format(
            train_data.num_interactions,
            train_data.num_unique_nodes,
            val_data.num_interactions,
            test_data.num_interactions,
        )
    )
    print(
        "{} nodes held out for inductive testing (never in training)".format(
            len(new_test_node_set)
        )
    )

    temporal_npz = base / f"ml_{dataset_name}_temporal_edges.npz"
    temporal_full_data = None
    temporal_train_data = None
    if temporal_npz.is_file():
        z = np.load(str(temporal_npz))
        te_src = z["src"].astype(np.longlong)
        te_dst = z["dst"].astype(np.longlong)
        te_eid = z["edge_idx"].astype(np.longlong)
        te_ts = z["ts"].astype(np.float64)
        n_te = len(te_src)
        te_lab = np.zeros(n_te, dtype=np.float64)
        temporal_full_data = Data(
            src_node_ids=te_src,
            dst_node_ids=te_dst,
            node_interact_times=te_ts,
            edge_ids=te_eid,
            labels=te_lab,
        )
        holdout = np.array(list(new_test_node_set), dtype=np.int64)
        if len(holdout) == 0:
            tr_m = te_ts <= val_time
        else:
            bad_s = np.isin(te_src, holdout)
            bad_d = np.isin(te_dst, holdout)
            tr_m = (te_ts <= val_time) & (~bad_s) & (~bad_d)
        temporal_train_data = Data(
            src_node_ids=te_src[tr_m],
            dst_node_ids=te_dst[tr_m],
            node_interact_times=te_ts[tr_m],
            edge_ids=te_eid[tr_m],
            labels=te_lab[tr_m],
        )
        print(
            "Tripartite: temporal graph {} edges (full), {} edges (train period, inductive-filtered)".format(
                n_te, int(tr_m.sum())
            )
        )
    else:
        print(
            "Tripartite: {} not found — use hyperedge-only neighbor graph or rerun preprocess.".format(
                temporal_npz.name
            )
        )

    return (
        node_raw_features,
        edge_raw_features,
        node_type_ids,
        full_data,
        train_data,
        val_data,
        test_data,
        new_node_val_data,
        new_node_test_data,
        temporal_full_data,
        temporal_train_data,
    )


def get_node_classification_data(dataset_name: str, val_ratio: float, test_ratio: float):
    """
    generate data for node classification task
    :param dataset_name: str, dataset name
    :param val_ratio: float, validation data ratio
    :param test_ratio: float, test data ratio
    :return: node_raw_features, edge_raw_features, (np.ndarray),
            full_data, train_data, val_data, test_data, (Data object)
    """
    # Load data and train val test split
    graph_df = pd.read_csv('./processed_data/{}/ml_{}.csv'.format(dataset_name, dataset_name))
    edge_raw_features = np.load('./processed_data/{}/ml_{}.npy'.format(dataset_name, dataset_name))
    node_raw_features = np.load('./processed_data/{}/ml_{}_node.npy'.format(dataset_name, dataset_name))

    NODE_FEAT_DIM = EDGE_FEAT_DIM = 172
    assert NODE_FEAT_DIM >= node_raw_features.shape[1], f'Node feature dimension in dataset {dataset_name} is bigger than {NODE_FEAT_DIM}!'
    assert EDGE_FEAT_DIM >= edge_raw_features.shape[1], f'Edge feature dimension in dataset {dataset_name} is bigger than {EDGE_FEAT_DIM}!'
    # padding the features of edges and nodes to the same dimension (172 for all the datasets)
    if node_raw_features.shape[1] < NODE_FEAT_DIM:
        node_zero_padding = np.zeros((node_raw_features.shape[0], NODE_FEAT_DIM - node_raw_features.shape[1]))
        node_raw_features = np.concatenate([node_raw_features, node_zero_padding], axis=1)
    if edge_raw_features.shape[1] < EDGE_FEAT_DIM:
        edge_zero_padding = np.zeros((edge_raw_features.shape[0], EDGE_FEAT_DIM - edge_raw_features.shape[1]))
        edge_raw_features = np.concatenate([edge_raw_features, edge_zero_padding], axis=1)

    assert NODE_FEAT_DIM == node_raw_features.shape[1] and EDGE_FEAT_DIM == edge_raw_features.shape[1], 'Unaligned feature dimensions after feature padding!'

    # get the timestamp of validate and test set
    val_time, test_time = list(np.quantile(graph_df.ts, [(1 - val_ratio - test_ratio), (1 - test_ratio)]))

    src_node_ids = graph_df.u.values.astype(np.longlong)
    dst_node_ids = graph_df.i.values.astype(np.longlong)
    node_interact_times = graph_df.ts.values.astype(np.float64)
    edge_ids = graph_df.idx.values.astype(np.longlong)
    labels = graph_df.label.values

    # The setting of seed follows previous works
    random.seed(2020)

    train_mask = node_interact_times <= val_time
    val_mask = np.logical_and(node_interact_times <= test_time, node_interact_times > val_time)
    test_mask = node_interact_times > test_time

    full_data = Data(src_node_ids=src_node_ids, dst_node_ids=dst_node_ids, node_interact_times=node_interact_times, edge_ids=edge_ids, labels=labels)
    train_data = Data(src_node_ids=src_node_ids[train_mask], dst_node_ids=dst_node_ids[train_mask],
                      node_interact_times=node_interact_times[train_mask],
                      edge_ids=edge_ids[train_mask], labels=labels[train_mask])
    val_data = Data(src_node_ids=src_node_ids[val_mask], dst_node_ids=dst_node_ids[val_mask],
                    node_interact_times=node_interact_times[val_mask], edge_ids=edge_ids[val_mask], labels=labels[val_mask])
    test_data = Data(src_node_ids=src_node_ids[test_mask], dst_node_ids=dst_node_ids[test_mask],
                     node_interact_times=node_interact_times[test_mask], edge_ids=edge_ids[test_mask], labels=labels[test_mask])

    return node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data
