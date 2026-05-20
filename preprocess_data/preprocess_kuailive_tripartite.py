"""
KuaiLive → Tripartite hypergraph preprocessing for HT-Transformer.

語意（與欄位無關；**不靠 label 辨識角色**）：
- **etype** 決定這條有向邊在「同一場直播事件」裡的角色：
  - etype=0：User → Streamer（src=使用者, dst=直播主）
  - etype=1：User → Room / Item（src=使用者, dst=直播間）
  - etype=2：Room → Streamer（src=直播間, dst=直播主）
- **同一時間戳 ts** 上若三層邊都存在且可閉合成 (u, streamer, room)，即為一筆**三方超邊**
  （對應「同一時間使用者在某直播主的某直播間」的同時發生的三種關係）。
- **label**：互動類型（如 click=0, comment=1, like=2, negative=3）。**不同 label 視為不同邊**，
  去重僅在 (ts, etype, src, dst, label) 完全相同時保留 idx 最小列；所有保留邊都會寫入
  ``ml_*_temporal_edges.npz`` 供 NeighborSampler 建圖。

ID 空間：
- 將 User / Streamer / Room 合併為連續整數：0=PAD, Users 1..N_u,
  Streamers N_u+1..N_u+N_s, Rooms N_u+N_s+1..（與 node_type_ids 對齊）。
- 目前假設邊表中的 src/dst 與 ml_kuailive_node.npy 的列索引一致（DyGLib 慣例）。
  id_maps_*.csv 若為「原始業務 ID」則與本檔內已重編號之 id 不同，需先在資料端對齊後再餵入。

canonical edge_idx = etype==0 該列的 idx，對齊 ml_*_features.npy 列號。

Usage (from repo root):
  python preprocess_data/preprocess_kuailive_tripartite.py \\
    --edges ml_kuailive_edges.csv \\
    --node_features ml_kuailive_node.npy \\
    --edge_features ml_kuailive_features.npy \\
    --out_dir processed_data/kuailive_tripartite \\
    --dataset_name kuailive_tripartite
"""

from __future__ import annotations

import argparse
import json
from itertools import chain
from pathlib import Path

import numpy as np
import pandas as pd

# Type ids (must match HTTransformer / user spec)
TYPE_PAD = 0
TYPE_USER = 1
TYPE_STREAMER = 2
TYPE_ROOM = 3


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={c: c.strip().lower() for c in df.columns})
    for col in ("src", "dst", "ts", "label", "etype", "idx"):
        if col not in df.columns:
            raise ValueError(f"Missing column {col}; got {list(df.columns)}")
    return df


def _dedupe_labeled_edges(df: pd.DataFrame) -> pd.DataFrame:
    """
    僅當 (ts, etype, src, dst, label) 完全相同時視為重複，保留 idx 最小列。
    不同互動類型（label）的邊全部保留。
    """
    df = _normalize_columns(df.copy())
    df["src"] = df["src"].astype(np.int64)
    df["dst"] = df["dst"].astype(np.int64)
    df["etype"] = df["etype"].astype(np.int64)
    df["idx"] = df["idx"].astype(np.int64)
    df["ts"] = df["ts"].astype(np.float64)
    df["label"] = df["label"].astype(np.float64)

    df = df.sort_values(
        ["ts", "etype", "src", "dst", "label", "idx"],
        ascending=[True, True, True, True, True, True],
    )
    before = len(df)
    df = df.drop_duplicates(subset=["ts", "etype", "src", "dst", "label"], keep="first")
    deduped = before - len(df)
    if deduped > 0:
        print(
            f"[preprocess_kuailive_tripartite] Dropped {deduped} duplicate rows "
            f"(identical ts,etype,src,dst,label)."
        )
    return df.reset_index(drop=True)


def _build_global_id_maps_from_edge_table(df: pd.DataFrame) -> tuple[dict, dict, dict, int, int, int]:
    """由**完整**去重後邊表推斷所有出現過的 user / streamer / room raw id，再配置 global id。"""
    users = set(df.loc[df["etype"] == 0, "src"]).union(set(df.loc[df["etype"] == 1, "src"]))
    streamers = set(df.loc[df["etype"] == 0, "dst"]).union(set(df.loc[df["etype"] == 2, "dst"]))
    rooms = set(df.loc[df["etype"] == 1, "dst"]).union(set(df.loc[df["etype"] == 2, "src"]))
    users, streamers, rooms = sorted(users), sorted(streamers), sorted(rooms)
    nu, ns, nr = len(users), len(streamers), len(rooms)

    user_map = {oid: 1 + i for i, oid in enumerate(users)}
    streamer_map = {oid: 1 + nu + i for i, oid in enumerate(streamers)}
    room_map = {oid: 1 + nu + ns + i for i, oid in enumerate(rooms)}
    return user_map, streamer_map, room_map, nu, ns, nr


def _raw_edge_to_global(
    src: int,
    dst: int,
    etype: int,
    user_map: dict,
    streamer_map: dict,
    room_map: dict,
) -> tuple[int, int]:
    e = int(etype)
    s, d = int(src), int(dst)
    if e == 0:
        return user_map[s], streamer_map[d]
    if e == 1:
        return user_map[s], room_map[d]
    if e == 2:
        return room_map[s], streamer_map[d]
    raise ValueError(f"Unknown etype {e}")


def _assemble_hyperedges(df: pd.DataFrame) -> pd.DataFrame:
    """由已去重邊表組 (u, streamer, room, ts, label, idx)；label 取 etype==0 該列。"""
    rows = []
    skipped_ts = 0
    for ts, g in df.groupby("ts", sort=True):
        g0 = g[g["etype"] == 0]
        g1 = g[g["etype"] == 1]
        g2 = g[g["etype"] == 2]
        if g0.empty or g1.empty or g2.empty:
            skipped_ts += 1
            continue

        for _, r0 in g0.iterrows():
            u = int(r0["src"])
            streamer = int(r0["dst"])
            r1_cands = g1[g1["src"] == u]
            if r1_cands.empty:
                continue
            matched = False
            for _, r1 in r1_cands.iterrows():
                room = int(r1["dst"])
                r2_match = g2[(g2["src"] == room) & (g2["dst"] == streamer)]
                if r2_match.empty:
                    continue
                label = float(r0["label"])
                canonical_idx = int(r0["idx"])
                rows.append(
                    {
                        "u_raw": u,
                        "streamer_raw": streamer,
                        "room_raw": room,
                        "ts": float(ts),
                        "label": label,
                        "idx": canonical_idx,
                    }
                )
                matched = True
                break
            if not matched:
                continue

    if skipped_ts > 0:
        print(
            f"[preprocess_kuailive_tripartite] Skipped {skipped_ts} timestamps with incomplete etype layers "
            "(missing 0, 1, or 2)."
        )

    if not rows:
        raise ValueError("No valid tripartite hyperedges parsed; check ml_kuailive_edges.csv format.")

    hyper_df = pd.DataFrame(rows)
    hyper_df = hyper_df.sort_values(["ts", "idx"], ascending=True).reset_index(drop=True)
    return hyper_df


def _build_node_type_ids(max_node_id: int, user_map: dict, streamer_map: dict, room_map: dict) -> np.ndarray:
    """shape (max_node_id + 1,), padding index 0 = TYPE_PAD."""
    node_type_ids = np.zeros(max_node_id + 1, dtype=np.int64)
    for _old, gid in user_map.items():
        node_type_ids[gid] = TYPE_USER
    for _old, gid in streamer_map.items():
        node_type_ids[gid] = TYPE_STREAMER
    for _old, gid in room_map.items():
        node_type_ids[gid] = TYPE_ROOM
    return node_type_ids


def _remap_node_features(
    old_node_feats: np.ndarray,
    user_map: dict,
    streamer_map: dict,
    room_map: dict,
) -> np.ndarray:
    """
    old_node_feats[row_i] corresponds to raw id i used in ml_kuailive_edges.csv
    (DyGLib convention: index == node id in edge list).
    """
    max_gid = max(chain(user_map.values(), streamer_map.values(), room_map.values()), default=0)
    feat_dim = old_node_feats.shape[1]
    new_feats = np.zeros((max_gid + 1, feat_dim), dtype=old_node_feats.dtype)

    def copy_block(raw_to_global: dict):
        for raw_id, gid in raw_to_global.items():
            if raw_id < 0 or raw_id >= old_node_feats.shape[0]:
                raise IndexError(
                    f"Raw node id {raw_id} out of range for node matrix rows [0, {old_node_feats.shape[0]})"
                )
            new_feats[gid] = old_node_feats[raw_id]

    copy_block(user_map)
    copy_block(streamer_map)
    copy_block(room_map)
    return new_feats


def preprocess_kuailive_tripartite(
    edges_path: Path,
    node_features_path: Path,
    edge_features_path: Path,
    out_dir: Path,
    dataset_name: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    df_edges = pd.read_csv(edges_path)
    df = _dedupe_labeled_edges(df_edges)
    user_map, streamer_map, room_map, nu, ns, nr = _build_global_id_maps_from_edge_table(df)
    hyper_df = _assemble_hyperedges(df)

    hyper_df["u"] = hyper_df["u_raw"].map(user_map)
    hyper_df["streamer"] = hyper_df["streamer_raw"].map(streamer_map)
    hyper_df["room"] = hyper_df["room_raw"].map(room_map)

    max_gid = 1 + nu + ns + nr - 1
    node_type_ids = _build_node_type_ids(max_gid, user_map, streamer_map, room_map)

    old_node_feats = np.load(node_features_path)
    new_node_feats = _remap_node_features(old_node_feats, user_map, streamer_map, room_map)

    edge_raw = np.load(edge_features_path)
    max_idx = int(df["idx"].max())
    if max_idx >= edge_raw.shape[0]:
        raise IndexError(
            f"canonical edge idx {max_idx} >= edge feature rows {edge_raw.shape[0]}; "
            "ensure ml_kuailive_features.npy includes a zero row at index 0 if idx starts at 1."
        )

    out_csv = out_dir / f"ml_{dataset_name}.csv"
    out_node = out_dir / f"ml_{dataset_name}_node.npy"
    out_node_types = out_dir / f"ml_{dataset_name}_node_types.npy"
    out_edge = out_dir / f"ml_{dataset_name}.npy"
    out_temporal = out_dir / f"ml_{dataset_name}_temporal_edges.npz"
    out_meta = out_dir / f"{dataset_name}_tripartite_meta.json"

    g_src, g_dst, g_eid, g_ts = [], [], [], []
    for _, row in df.iterrows():
        a, b = _raw_edge_to_global(
            int(row["src"]),
            int(row["dst"]),
            int(row["etype"]),
            user_map,
            streamer_map,
            room_map,
        )
        g_src.append(a)
        g_dst.append(b)
        g_eid.append(int(row["idx"]))
        g_ts.append(float(row["ts"]))
    np.savez(
        out_temporal,
        src=np.asarray(g_src, dtype=np.int64),
        dst=np.asarray(g_dst, dtype=np.int64),
        edge_idx=np.asarray(g_eid, dtype=np.int64),
        ts=np.asarray(g_ts, dtype=np.float64),
    )

    hyper_out = hyper_df[["u", "streamer", "room", "ts", "label", "idx"]].copy()
    hyper_out.to_csv(out_csv, index=False)
    np.save(out_node, new_node_feats)
    np.save(out_node_types, node_type_ids)
    np.save(out_edge, edge_raw)

    meta = {
        "dataset_name": dataset_name,
        "num_users": nu,
        "num_streamers": ns,
        "num_rooms": nr,
        "num_global_nodes": int(max_gid),
        "num_hyperedges": len(hyper_out),
        "id_layout": "0=PAD; users 1..Nu; streamers Nu+1..Nu+Ns; rooms Nu+Ns+1..Nu+Ns+Nr",
        "node_type_ids_encoding": {
            "0": "padding",
            "1": "user",
            "2": "streamer",
            "3": "room",
        },
        "canonical_edge_idx": "idx column = row index into ml_{dataset_name}.npy (etype 0 row of each triplet)",
        "temporal_edges_npz": str(out_temporal.name),
        "num_temporal_pairwise_edges": len(g_src),
        "interaction_labels": {"click": 0, "comment": 1, "like": 2, "negative": 3},
        "source_edges_csv": str(edges_path.resolve()),
    }
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Wrote {out_csv} ({len(hyper_out)} hyperedges)")
    print(f"Wrote {out_node} shape {new_node_feats.shape}")
    print(f"Wrote {out_node_types} len {len(node_type_ids)}")
    print(f"Wrote {out_edge} shape {edge_raw.shape}")
    print(f"Wrote {out_temporal} ({len(g_src)} undirected pairs as directed rows; sampler doubles endpoints)")
    print(f"Nu={nu}, Ns={ns}, Nr={nr}, max global id={max_gid}")
    print(f"Meta: {out_meta}")


def main():
    p = argparse.ArgumentParser(description="Preprocess KuaiLive CSV+NPY for tripartite HT-Transformer.")
    p.add_argument("--edges", type=Path, default=Path("kuailive_data/ml_kuailive_edges.csv"))
    p.add_argument("--node_features", type=Path, default=Path("kuailive_data/ml_kuailive_node.npy"))
    p.add_argument("--edge_features", type=Path, default=Path("kuailive_data/ml_kuailive_features.npy"))
    p.add_argument("--out_dir", type=Path, default=Path("processed_data/kuailive_tripartite"))
    p.add_argument("--dataset_name", type=str, default="kuailive_tripartite")
    args = p.parse_args()
    preprocess_kuailive_tripartite(
        edges_path=args.edges,
        node_features_path=args.node_features,
        edge_features_path=args.edge_features,
        out_dir=args.out_dir,
        dataset_name=args.dataset_name,
    )


if __name__ == "__main__":
    main()
