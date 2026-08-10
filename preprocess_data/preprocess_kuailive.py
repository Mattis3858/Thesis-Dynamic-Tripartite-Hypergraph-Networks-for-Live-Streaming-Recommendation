"""
Preprocess KuaiLive data into TGN-style tripartite graph files.

Inputs (under `data/KuaiLive`):
    - click.csv    : user_id, live_id, streamer_id, timestamp[, ...]
    - comment.csv  : same columns as above
    - like.csv     : same columns as above
    - negative.csv : same columns + extra fields (ignored)

Each row is treated as a (user, streamer, room, ts, interaction_type) 事件，其中：
    - user  節點類型：user
    - streamer 節點類型：streamer
    - room = live_id 節點類型：room

We construct 3 種二元關係 (三方圖展開成三條邊)：
    etype = 0: user  -> room
    etype = 1: user  -> streamer
    etype = 2: streamer -> room

另外再用 label 編碼互動型態：
    label = 0: click
    label = 1: comment
    label = 2: like
    label = 3: negative

Outputs (under `training_data`):
    - ml_kuailive_edges.csv       (src, dst, ts, label, etype, idx)
    - ml_kuailive_features.npy    (edge features; 3 維 one-hot 對應 etype)
    - ml_kuailive_node.npy        (node features; zeros [n_nodes+1 x 172])
    - id_maps_kuailive_users.csv      (原始 user_id -> 連續 id)
    - id_maps_kuailive_streamers.csv  (原始 streamer_id -> 連續 id)
    - id_maps_kuailive_rooms.csv      (原始 room_id(live_id) -> 連續 id)

這個格式基本上沿用 `preprocess_data.py` 產生的 `ml_tripartite_*` 檔案，只是把實體類型
從 (user, streamer, item) 換成 (user, streamer, room)，並且用 label 加上互動型態。
"""

import os
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# ---------- 路徑設定 ----------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "KuaiLive"
OUT_DIR = ROOT / "training_data"

OUT_EDGES = OUT_DIR / "ml_kuailive_edges.csv"
OUT_EFEAT = OUT_DIR / "ml_kuailive_features.npy"
OUT_NFEAT = OUT_DIR / "ml_kuailive_node.npy"
OUT_USERS = OUT_DIR / "id_maps_kuailive_users.csv"
OUT_STREAMERS = OUT_DIR / "id_maps_kuailive_streamers.csv"
OUT_ROOMS = OUT_DIR / "id_maps_kuailive_rooms.csv"


INTERACTION_FILES = {
    "click": DATA_DIR / "click.csv",
    "comment": DATA_DIR / "comment.csv",
    "like": DATA_DIR / "like.csv",
    "negative": DATA_DIR / "negative.csv",
}

INTERACTION_LABEL = {
    "click": 0,
    "comment": 1,
    "like": 2,
    "negative": 3,
}


def sample_streamers(counts: pd.Series, n: int, how: str, seed: int) -> pd.Index:
    """
    挑選要保留的 streamer。

    top        : 互動量前 N（原本的行為，偏向頭部）
    stratified : 依互動量分成 N 層、每層抽一個，涵蓋長尾（描述性分析建議用這個）
    random     : 均勻隨機抽
    """
    if how == "top":
        return counts.head(n).index

    rng = np.random.RandomState(seed)
    ordered = counts.sort_values(ascending=False)
    if how == "random":
        pick = rng.choice(len(ordered), size=min(n, len(ordered)), replace=False)
        return ordered.index[np.sort(pick)]
    if how == "stratified":
        bins = np.array_split(np.arange(len(ordered)), min(n, len(ordered)))
        pick = [int(rng.choice(b)) for b in bins if len(b) > 0]
        return ordered.index[np.array(sorted(pick))]
    raise ValueError(f"Unknown streamer sampling strategy: {how}")


def load_interactions(
    max_streamers: int | None = None,
    max_users_per_streamer: int | None = None,
    streamer_sampling: str = "top",
    seed: int = 2020,
) -> pd.DataFrame:
    """
    讀取四種互動檔案，合併成一個 DataFrame，欄位至少包含：
        user_id, live_id, streamer_id, timestamp, interaction_type, label
    """
    dfs = []
    for itype, path in INTERACTION_FILES.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing KuaiLive interaction file: {path}")

        df = pd.read_csv(path)
        cols_lower = {c.lower(): c for c in df.columns}

        # 對應必要欄位名稱（大小寫容錯）
        user_col = cols_lower.get("user_id")
        live_col = cols_lower.get("live_id")
        streamer_col = cols_lower.get("streamer_id")
        ts_col = cols_lower.get("timestamp")

        missing = [
            name
            for name, col in [
                ("user_id", user_col),
                ("live_id", live_col),
                ("streamer_id", streamer_col),
                ("timestamp", ts_col),
            ]
            if col is None
        ]
        if missing:
            raise ValueError(f"{path} 缺少必要欄位: {missing}, 目前欄位: {list(df.columns)}")

        sub = df[[user_col, live_col, streamer_col, ts_col]].copy()
        sub.columns = ["user_id", "room_id", "streamer_id", "timestamp"]

        # timestamp 轉成 float (若是整數毫秒就直接用；若有缺失則填補)
        if not np.issubdtype(sub["timestamp"].dtype, np.number):
            sub["timestamp"] = pd.to_numeric(sub["timestamp"], errors="coerce")
        if sub["timestamp"].isna().any():
            # 用行索引填補缺失，至少保證嚴格遞增
            sub["timestamp"] = sub["timestamp"].fillna(
                np.arange(len(sub), dtype=float)
            )

        sub["interaction_type"] = itype
        sub["label"] = INTERACTION_LABEL[itype]
        dfs.append(sub)

    all_df = pd.concat(dfs, ignore_index=True)

    # 如果有指定最多保留幾個 streamer，這裡先做子集選取
    # 策略：保留「互動次數最多」的前 max_streamers 位 streamer，其餘全部丟掉
    if max_streamers is not None:
        vc = all_df["streamer_id"].value_counts()
        keep_ids = sample_streamers(vc, max_streamers, streamer_sampling, seed)
        all_df = all_df[all_df["streamer_id"].isin(keep_ids)].reset_index(drop=True)
        print(
            f"[streamer sampling] strategy={streamer_sampling}, kept {len(keep_ids)} of {len(vc)} streamers"
        )

    # 如果有指定每個 streamer 最多保留多少 user，
    # 策略：對每個 streamer，保留「互動次數最多」的前 max_users_per_streamer 個 user，
    #       這些 user 的所有互動都保留，其餘 user 全部丟掉。
    if max_users_per_streamer is not None:
        print(
            "[!! WARNING] --max_users_per_streamer truncates every streamer to the same number of "
            "top viewers. Audience-side concentration statistics (audience HHI / entropy / repeat "
            "ratio) then measure the SAMPLING RULE, not real behaviour -- every streamer ends up "
            "with exactly K viewers. Do NOT use such a slice for analyze_streamer_bias.py; it is "
            "only appropriate for building a small training subset."
        )

        def _limit_users_per_streamer(df: pd.DataFrame) -> pd.DataFrame:
            user_counts = df["user_id"].value_counts()
            keep_users = user_counts.head(max_users_per_streamer).index
            return df[df["user_id"].isin(keep_users)]

        all_df = (
            all_df.groupby("streamer_id", group_keys=False)
            .apply(_limit_users_per_streamer)
            .reset_index(drop=True)
        )

    # 依 timestamp 排序，確保時間順序
    all_df = all_df.sort_values("timestamp").reset_index(drop=True)
    return all_df


def build_id_map(series: pd.Series, name: str) -> pd.DataFrame:
    """建立某一實體類型的原始 ID -> 連續整數 id 對照表。"""
    uniq = pd.Index(series.unique())
    mapping = pd.DataFrame(
        {name: uniq, f"{name}_cid": np.arange(len(uniq), dtype=int)}
    )
    return mapping


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    parser = argparse.ArgumentParser(
        description="Preprocess KuaiLive into ml_kuailive_* tripartite graph files"
    )
    parser.add_argument(
        "--max_streamers",
        type=int,
        default=None,
        help="僅保留互動次數最多的前 N 位 streamer；預設 None 表示不限制",
    )
    parser.add_argument(
        "--max_users_per_streamer",
        type=int,
        default=None,
        help="每個 streamer 只保留互動次數最多的前 K 位 user；預設 None 表示不限制。"
        "注意：設了這個就不能拿來做 audience 端的集中度分析（見程式內警告）",
    )
    parser.add_argument(
        "--streamer_sampling",
        type=str,
        default="top",
        choices=["top", "stratified", "random"],
        help="搭配 --max_streamers 使用；top=互動量前 N（預設，偏頭部），"
        "stratified=依互動量分層抽（涵蓋長尾，描述性分析建議），random=均勻隨機",
    )
    parser.add_argument("--seed", type=int, default=2020, help="streamer 抽樣的亂數種子")
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help=f"輸出資料夾；預設 {OUT_DIR}。分析用的大切片建議另外指定，避免蓋掉訓練資料",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_edges = out_dir / OUT_EDGES.name
    out_efeat = out_dir / OUT_EFEAT.name
    out_nfeat = out_dir / OUT_NFEAT.name
    out_users = out_dir / OUT_USERS.name
    out_streamers = out_dir / OUT_STREAMERS.name
    out_rooms = out_dir / OUT_ROOMS.name

    # 1. 載入所有互動記錄（可選擇只保留前 N 個 streamer / 每 streamer 前 K 個 user）
    df = load_interactions(
        max_streamers=args.max_streamers,
        max_users_per_streamer=args.max_users_per_streamer,
        streamer_sampling=args.streamer_sampling,
        seed=args.seed,
    )

    # 2. 建立三種實體的 id map：user, streamer, room
    users_map = build_id_map(df["user_id"], "user_id")
    streamers_map = build_id_map(df["streamer_id"], "streamer_id")
    rooms_map = build_id_map(df["room_id"], "room_id")

    # 合併回互動表，取得對應的連續 id
    tmp = df[
        ["user_id", "streamer_id", "room_id", "timestamp", "label", "interaction_type"]
    ].copy()
    tmp = tmp.merge(users_map, on="user_id", how="left")
    tmp = tmp.merge(streamers_map, on="streamer_id", how="left")
    tmp = tmp.merge(rooms_map, on="room_id", how="left")

    n_users = len(users_map)
    n_streamers = len(streamers_map)
    n_rooms = len(rooms_map)

    # 3. 把三種實體放進同一連續節點空間，留 0 做 padding
    USER_OFFSET = 1
    STREAMER_OFFSET = USER_OFFSET + n_users
    ROOM_OFFSET = STREAMER_OFFSET + n_streamers

    # 4. 針對每一個 (user, streamer, room, ts, label) 產生三條邊
    #    etype: 0=user-room, 1=user-streamer, 2=streamer-room
    #    (向量化：逐列 iterrows 在百萬列的切片上會慢到不可用，順序與原本逐列版本相同)
    ug = USER_OFFSET + tmp["user_id_cid"].to_numpy(dtype=np.int64)
    sg = STREAMER_OFFSET + tmp["streamer_id_cid"].to_numpy(dtype=np.int64)
    rg = ROOM_OFFSET + tmp["room_id_cid"].to_numpy(dtype=np.int64)
    ts_arr = tmp["timestamp"].to_numpy(dtype=np.float64)
    lbl_arr = tmp["label"].to_numpy(dtype=np.int64)
    n_rows = len(tmp)

    edges = pd.DataFrame(
        {
            "src": np.stack([ug, ug, sg], axis=1).reshape(-1),
            "dst": np.stack([rg, sg, rg], axis=1).reshape(-1),
            "ts": np.repeat(ts_arr, 3),
            "label": np.repeat(lbl_arr, 3),
            "etype": np.tile(np.array([0, 1, 2], dtype=np.int64), n_rows),
        }
    )
    edges = edges.sort_values(["ts", "etype"]).reset_index(drop=True)
    edges["idx"] = edges.index + 1  # 1-based index

    # 5. 邊特徵：簡單 3 維 one-hot (針對 etype)，和 `ml_tripartite_features.npy` 一致
    etype_eye = np.eye(3, dtype=float)
    edge_feat = np.vstack(
        [np.zeros((1, 3), dtype=float), etype_eye[edges["etype"].to_numpy()]]
    )

    # 6. 節點特徵：先全部填 0，維度 172，和原本 tripartite 節點特徵維度對齊
    n_nodes = ROOM_OFFSET + n_rooms
    node_feat = np.zeros((n_nodes + 1, 172), dtype=float)

    # 7. 輸出檔案
    edges.to_csv(out_edges, index=False)
    np.save(out_efeat, edge_feat)
    np.save(out_nfeat, node_feat)

    users_map.to_csv(out_users, index=False)
    streamers_map.to_csv(out_streamers, index=False)
    rooms_map.to_csv(out_rooms, index=False)
    print(f"\nWrote outputs to {out_dir}")

    # 簡單印出預覽與統計，方便手動檢查
    print("Preview of edges:")
    print(edges.head(12))

    summary = pd.DataFrame(
        {
            "n_users": [n_users],
            "n_streamers": [n_streamers],
            "n_rooms": [n_rooms],
            "n_nodes_total": [n_nodes],
            "n_edges": [len(edges)],
        }
    )
    print("\nSummary:")
    print(summary)


if __name__ == "__main__":
    main()

