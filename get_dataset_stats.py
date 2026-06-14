import pandas as pd
from utils.DataLoader import get_tripartite_link_prediction_data

# 讀取你的新資料集
dataset_name = "kuailive_tripartite" # 請確認你的資料集名稱
data = get_tripartite_link_prediction_data(dataset_name=dataset_name, val_ratio=0.15, test_ratio=0.15)

train_data = data[4]
val_data = data[5]
test_data = data[6]

full_data = data[3]
users = len(set(full_data.user_node_ids))
streamers = len(set(full_data.streamer_node_ids))
items = len(set(full_data.item_node_ids))
interactions = len(full_data.user_node_ids)

print(f"Number of users: {users}")
print(f"Number of streamers: {streamers}")
print(f"Number of items: {items}")
print(f"Number of interactions: {interactions}")
print(f"Train edges: {len(train_data.user_node_ids)} ({len(train_data.user_node_ids)/interactions:.2%})")
print(f"Val edges: {len(val_data.user_node_ids)} ({len(val_data.user_node_ids)/interactions:.2%})")
print(f"Test edges: {len(test_data.user_node_ids)} ({len(test_data.user_node_ids)/interactions:.2%})")