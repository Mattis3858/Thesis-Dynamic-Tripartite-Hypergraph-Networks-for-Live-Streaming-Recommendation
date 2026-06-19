# =========================================================
# 共用實驗設定（公平比較的單一真實來源 / single source of truth）
#
# run_main_comparison.ps1（主比較）與 run_experiments.ps1（消融 + 超參數）都
# 透過 dot-source 載入這個檔（. .\common_config.ps1），確保兩邊用「同一組」訓練
# 基準。要調整對齊基準時，只改這一個檔，兩支腳本同步生效。
#
# 對齊重點：
#   - --train_neg_ratio / --train_neg_sampling 對齊所有模型（負樣本「數量」與「分布」一致）。
#   - 驗證（early-stopping）負樣本已在 train_tripartite_link_prediction.py 預設為 popularity，
#     與評估的 popularity 成分對齊，故此處不需指定。
#   - --num_runs 不放進 $Common：主比較要多 seed（mean±std），消融通常 1 個即可，
#     由各腳本自行附加。
# 注意：DyGLib 系 baseline 的 --num_neighbors 等沿用各自慣用預設（不在此覆寫）。
# =========================================================

$Gpu          = 0
$NumEpochs    = 30            # early stopping 的上限；實際以 patience 為準
$Patience     = 10
$BatchSize    = 64
$NegRatio     = 5             # 每個正樣本配的負樣本數（所有模型一致；時間緊可降到 3）
$NegSampling  = "popularity"  # 訓練負樣本分布：popularity / uniform（所有模型一致）

# 共用參數（不含 --num_runs，由各腳本附加）
$Common = @(
    "--num_epochs", $NumEpochs,
    "--patience", $Patience,
    "--batch_size", $BatchSize,
    "--gpu", $Gpu,
    "--train_neg_ratio", $NegRatio,
    "--train_neg_sampling", $NegSampling
)
