# =========================================================
# 消融 + 超參數敏感度（HTTransformer only）。
#
# 薄包裝：交給 run_experiments.py，自動彙整成 experiment_results/results_ablation.csv
# 與 results_hyperparam.csv。
#   --mode ablation   ：--no-use_* 三個開關的完整 2^3 = 8 組合。
#   --mode hyperparam ：patch_size {1,5,10,20} × channel_embedding_dim {32,64,128,256} = 16 組。
#
# 共用基準從 common_config.ps1 載入（與 run_main_comparison.ps1 同一份，baseline 設定一致）。
# 主比較（HT vs. 所有 baseline）請改跑 run_main_comparison.ps1。
# =========================================================

. "$PSScriptRoot\common_config.ps1"

$AblationRuns = 1   # 消融/超參數通常 1 個 seed 即可

Write-Host "=== 消融 (2^3) + 超參數 (4×4)：run_experiments.py ===" -ForegroundColor Green
Write-Host ("共用設定：neg_ratio={0} neg_sampling={1} epochs={2} patience={3} runs={4} batch={5}" `
    -f $NegRatio, $NegSampling, $NumEpochs, $Patience, $AblationRuns, $BatchSize) -ForegroundColor DarkGray

Write-Host "[1/2] 消融 (ablation)" -ForegroundColor Cyan
python run_experiments.py --mode ablation --num_runs $AblationRuns $Common

Write-Host "[2/2] 超參數 (hyperparam)" -ForegroundColor Cyan
python run_experiments.py --mode hyperparam --num_runs $AblationRuns $Common

Write-Host "=== 完成 -> experiment_results/results_ablation.csv, results_hyperparam.csv ===" -ForegroundColor Green
