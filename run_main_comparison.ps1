# =========================================================
# 主比較實驗（overall）：HT-Transformer vs. 所有可比 baseline。
#
# 薄包裝：實際工作交給 run_experiments.py --mode overall（它會逐模型訓練、彙整成
# experiment_results/results_overall.csv，含 test_/val_ 的 P@K、N@K、AUC、AP）。
# 這裡只負責：載入 common_config.ps1 的共用基準（$Common），確保所有模型對齊；附加 seed 數。
#
# 對齊重點（來自 common_config.ps1）：--train_neg_ratio / --train_neg_sampling 一致；
#   驗證負樣本已在 train 端預設 popularity。改基準只改 common_config.ps1。
# 排除 EdgeBank：非參數模型，BaselineTripartiteWrapper 不支援（run_experiments.py 的清單也未含）。
# 注意：run_experiments.py 對單一模型失敗是 check=True（會中止整批）；某 baseline 掛掉需先排除環境問題再續跑。
# =========================================================

. "$PSScriptRoot\common_config.ps1"

$MainRuns = 5   # 論文主表用多 seed 報 mean±std

Write-Host "=== 主比較 (overall)：run_experiments.py --mode overall × $MainRuns seeds ===" -ForegroundColor Green
Write-Host ("共用設定：neg_ratio={0} neg_sampling={1} epochs={2} patience={3} batch={4}" `
    -f $NegRatio, $NegSampling, $NumEpochs, $Patience, $BatchSize) -ForegroundColor DarkGray

python run_experiments.py --mode overall --num_runs $MainRuns $Common

if ($LASTEXITCODE -ne 0) {
    Write-Host "[FAIL] 主比較中止 (exit $LASTEXITCODE)。" -ForegroundColor Red
} else {
    Write-Host "[OK] 完成 -> experiment_results/results_overall.csv" -ForegroundColor Green
}
