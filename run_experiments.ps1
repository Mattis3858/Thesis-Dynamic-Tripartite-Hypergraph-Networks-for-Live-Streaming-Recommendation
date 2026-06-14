Write-Host "=== 開始執行論文後半段所有消融與超參數實驗 ===" -ForegroundColor Green

# ---------------------------------------------------------
# [實驗 5.8] 異質圖模組消融實驗
# ---------------------------------------------------------
# Write-Host "1. 訓練 w/o type-aware initialization" -ForegroundColor Cyan
# python train_tripartite_link_prediction.py --model_name HTTransformer --no-use_type_init --num_epochs 50 --patience 10 --num_runs 1 --batch_size 64

# Write-Host "2. 訓練 w/o heterogeneous co-occurrence" -ForegroundColor Cyan
# python train_tripartite_link_prediction.py --model_name HTTransformer --no-use_hetero_coocc --num_epochs 50 --patience 10 --num_runs 1 --batch_size 64

# ---------------------------------------------------------
# [實驗 5.11] 精準度架構微調實驗
# ---------------------------------------------------------
# Write-Host "3. 訓練 Alternative fusion design (Mean pooling)" -ForegroundColor Cyan
# python train_tripartite_link_prediction.py --model_name HTTransformer --fusion_mode mean --num_epochs 30 --patience 10 --num_runs 1 --batch_size 64

# Write-Host "4. 訓練 Top-ranked emphasis (BPR Loss)" -ForegroundColor Cyan
# python train_tripartite_link_prediction.py --model_name HTTransformer --loss_type bpr --num_epochs 30 --patience 10 --num_runs 1 --batch_size 64

# Write-Host "5. 訓練 Modified negative sampling (1對5負樣本)" -ForegroundColor Cyan
# python train_tripartite_link_prediction.py --model_name HTTransformer --train_neg_ratio 5 --num_epochs 30 --patience 10 --num_runs 1 --batch_size 64

# ---------------------------------------------------------
# [實驗 5.15] 超參數敏感度分析 (Patch Size)
# ---------------------------------------------------------
# Write-Host "6. 訓練 Patch Size 5" -ForegroundColor Yellow
# python train_tripartite_link_prediction.py --model_name HTTransformer --patch_size 5 --num_epochs 30 --patience 10 --num_runs 1 --batch_size 64

# Write-Host "7. 訓練 Patch Size 10" -ForegroundColor Yellow
# python train_tripartite_link_prediction.py --model_name HTTransformer --patch_size 10 --num_epochs 30 --patience 10 --num_runs 1 --batch_size 64

Write-Host "8. 訓練 Patch Size 20" -ForegroundColor Yellow
python train_tripartite_link_prediction.py --model_name HTTransformer --patch_size 20 --num_epochs 30 --patience 10 --num_runs 1 --batch_size 64

# ---------------------------------------------------------
# [實驗 5.15] 超參數敏感度分析 (Embedding Dimension)
# ---------------------------------------------------------
Write-Host "9. 訓練 Embedding Dimension 32" -ForegroundColor Yellow
python train_tripartite_link_prediction.py --model_name HTTransformer --channel_embedding_dim 32 --cooccurrence_dim 32 --num_epochs 30 --patience 10 --num_runs 1 --batch_size 64

Write-Host "10. 訓練 Embedding Dimension 64" -ForegroundColor Yellow
python train_tripartite_link_prediction.py --model_name HTTransformer --channel_embedding_dim 64 --cooccurrence_dim 64 --num_epochs 30 --patience 10 --num_runs 1 --batch_size 64

Write-Host "11. 訓練 Embedding Dimension 128" -ForegroundColor Yellow
python train_tripartite_link_prediction.py --model_name HTTransformer --channel_embedding_dim 128 --cooccurrence_dim 128 --num_epochs 30 --patience 10 --num_runs 1 --batch_size 64

Write-Host "=== 所有訓練任務已全數派發完畢！可以安心去睡覺了！ ===" -ForegroundColor Green