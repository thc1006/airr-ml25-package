#!/bin/bash
# AIRR-ML-25 全自動提交腳本

echo "=================================================="
echo "AIRR-ML-25 全自動提交流程"
echo "=================================================="

# 等待 GPU 管道完成
echo "[1/4] 等待 GPU 管道完成..."
while pgrep -f "run_gpu_optimized.py" > /dev/null; do
    sleep 30
    echo "  ... 仍在執行中 ($(date '+%H:%M:%S'))"
    tail -3 results_gpu_optimized/run.log 2>/dev/null | grep -E "(Processing|AUC|Complete|Total)" || true
done
echo "  ✓ GPU 管道完成！"

# 檢查結果
echo ""
echo "[2/4] 檢查結果..."
if [ -f "results_gpu_optimized/submission.csv" ]; then
    ROWS=$(wc -l < results_gpu_optimized/submission.csv)
    echo "  ✓ submission.csv 已生成"
    echo "  ✓ 行數: $ROWS"
    
    # 驗證格式
    echo ""
    echo "[3/4] 驗證提交格式..."
    EXPECTED=404214  # 404213 + header
    if [ "$ROWS" -eq "$EXPECTED" ]; then
        echo "  ✓ 行數正確: $ROWS (預期 $EXPECTED)"
    else
        echo "  ⚠ 行數不符: $ROWS (預期 $EXPECTED)"
    fi
    
    # 檢查欄位
    HEAD=$(head -1 results_gpu_optimized/submission.csv)
    echo "  欄位: $HEAD"
    
    # 顯示樣本
    echo ""
    echo "  前 5 行預測:"
    head -6 results_gpu_optimized/submission.csv | tail -5
    
    # 提交到 Kaggle
    echo ""
    echo "[4/4] 提交到 Kaggle..."
    cd /home/thc1006/dev/airr-ml25-package
    kaggle competitions submit \
        -c adaptive-immune-profiling-challenge-2025 \
        -f results_gpu_optimized/submission.csv \
        -m "GPU XGBoost k=3 k-mers - auto submission $(date '+%Y-%m-%d %H:%M')"
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "=================================================="
        echo "✓ 提交成功！"
        echo "=================================================="
        
        # 等待並檢查分數
        echo ""
        echo "等待 60 秒後檢查分數..."
        sleep 60
        kaggle competitions submissions -c adaptive-immune-profiling-challenge-2025 | head -5
    else
        echo "✗ 提交失敗"
    fi
else
    echo "  ✗ submission.csv 未找到！"
    ls -la results_gpu_optimized/
fi

echo ""
echo "完成時間: $(date)"
