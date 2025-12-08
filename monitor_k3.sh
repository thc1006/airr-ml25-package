#!/bin/bash
# GPU k=3 訓練監控腳本

PID=$(cat logs/gpu_k3.pid 2>/dev/null)

if [ -z "$PID" ]; then
    echo "❌ 無法找到 PID 檔案"
    exit 1
fi

echo "=== GPU k=3 訓練監控 ==="
echo "PID: $PID"
echo ""

while true; do
    # 檢查進程是否還在運行
    if ! ps -p $PID > /dev/null 2>&1; then
        echo "$(date +%H:%M:%S) - ⚠️  進程已結束"
        echo ""
        echo "=== 檢查輸出檔案 ==="
        if [ -f ./results_k4/train_dataset_8_test_predictions.tsv ]; then
            echo "✅ train_dataset_8_test_predictions.tsv 存在"
            wc -l ./results_k4/train_dataset_8_test_predictions.tsv
        else
            echo "❌ train_dataset_8_test_predictions.tsv 不存在"
        fi

        if [ -f ./results_k4/train_dataset_8_important_sequences.tsv ]; then
            echo "✅ train_dataset_8_important_sequences.tsv 存在"
            wc -l ./results_k4/train_dataset_8_important_sequences.tsv
        else
            echo "❌ train_dataset_8_important_sequences.tsv 不存在"
        fi

        echo ""
        echo "=== 最後 30 行日誌 ==="
        tail -30 logs/gpu_k3_*.log
        break
    fi

    # 顯示進程資源使用
    CPU_MEM=$(ps -p $PID -o %cpu,%mem --no-headers)
    GPU_INFO=$(nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null)

    # 顯示訓練進度（從日誌中提取）
    PROGRESS=$(tail -1 logs/gpu_k3_*.log 2>/dev/null | grep -oP '\d+%|\d+/\d+' | tail -1)

    echo "$(date +%H:%M:%S) - CPU/MEM: $CPU_MEM | GPU: $GPU_INFO | Progress: $PROGRESS"

    sleep 15
done
