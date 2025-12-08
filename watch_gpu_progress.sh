#!/bin/bash
# 持續監控 GPU 訓練進度

LOG_FILE="logs/gpu_training_$(ls -t logs/gpu_training_*.log 2>/dev/null | head -1 | xargs basename)"

echo "監控 GPU 訓練進度..."
echo "日誌檔案: $LOG_FILE"
echo "按 Ctrl+C 停止監控"
echo ""

while true; do
    clear
    echo "=============================================="
    echo "GPU 訓練即時監控 - $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=============================================="

    # 訓練進度
    DATASET_NUM=$(tail -200 "$LOG_FILE" 2>/dev/null | grep -oP "Processing: train_dataset_\K\d+" | tail -1)
    if [ -n "$DATASET_NUM" ]; then
        echo "當前資料集: Dataset $DATASET_NUM / 8"
    else
        echo "當前資料集: 啟動中..."
    fi

    # 最新 AUC
    LATEST_AUC=$(tail -200 "$LOG_FILE" 2>/dev/null | grep "Final Validation AUC" | tail -1 | grep -oP "\d+\.\d+")
    if [ -n "$LATEST_AUC" ]; then
        echo "最新 Validation AUC: $LATEST_AUC"
    fi

    echo ""
    echo "--- GPU 狀態 ---"
    nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,temperature.gpu --format=csv,noheader

    echo ""
    echo "--- 進程狀態 ---"
    PID=$(cat logs/gpu_training.pid 2>/dev/null)
    if [ -n "$PID" ] && ps -p $PID > /dev/null 2>&1; then
        echo "✅ 運行中 (PID: $PID) | CPU: $(ps -p $PID -o %cpu | tail -1)% | RAM: $(ps -p $PID -o rss | tail -1 | awk '{printf "%.1f GB", $1/1024/1024}')"
    else
        echo "❌ 進程已停止"
        break
    fi

    echo ""
    echo "--- 最新訓練日誌 (最後 5 行) ---"
    tail -5 "$LOG_FILE" 2>/dev/null | grep -E "(Processing|Training|Final|Identifying|Got)" || echo "等待輸出..."

    echo ""
    echo "--- 已完成資料集 ---"
    COMPLETED=$(tail -500 "$LOG_FILE" 2>/dev/null | grep "Final Validation AUC" | wc -l)
    echo "已完成: $COMPLETED / 8 datasets"

    sleep 10
done

echo ""
echo "訓練已完成或中斷！"
