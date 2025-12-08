#!/bin/bash
# GPU 訓練進度監控腳本

echo "==================================================="
echo "AIRR-ML-25 GPU 訓練監控儀表板"
echo "==================================================="
echo ""

# 檢查進程狀態
PID=$(cat logs/gpu_training.pid 2>/dev/null)
if [ -n "$PID" ] && ps -p $PID > /dev/null 2>&1; then
    echo "✅ 訓練進程運行中 (PID: $PID)"
    echo "   CPU: $(ps -p $PID -o %cpu | tail -1)%"
    echo "   RAM: $(ps -p $PID -o rss | tail -1 | awk '{printf "%.1f GB\n", $1/1024/1024}')"
else
    echo "❌ 訓練進程未運行"
fi

echo ""
echo "--- GPU 狀態 ---"
nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.free,temperature.gpu --format=csv,noheader

echo ""
echo "--- 最新訓練日誌 (最後 30 行) ---"
tail -30 logs/gpu_training_*.log 2>/dev/null | grep -E "(Processing:|Training|Validation|Predicting|sequences|AUC|Dataset)" || echo "無日誌輸出"

echo ""
echo "--- 已生成結果 ---"
ls -lh results_gpu_optimized/ 2>/dev/null || echo "results_gpu_optimized/ 目錄不存在"

echo ""
echo "==================================================="
echo "按 Ctrl+C 退出監控"
