#!/bin/bash
# 快速檢查訓練狀態

echo "🔍 Championship Training - Quick Status Check"
echo "=============================================="
echo ""

# Check process
PID=3376440
if ps -p $PID > /dev/null 2>&1; then
    echo "✅ Training Process: RUNNING (PID: $PID)"
    ps -p $PID -o pid,pcpu,pmem,etime,cmd --no-headers
else
    echo "⚠️  Training Process: NOT RUNNING"
    echo "   Check if it completed successfully:"
    echo "   ls -lh ./models/championship_fold*.pt"
fi

echo ""
echo "🎮 GPU Status:"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader | \
    awk -F', ' '{print "   Utilization: " $1 "\n   Memory: " $2 " / " $3 "\n   Temperature: " $4 "\n   Power: " $5}'

echo ""
echo "📊 Latest Training Progress:"
LATEST_LOG=$(ls -t ./logs/auto_train_*.log 2>/dev/null | head -1)
if [ -f "$LATEST_LOG" ]; then
    tail -5 "$LATEST_LOG" | grep -E "Dataset|Loading|Epoch|AUC|Fold" | tail -3
    echo ""
    echo "📝 Full log: $LATEST_LOG"
else
    echo "   No log file found"
fi

echo ""
echo "⏱️  Elapsed Time: $(ps -p $PID -o etime= 2>/dev/null || echo 'N/A')"
echo ""
