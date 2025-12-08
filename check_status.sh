#!/bin/bash

# Championship Training Status Checker

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  🏆 Championship Training Status                            ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Check if training is running
if pgrep -f "championship_dl.py" > /dev/null; then
    PID=$(pgrep -f "championship_dl.py")
    echo "✅ Training is RUNNING"
    echo "   PID: $PID"
    echo "   Started: $(ps -p $PID -o lstart=)"
    echo ""
else
    echo "⚠️  Training is NOT running"
    echo ""
fi

# Check GPU status
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🖥️  GPU Status"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader | \
    awk -F', ' '{printf "   GPU: %s\n   Memory: %s / %s\n   Utilization: %s\n", $1, $2, $3, $4}'
echo ""

# Check log file
if [ -f "./logs/championship_training.log" ]; then
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📋 Recent Log Entries (last 10 lines)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    tail -10 ./logs/championship_training.log
    echo ""

    # Extract AUC scores if available
    if grep -q "Val AUC" ./logs/championship_training.log; then
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "📈 Validation AUC Scores"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        grep "Val AUC" ./logs/championship_training.log | tail -5
        echo ""
    fi
else
    echo "⚠️  No log file found at ./logs/championship_training.log"
    echo ""
fi

# Check model checkpoints
if [ -d "./models" ] && [ "$(ls -A ./models)" ]; then
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "💾 Model Checkpoints"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    ls -lh ./models/ | grep -v "^total" | awk '{printf "   %s  %s\n", $9, $5}'
    echo ""

    MODEL_COUNT=$(ls -1 ./models/*.pt 2>/dev/null | wc -l)
    echo "   Total models: $MODEL_COUNT / 8"
    echo ""
else
    echo "⚠️  No model checkpoints found in ./models/"
    echo ""
fi

# Commands
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📝 Useful Commands"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "   Monitor log:    tail -f ./logs/championship_training.log"
echo "   Watch GPU:      watch -n 1 nvidia-smi"
echo "   Stop training:  pkill -f championship_dl.py"
echo "   Check status:   ./check_status.sh"
echo ""
