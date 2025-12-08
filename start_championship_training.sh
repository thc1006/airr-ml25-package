#!/bin/bash

# Championship Training Launcher
# This script starts the deep learning training pipeline in the background

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  🏆 AIRR-ML-25 Championship Training Launcher 🏆            ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Check if already running
if pgrep -f "championship_dl.py" > /dev/null; then
    echo "⚠️  Training already in progress!"
    echo "   PID: $(pgrep -f championship_dl.py)"
    echo ""
    echo "To monitor: tail -f championship_training.log"
    echo "To stop: pkill -f championship_dl.py"
    exit 1
fi

# Create logs directory
mkdir -p ./logs

# Start training
echo "🚀 Starting championship training..."
echo "   Log file: ./logs/championship_training.log"
echo "   PID file: ./logs/championship.pid"
echo ""

nohup python3 -u championship_dl.py > ./logs/championship_training.log 2>&1 &
TRAINING_PID=$!
echo $TRAINING_PID > ./logs/championship.pid

echo "✅ Training started!"
echo "   PID: $TRAINING_PID"
echo ""
echo "Commands:"
echo "  Monitor progress:  tail -f ./logs/championship_training.log"
echo "  Check status:      ps -p $TRAINING_PID"
echo "  Stop training:     kill $TRAINING_PID"
echo "  GPU usage:         watch -n 1 nvidia-smi"
echo ""
