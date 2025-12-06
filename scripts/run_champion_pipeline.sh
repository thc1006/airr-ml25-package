#!/bin/bash
#
# Run Champion Pipeline for AIRR-ML-25
# =====================================
#
# This script runs the complete champion pipeline with GPU acceleration.
#
# Requirements:
# - CUDA-capable GPU (tested on RTX 5080)
# - Python 3.10+
# - Dependencies from requirements_gpu.txt
#
# Usage:
#   ./run_champion_pipeline.sh [cpu|cuda]
#

set -e  # Exit on error

# Configuration
DEVICE=${1:-cuda}  # Default to cuda, override with 'cpu' argument
N_JOBS=8
TRAIN_ROOT="./data/train_datasets/train_datasets"
TEST_ROOT="./data/test_datasets/test_datasets"
OUT_DIR="./results_champion_pipeline"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=================================================="
echo "AIRR-ML-25 Champion Pipeline"
echo "=================================================="
echo ""
echo "Configuration:"
echo "  Device: $DEVICE"
echo "  Jobs: $N_JOBS"
echo "  Train: $TRAIN_ROOT"
echo "  Test: $TEST_ROOT"
echo "  Output: $OUT_DIR"
echo ""

# Check data directories
if [ ! -d "$TRAIN_ROOT" ]; then
    echo -e "${RED}Error: Training data not found at $TRAIN_ROOT${NC}"
    echo "Please download data first:"
    echo "  kaggle competitions download -c adaptive-immune-profiling-challenge-2025"
    exit 1
fi

if [ ! -d "$TEST_ROOT" ]; then
    echo -e "${RED}Error: Test data not found at $TEST_ROOT${NC}"
    exit 1
fi

# Check Python environment
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: python3 not found${NC}"
    exit 1
fi

# Check required packages
echo "Checking dependencies..."
python3 -c "import xgboost, lightgbm, catboost, sklearn, pandas, numpy" 2>/dev/null
if [ $? -ne 0 ]; then
    echo -e "${YELLOW}Warning: Some dependencies missing${NC}"
    echo "Install with: pip install -r requirements_gpu.txt"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# GPU check
if [ "$DEVICE" = "cuda" ]; then
    echo "Checking GPU availability..."
    python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'" 2>/dev/null
    if [ $? -ne 0 ]; then
        echo -e "${YELLOW}Warning: CUDA not available, falling back to CPU${NC}"
        DEVICE="cpu"
    else
        GPU_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))")
        echo -e "${GREEN}GPU detected: $GPU_NAME${NC}"
    fi
fi

# Create output directory
mkdir -p "$OUT_DIR"

# Run pipeline
echo ""
echo "=================================================="
echo "Starting pipeline execution..."
echo "=================================================="
echo ""

python3 src/pipeline/champion_pipeline.py \
    --train_root "$TRAIN_ROOT" \
    --test_root "$TEST_ROOT" \
    --out_dir "$OUT_DIR" \
    --n_jobs $N_JOBS \
    --device $DEVICE

# Check results
if [ -f "$OUT_DIR/submissions.csv" ]; then
    echo ""
    echo "=================================================="
    echo -e "${GREEN}SUCCESS!${NC}"
    echo "=================================================="
    echo ""
    echo "Submission file: $OUT_DIR/submissions.csv"

    # Count rows
    TOTAL_ROWS=$(wc -l < "$OUT_DIR/submissions.csv")
    TOTAL_ROWS=$((TOTAL_ROWS - 1))  # Subtract header

    echo "Total rows: $TOTAL_ROWS (expected: 404,213)"

    if [ $TOTAL_ROWS -eq 404213 ]; then
        echo -e "${GREEN}✓ Row count is correct!${NC}"
    else
        echo -e "${YELLOW}⚠ Row count mismatch (diff: $((TOTAL_ROWS - 404213)))${NC}"
    fi

    echo ""
    echo "Next steps:"
    echo "  1. Validate submission: python validate_submission.py"
    echo "  2. Submit to Kaggle:"
    echo "     kaggle competitions submit -c adaptive-immune-profiling-challenge-2025 \\"
    echo "       -f $OUT_DIR/submissions.csv -m 'Champion pipeline submission'"
else
    echo ""
    echo -e "${RED}ERROR: Submission file not created${NC}"
    exit 1
fi
