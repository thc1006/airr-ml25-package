#!/bin/bash
# run_enhanced_pipeline.sh
# Complete pipeline execution script for main_v2.py

set -e  # Exit on error

# Configuration
TRAIN_ROOT="./data/train_datasets/train_datasets"
TEST_ROOT="./data/test_datasets/test_datasets"
OUT_DIR="./results_v2_$(date +%Y%m%d_%H%M%S)"
N_JOBS=8
K_VALUES="3 4 5"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "========================================================================"
echo "AIRR-ML-25 Enhanced Pipeline Runner (main_v2.py)"
echo "========================================================================"
echo ""
echo "Configuration:"
echo "  Train root:  $TRAIN_ROOT"
echo "  Test root:   $TEST_ROOT"
echo "  Output dir:  $OUT_DIR"
echo "  CPU cores:   $N_JOBS"
echo "  K-mer scales: $K_VALUES"
echo ""

# Check if data exists
if [ ! -d "$TRAIN_ROOT" ]; then
    echo -e "${RED}ERROR: Training data not found at $TRAIN_ROOT${NC}"
    echo "Please download the competition data:"
    echo "  kaggle competitions download -c adaptive-immune-profiling-challenge-2025 -p ./data/"
    echo "  cd data && unzip adaptive-immune-profiling-challenge-2025.zip"
    exit 1
fi

if [ ! -d "$TEST_ROOT" ]; then
    echo -e "${RED}ERROR: Test data not found at $TEST_ROOT${NC}"
    exit 1
fi

# Check if main_v2.py exists
if [ ! -f "main_v2.py" ]; then
    echo -e "${RED}ERROR: main_v2.py not found in current directory${NC}"
    exit 1
fi

# Create output directory
mkdir -p "$OUT_DIR"

# Start timer
START_TIME=$(date +%s)

echo -e "${GREEN}Starting enhanced pipeline...${NC}"
echo ""

# Run main_v2.py
python3 main_v2.py \
    --train_root "$TRAIN_ROOT" \
    --test_root "$TEST_ROOT" \
    --out_dir "$OUT_DIR" \
    --n_jobs "$N_JOBS" \
    --k_values $K_VALUES

# Check if successful
if [ $? -eq 0 ]; then
    # End timer
    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))
    MINUTES=$((ELAPSED / 60))
    SECONDS=$((ELAPSED % 60))

    echo ""
    echo -e "${GREEN}========================================================================"
    echo "Pipeline completed successfully!"
    echo "========================================================================${NC}"
    echo ""
    echo "Elapsed time: ${MINUTES}m ${SECONDS}s"
    echo ""

    # Validate submission file
    SUBMISSION_FILE="$OUT_DIR/submissions.csv"
    if [ -f "$SUBMISSION_FILE" ]; then
        ROW_COUNT=$(tail -n +2 "$SUBMISSION_FILE" | wc -l)
        echo "Submission file: $SUBMISSION_FILE"
        echo "  Total rows: $ROW_COUNT"
        echo "  Expected:   404213"

        if [ "$ROW_COUNT" -eq 404213 ]; then
            echo -e "  Status: ${GREEN}✓ VALID${NC}"
            echo ""
            echo "Ready to submit!"
            echo "  kaggle competitions submit -c adaptive-immune-profiling-challenge-2025 \\"
            echo "                            -f $SUBMISSION_FILE \\"
            echo "                            -m 'Enhanced features: multi-scale k-mers + V/J + clonality + length'"
        else
            echo -e "  Status: ${YELLOW}⚠ WARNING - Row count mismatch${NC}"
            echo "  Difference: $((ROW_COUNT - 404213))"
        fi
    else
        echo -e "${RED}ERROR: Submission file not found!${NC}"
        exit 1
    fi
else
    echo -e "${RED}ERROR: Pipeline failed!${NC}"
    exit 1
fi

echo ""
echo "Output files:"
ls -lh "$OUT_DIR"

echo ""
echo "========================================================================"
echo "Next steps:"
echo "  1. Review outputs in $OUT_DIR"
echo "  2. Validate submission format"
echo "  3. Submit to Kaggle (if validation passed)"
echo "========================================================================"
