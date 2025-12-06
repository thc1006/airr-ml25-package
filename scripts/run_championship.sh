#!/bin/bash
# =============================================================================
# AIRR-ML-25 Championship Runner
# =============================================================================
# GPU-accelerated pipeline with RAPIDS cuML + XGBoost + LightGBM + CatBoost
# Target: Score > 0.82 for first place
# =============================================================================

set -e

echo "=============================================="
echo "AIRR-ML-25 Championship Pipeline"
echo "=============================================="
echo "Start time: $(date)"
echo ""

# Set CUDA library paths for cuML
export LD_LIBRARY_PATH="/home/thc1006/.local/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib:/home/thc1006/.local/lib/python3.10/site-packages/nvidia/cusolver/lib:/home/thc1006/.local/lib/python3.10/site-packages/nvidia/cublas/lib:/home/thc1006/.local/lib/python3.10/site-packages/nvidia/cusparse/lib:$LD_LIBRARY_PATH"

# GPU memory optimization for RTX 5080
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Create output directory
OUTPUT_DIR="./results_championship"
mkdir -p "$OUTPUT_DIR"

echo "Output directory: $OUTPUT_DIR"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'Not detected')"
echo ""

# Check if cuML acceleration is available
if python3 -c "import cuml" 2>/dev/null; then
    echo "Using RAPIDS cuML acceleration"
    PYTHON_CMD="python3 -m cuml.accel"
else
    echo "Running without cuML (GPU models still enabled)"
    PYTHON_CMD="python3"
fi

# Run the championship pipeline
$PYTHON_CMD - << 'EOF'
import sys
sys.path.insert(0, '.')

from src.pipeline.champion_pipeline import ImmuneStatePredictor
import pandas as pd
from pathlib import Path

print("=" * 50)
print("Loading Champion Pipeline...")
print("=" * 50)

# Initialize predictor with GPU
predictor = ImmuneStatePredictor(
    n_jobs=8,
    device='cuda',
    verbose=True
)

# Paths
train_root = Path("./data/train_datasets/train_datasets")
test_root = Path("./data/test_datasets/test_datasets")
out_dir = Path("./results_championship")

# Get all training datasets
train_datasets = sorted([d for d in train_root.iterdir() if d.is_dir()])
test_datasets = sorted([d for d in test_root.iterdir() if d.is_dir()])

print(f"\nFound {len(train_datasets)} training datasets")
print(f"Found {len(test_datasets)} test datasets")

all_predictions = []
all_sequences = []

# Process each training dataset and its corresponding test dataset(s)
for train_dir in train_datasets:
    dataset_name = train_dir.name
    print(f"\n{'='*50}")
    print(f"Processing: {dataset_name}")
    print(f"{'='*50}")

    # Fit on training data
    predictor.fit(str(train_dir))

    # Find corresponding test datasets
    # Test datasets may have suffixes like dataset_7_1, dataset_7_2
    base_num = dataset_name.split('_')[-1]  # Get the number
    matching_tests = [t for t in test_datasets if f"dataset_{base_num}" in t.name or t.name == f"test_dataset_{base_num}"]

    for test_dir in matching_tests:
        print(f"\n  Predicting: {test_dir.name}")

        # Get predictions
        preds = predictor.predict_proba(str(test_dir))
        all_predictions.append(preds)

    # Get important sequences for Task B
    print(f"\n  Identifying disease-associated sequences...")
    seq_df = predictor.identify_associated_sequences(str(train_dir), top_k=50000)
    all_sequences.append(seq_df)

# Combine all predictions
if all_predictions:
    final_predictions = pd.concat(all_predictions, ignore_index=True)
    print(f"\nTotal predictions: {len(final_predictions)}")

# Combine all sequences
if all_sequences:
    final_sequences = pd.concat(all_sequences, ignore_index=True)
    print(f"Total sequences: {len(final_sequences)}")

# Create final submission
print("\nCreating submission file...")
submission = pd.concat([final_predictions, final_sequences], ignore_index=True)
submission_path = out_dir / "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission saved to: {submission_path}")
print(f"Submission shape: {submission.shape}")

print("\n" + "=" * 50)
print("Championship Pipeline Complete!")
print("=" * 50)
EOF

echo ""
echo "=============================================="
echo "Completed at: $(date)"
echo "=============================================="

# Show results summary
if [ -f "$OUTPUT_DIR/submission.csv" ]; then
    echo ""
    echo "Submission Summary:"
    wc -l "$OUTPUT_DIR/submission.csv"
    head -5 "$OUTPUT_DIR/submission.csv"
fi
