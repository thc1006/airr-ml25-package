#!/usr/bin/env python3
"""
AIRR-ML-25 Championship Runner
==============================
GPU-accelerated pipeline without cuML (more stable).
Uses GPU for XGBoost, LightGBM, CatBoost directly.
"""

import sys
import os
import gc
from pathlib import Path

# Set up paths
sys.path.insert(0, '.')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from src.pipeline.champion_pipeline import ImmuneStatePredictor

def main():
    print("=" * 60)
    print("AIRR-ML-25 Championship Pipeline (GPU Mode)")
    print("=" * 60)

    # Paths
    train_root = Path("./data/train_datasets/train_datasets")
    test_root = Path("./data/test_datasets/test_datasets")
    out_dir = Path("./results_championship")
    out_dir.mkdir(exist_ok=True)

    # Get datasets
    train_datasets = sorted([d for d in train_root.iterdir() if d.is_dir()])
    test_datasets = sorted([d for d in test_root.iterdir() if d.is_dir()])

    print(f"\nFound {len(train_datasets)} training datasets")
    print(f"Found {len(test_datasets)} test datasets")

    all_predictions = []
    all_sequences = []

    # Process each training dataset
    for train_dir in train_datasets:
        dataset_name = train_dir.name
        print(f"\n{'='*60}")
        print(f"Processing: {dataset_name}")
        print(f"{'='*60}")

        # Initialize predictor with GPU (lower n_jobs to reduce memory)
        predictor = ImmuneStatePredictor(
            n_jobs=4,  # Reduced for stability
            device='cuda',
            verbose=True
        )

        # Fit on training data
        try:
            predictor.fit(str(train_dir))
        except Exception as e:
            print(f"Error fitting {dataset_name}: {e}")
            continue

        # Find corresponding test datasets
        base_num = dataset_name.split('_')[-1]
        matching_tests = [
            t for t in test_datasets
            if f"dataset_{base_num}" in t.name or t.name == f"test_dataset_{base_num}"
        ]

        for test_dir in matching_tests:
            print(f"\n  Predicting: {test_dir.name}")
            try:
                preds = predictor.predict_proba(str(test_dir))
                all_predictions.append(preds)
                print(f"    Got {len(preds)} predictions")
            except Exception as e:
                print(f"  Error predicting {test_dir.name}: {e}")

        # Get important sequences for Task B
        print(f"\n  Identifying disease-associated sequences...")
        try:
            seq_df = predictor.identify_associated_sequences(str(train_dir), top_k=50000)
            all_sequences.append(seq_df)
            print(f"    Got {len(seq_df)} sequences")
        except Exception as e:
            print(f"  Error identifying sequences: {e}")

        # Clean up memory
        del predictor
        gc.collect()

    # Combine all predictions
    if all_predictions:
        final_predictions = pd.concat(all_predictions, ignore_index=True)
        print(f"\nTotal predictions: {len(final_predictions)}")
    else:
        final_predictions = pd.DataFrame()
        print("\nWarning: No predictions generated!")

    # Combine all sequences
    if all_sequences:
        final_sequences = pd.concat(all_sequences, ignore_index=True)
        print(f"Total sequences: {len(final_sequences)}")
    else:
        final_sequences = pd.DataFrame()
        print("Warning: No sequences generated!")

    # Create submission
    if len(final_predictions) > 0 or len(final_sequences) > 0:
        print("\nCreating submission file...")
        submission = pd.concat([final_predictions, final_sequences], ignore_index=True)
        submission_path = out_dir / "submission.csv"
        submission.to_csv(submission_path, index=False)
        print(f"Submission saved to: {submission_path}")
        print(f"Submission shape: {submission.shape}")
    else:
        print("\nError: No data to submit!")

    print("\n" + "=" * 60)
    print("Championship Pipeline Complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
