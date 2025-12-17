#!/usr/bin/env python3
"""
Analyze V5 and V9 prediction distributions for Dataset 7
"""

import pandas as pd
import numpy as np
from pathlib import Path

def load_predictions(csv_path):
    """Load predictions CSV"""
    df = pd.read_csv(csv_path)
    # Filter only predictions (not sequences)
    # junction_aa should be string "-999.0" or float -999.0
    df = df[(df['junction_aa'] == -999.0) | (df['junction_aa'] == '-999.0')]
    return df

def analyze_dataset_predictions(v5_df, v9_df, dataset_name):
    """Analyze predictions for a specific dataset"""

    v5_data = v5_df[v5_df['dataset'] == dataset_name]['label_positive_probability']
    v9_data = v9_df[v9_df['dataset'] == dataset_name]['label_positive_probability']

    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*60}")

    print(f"\nV5 Predictions:")
    print(f"  Count: {len(v5_data)}")
    print(f"  Mean: {v5_data.mean():.4f}")
    print(f"  Std: {v5_data.std():.4f}")
    print(f"  Min: {v5_data.min():.4f}")
    print(f"  25%: {v5_data.quantile(0.25):.4f}")
    print(f"  Median: {v5_data.median():.4f}")
    print(f"  75%: {v5_data.quantile(0.75):.4f}")
    print(f"  Max: {v5_data.max():.4f}")

    # Check for collapse patterns
    near_zero = (v5_data < 0.1).sum()
    near_half = ((v5_data >= 0.4) & (v5_data <= 0.6)).sum()
    near_one = (v5_data > 0.9).sum()

    print(f"\nV5 Prediction Distribution:")
    print(f"  < 0.1 (strong negative): {near_zero} ({near_zero/len(v5_data)*100:.1f}%)")
    print(f"  0.4-0.6 (uncertain): {near_half} ({near_half/len(v5_data)*100:.1f}%)")
    print(f"  > 0.9 (strong positive): {near_one} ({near_one/len(v5_data)*100:.1f}%)")

    print(f"\nV9 Predictions:")
    print(f"  Count: {len(v9_data)}")
    print(f"  Mean: {v9_data.mean():.4f}")
    print(f"  Std: {v9_data.std():.4f}")
    print(f"  Min: {v9_data.min():.4f}")
    print(f"  25%: {v9_data.quantile(0.25):.4f}")
    print(f"  Median: {v9_data.median():.4f}")
    print(f"  75%: {v9_data.quantile(0.75):.4f}")
    print(f"  Max: {v9_data.max():.4f}")

    near_zero = (v9_data < 0.1).sum()
    near_half = ((v9_data >= 0.4) & (v9_data <= 0.6)).sum()
    near_one = (v9_data > 0.9).sum()

    print(f"\nV9 Prediction Distribution:")
    print(f"  < 0.1 (strong negative): {near_zero} ({near_zero/len(v9_data)*100:.1f}%)")
    print(f"  0.4-0.6 (uncertain): {near_half} ({near_half/len(v9_data)*100:.1f}%)")
    print(f"  > 0.9 (strong positive): {near_one} ({near_one/len(v9_data)*100:.1f}%)")

    # Correlation
    if len(v5_data) == len(v9_data):
        corr = np.corrcoef(v5_data, v9_data)[0, 1]
        print(f"\nCorrelation: {corr:.4f}")

        # Mean absolute difference
        mae = np.abs(v5_data.values - v9_data.values).mean()
        print(f"Mean Absolute Difference: {mae:.4f}")

        # Agreement on direction (both > 0.5 or both < 0.5)
        v5_pos = v5_data > 0.5
        v9_pos = v9_data > 0.5
        agreement = (v5_pos == v9_pos).sum() / len(v5_data)
        print(f"Agreement on classification (>0.5): {agreement*100:.1f}%")

        # Disagreement cases
        disagreement = (v5_pos != v9_pos).sum()
        print(f"Disagreement cases: {disagreement} ({disagreement/len(v5_data)*100:.1f}%)")

        if disagreement > 0:
            print(f"\nDisagreement details:")
            v5_says_pos_v9_says_neg = ((v5_data > 0.5) & (v9_data <= 0.5)).sum()
            v5_says_neg_v9_says_pos = ((v5_data <= 0.5) & (v9_data > 0.5)).sum()
            print(f"  V5 positive, V9 negative: {v5_says_pos_v9_says_neg}")
            print(f"  V5 negative, V9 positive: {v5_says_neg_v9_says_pos}")

def main():
    print("Dataset 7 Prediction Distribution Analysis")
    print("="*60)

    # Use specific files
    v5_file = Path('/home/thc1006/dev/airr-ml25-package/submissions/v5_submission_20251215_225131.csv')
    v9_file = Path('/home/thc1006/dev/airr-ml25-package/submission_ultimate.csv')

    if not v5_file.exists():
        print(f"\nERROR: V5 file not found: {v5_file}")
        return

    if not v9_file.exists():
        print(f"\nERROR: V9 file not found: {v9_file}")
        return

    print(f"\nUsing:")
    print(f"  V5: {v5_file.name}")
    print(f"  V9: {v9_file.name}")

    # Load predictions
    print("\nLoading predictions...")
    v5_df = load_predictions(v5_file)
    v9_df = load_predictions(v9_file)

    # Analyze Dataset 7
    analyze_dataset_predictions(v5_df, v9_df, 'test_dataset_7_1')
    analyze_dataset_predictions(v5_df, v9_df, 'test_dataset_7_2')

    # Compare with other datasets for reference
    print(f"\n{'='*60}")
    print("COMPARISON: Dataset 1 (for reference)")
    print(f"{'='*60}")
    analyze_dataset_predictions(v5_df, v9_df, 'test_dataset_1')

if __name__ == "__main__":
    main()
