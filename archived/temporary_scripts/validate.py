#!/usr/bin/env python3
"""
Validate AIRR-ML-25 submission file.

Checks:
1. Row count (404,213 total)
2. Column names and order
3. Data types
4. Missing values (should use -999.0)
5. Probability ranges [0, 1]
6. ID format
"""

import sys
import pandas as pd
import numpy as np


def validate_submission(file_path: str) -> bool:
    """Validate submission file format.

    Args:
        file_path: Path to submissions.csv

    Returns:
        True if valid, False otherwise
    """
    print("="*60)
    print("AIRR-ML-25 Submission Validator")
    print("="*60)
    print(f"\nFile: {file_path}\n")

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return False

    all_valid = True

    # Check 1: Row count
    print("[1] Row Count")
    expected_rows = 404213
    actual_rows = len(df)
    if actual_rows == expected_rows:
        print(f"  ✓ {actual_rows} rows (correct)")
    else:
        print(f"  ❌ {actual_rows} rows (expected {expected_rows})")
        print(f"     Difference: {actual_rows - expected_rows:+d}")
        all_valid = False

    # Check 2: Columns
    print("\n[2] Columns")
    required_cols = ['ID', 'dataset', 'label_positive_probability', 'junction_aa', 'v_call', 'j_call']
    actual_cols = df.columns.tolist()

    if actual_cols == required_cols:
        print(f"  ✓ All columns present in correct order")
    else:
        print(f"  ❌ Column mismatch")
        print(f"     Expected: {required_cols}")
        print(f"     Actual: {actual_cols}")
        all_valid = False

    # Check 3: Data types
    print("\n[3] Data Types")
    if df['ID'].dtype == object:
        print(f"  ✓ ID is string")
    else:
        print(f"  ❌ ID should be string, got {df['ID'].dtype}")
        all_valid = False

    if df['dataset'].dtype == object:
        print(f"  ✓ dataset is string")
    else:
        print(f"  ❌ dataset should be string, got {df['dataset'].dtype}")
        all_valid = False

    if np.issubdtype(df['label_positive_probability'].dtype, np.number):
        print(f"  ✓ label_positive_probability is numeric")
    else:
        print(f"  ❌ label_positive_probability should be numeric, got {df['label_positive_probability'].dtype}")
        all_valid = False

    # Check 4: Missing values
    print("\n[4] Missing Values")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print(f"  ✓ No missing values")
    else:
        print(f"  ❌ Missing values found:")
        for col, count in missing[missing > 0].items():
            print(f"     {col}: {count}")
        all_valid = False

    # Check 5: Probabilities
    print("\n[5] Probability Ranges")
    probs = df['label_positive_probability']

    # Predictions should be in [0, 1]
    predictions = probs[probs != -999.0]
    if len(predictions) > 0:
        min_prob = predictions.min()
        max_prob = predictions.max()
        if min_prob >= 0 and max_prob <= 1:
            print(f"  ✓ Predictions in [0, 1] (min={min_prob:.4f}, max={max_prob:.4f})")
        else:
            print(f"  ❌ Predictions out of range: [{min_prob:.4f}, {max_prob:.4f}]")
            all_valid = False

        print(f"     {len(predictions)} prediction rows")
    else:
        print(f"  ⚠ No prediction rows found")

    # Sequences should be -999.0
    sequences = probs[probs == -999.0]
    if len(sequences) > 0:
        print(f"  ✓ {len(sequences)} sequence rows with -999.0")
    else:
        print(f"  ⚠ No sequence rows found")

    # Check 6: ID format
    print("\n[6] ID Format")
    prediction_ids = df[probs != -999.0]['ID']
    sequence_ids = df[probs == -999.0]['ID']

    # Check prediction IDs (should be repertoire IDs)
    if len(prediction_ids) > 0:
        sample_pred_ids = prediction_ids.head(3).tolist()
        print(f"  Sample prediction IDs: {sample_pred_ids}")

    # Check sequence IDs (should match pattern: dataset_seq_top_N)
    if len(sequence_ids) > 0:
        sample_seq_ids = sequence_ids.head(3).tolist()
        print(f"  Sample sequence IDs: {sample_seq_ids}")

        # Validate sequence ID format
        invalid_seq_ids = []
        for seq_id in sequence_ids:
            if not ('_seq_top_' in str(seq_id)):
                invalid_seq_ids.append(seq_id)

        if len(invalid_seq_ids) == 0:
            print(f"  ✓ All sequence IDs follow naming convention")
        else:
            print(f"  ❌ {len(invalid_seq_ids)} invalid sequence IDs")
            print(f"     Sample: {invalid_seq_ids[:3]}")
            all_valid = False

    # Check 7: Dataset breakdown
    print("\n[7] Dataset Breakdown")
    dataset_counts = df['dataset'].value_counts()
    print(f"  Unique datasets: {len(dataset_counts)}")

    # Expected: 8 training datasets (for sequences) + test datasets (for predictions)
    train_datasets = [d for d in dataset_counts.index if d.startswith('train_dataset_')]
    test_datasets = [d for d in dataset_counts.index if d.startswith('test_dataset_')]

    print(f"  Training datasets: {len(train_datasets)} (sequences)")
    print(f"  Test datasets: {len(test_datasets)} (predictions)")

    # Each training dataset should have ~50,000 sequences
    for train_ds in sorted(train_datasets):
        count = dataset_counts[train_ds]
        status = "✓" if count == 50000 else "⚠"
        print(f"    {status} {train_ds}: {count} sequences")

    # Summary
    print("\n" + "="*60)
    if all_valid:
        print("✅ VALIDATION PASSED - Submission file is valid!")
    else:
        print("❌ VALIDATION FAILED - Please fix errors above")
    print("="*60)

    return all_valid


def main():
    if len(sys.argv) < 2:
        print("Usage: python validate_submission.py <submission_file.csv>")
        sys.exit(1)

    file_path = sys.argv[1]
    valid = validate_submission(file_path)
    sys.exit(0 if valid else 1)


if __name__ == '__main__':
    main()
