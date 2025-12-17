#!/usr/bin/env python3
"""
Simple Update v4 - Sequential processing (no multiprocessing issues)
====================================================================
"""

import os
import pickle
import warnings
from pathlib import Path
from typing import List, Set
from collections import Counter

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import entropy

warnings.filterwarnings('ignore')

# Configuration
TEST_ROOT = Path('./data/test_datasets/test_datasets')
CHECKPOINT_DIR = Path('./checkpoints_v4')
EXISTING_SUBMISSION = Path('./submissions/esm_xgb_submission_20251215_114211.csv')
OUTPUT_PATH = Path('./submissions/v4_updated_submission.csv')

TEST_TO_TRAIN = {
    'test_dataset_1': 1, 'test_dataset_2': 2, 'test_dataset_3': 3,
    'test_dataset_4': 4, 'test_dataset_5': 5, 'test_dataset_6': 6,
    'test_dataset_7_1': 7, 'test_dataset_7_2': 7,
    'test_dataset_8_1': 8, 'test_dataset_8_2': 8, 'test_dataset_8_3': 8,
}


def extract_features(df: pd.DataFrame, feature_names: List[str], kmer_set: Set[str]) -> np.ndarray:
    """Extract features from repertoire DataFrame."""
    features = {}
    total = len(df)
    if total == 0:
        return np.zeros(len(feature_names), dtype=np.float32)

    # K-mer frequencies (k=4)
    if 'junction_aa' in df.columns:
        kmer_counts = Counter()
        for seq in df['junction_aa'].dropna().astype(str):
            for i in range(len(seq) - 3):
                kmer = seq[i:i+4]
                if kmer.isalpha() and kmer in kmer_set:
                    kmer_counts[kmer] += 1
        for kmer, count in kmer_counts.items():
            features[f'kmer_{kmer}'] = count / total

    # V gene usage
    if 'v_call' in df.columns:
        v_counts = df['v_call'].dropna().astype(str).apply(lambda x: x.split('*')[0]).value_counts()
        for gene, count in v_counts.items():
            features[f'v_{gene}'] = count / total

    # J gene usage
    if 'j_call' in df.columns:
        j_counts = df['j_call'].dropna().astype(str).apply(lambda x: x.split('*')[0]).value_counts()
        for gene, count in j_counts.items():
            features[f'j_{gene}'] = count / total

    # Diversity metrics
    if 'junction_aa' in df.columns:
        seqs = df['junction_aa'].dropna()
        if len(seqs) > 0:
            counts = seqs.value_counts()
            freqs = counts.values / counts.sum()

            shannon = entropy(freqs)
            max_ent = np.log(len(counts)) if len(counts) > 1 else 1
            features['clonality'] = 1 - (shannon / max_ent) if max_ent > 0 else 0
            features['shannon_entropy'] = shannon
            features['gini_simpson'] = 1 - np.sum(freqs ** 2)

            cumsum = np.cumsum(np.sort(freqs)[::-1])
            features['d50'] = (np.searchsorted(cumsum, 0.5) + 1) / len(freqs)

            lengths = seqs.str.len()
            features['mean_cdr3_length'] = lengths.mean()
            features['std_cdr3_length'] = lengths.std() if len(lengths) > 1 else 0
            features['top_clone_freq'] = freqs[0] if len(freqs) > 0 else 0
            features['unique_ratio'] = len(counts) / len(seqs)

    result = np.zeros(len(feature_names), dtype=np.float32)
    for i, name in enumerate(feature_names):
        if name in features:
            v = features[name]
            if pd.notna(v) and not np.isinf(v):
                result[i] = float(v)
    return result


def main():
    print("=" * 60)
    print("Simple Update v4 - Sequential Processing")
    print("=" * 60)

    # Load existing submission
    print(f"\nLoading existing submission: {EXISTING_SUBMISSION}")
    existing_df = pd.read_csv(EXISTING_SUBMISSION)

    # Separate Task A and Task B
    task_b_df = existing_df[existing_df['label_positive_probability'] == -999.0].copy()
    print(f"Task B rows to reuse: {len(task_b_df)}")

    # Generate new Task A predictions
    print("\n[Task A] Generating predictions with v4 models...")
    predictions = []

    for test_name, train_id in TEST_TO_TRAIN.items():
        test_dir = TEST_ROOT / test_name
        if not test_dir.exists():
            print(f"  Warning: {test_dir} not found")
            continue

        # Load model
        model_path = CHECKPOINT_DIR / f'xgb_ds{train_id}.pkl'
        if not model_path.exists():
            print(f"  Warning: {model_path} not found")
            continue

        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)

        model = model_data['model']
        feature_names = model_data['feature_names']
        kmer_set = set([f.replace('kmer_', '') for f in feature_names if f.startswith('kmer_')])
        scaler_mean = model_data.get('scaler_mean', np.zeros(len(feature_names)))
        scaler_scale = model_data.get('scaler_scale', np.ones(len(feature_names)))

        # Get all test files
        tsv_files = sorted(test_dir.glob('*.tsv'))
        print(f"  {test_name}: {len(tsv_files)} files", flush=True)

        # Process each file sequentially
        for tsv_path in tqdm(tsv_files, desc=f"    Predicting", leave=True):
            try:
                df = pd.read_csv(tsv_path, sep='\t')
                features = extract_features(df, feature_names, kmer_set)
                features_scaled = (features - scaler_mean) / scaler_scale
                prob = model.predict_proba(features_scaled.reshape(1, -1))[0, 1]
            except Exception as e:
                prob = 0.5

            predictions.append({
                'ID': tsv_path.stem,
                'dataset': test_name,
                'label_positive_probability': float(prob),
                'junction_aa': '-999.0',
                'v_call': '-999.0',
                'j_call': '-999.0'
            })

    print(f"  Total Task A predictions: {len(predictions)}", flush=True)

    # Create Task A DataFrame
    task_a_df = pd.DataFrame(predictions)

    # Combine Task A and Task B
    print("\n[Creating submission]...")
    submission_df = pd.concat([task_a_df, task_b_df], ignore_index=True)

    # Ensure correct column order
    submission_df = submission_df[['ID', 'dataset', 'label_positive_probability', 'junction_aa', 'v_call', 'j_call']]

    # Save
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    submission_df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nSubmission saved to: {OUTPUT_PATH}")
    print(f"Total rows: {len(submission_df)}")
    print(f"  Task A: {len(task_a_df)}")
    print(f"  Task B: {len(task_b_df)}")

    # Validate
    assert len(submission_df) == 404213, f"Expected 404213 rows, got {len(submission_df)}"

    # Show prediction statistics
    print("\nPrediction statistics:")
    for ds in sorted(task_a_df['dataset'].unique()):
        ds_preds = task_a_df[task_a_df['dataset'] == ds]['label_positive_probability']
        print(f"  {ds}: mean={ds_preds.mean():.4f}, std={ds_preds.std():.4f}")

    print("\n[DONE] Submission ready for Kaggle!")


if __name__ == '__main__':
    main()
