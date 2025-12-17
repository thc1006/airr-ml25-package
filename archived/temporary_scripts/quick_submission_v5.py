#!/usr/bin/env python3
"""
Quick Submission Generator v5 - Optimized for Speed
====================================================
- Uses pre-trained checkpoints from checkpoints_v4/
- Task B uses vectorized operations for speed
- Handles large files efficiently
"""

import os
import gc
import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings('ignore')

# Configuration
TRAIN_ROOT = Path('./data/train_datasets/train_datasets')
TEST_ROOT = Path('./data/test_datasets/test_datasets')
CHECKPOINT_DIR = Path('./checkpoints_v4')
OUTPUT_PATH = Path('./submissions/quick_v5_submission.csv')

TOP_K = 50000
NUM_WORKERS = 8

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

    # K-mer frequencies (k=4) - vectorized
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

            from scipy.stats import entropy
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


def process_test_file(args):
    """Process single test file for Task A."""
    tsv_path, feature_names, kmer_set, scaler_mean, scaler_scale, model_path = args

    try:
        df = pd.read_csv(tsv_path, sep='\t')
        features = extract_features(df, feature_names, kmer_set)
        features_scaled = (features - scaler_mean) / scaler_scale

        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        prob = model_data['model'].predict_proba(features_scaled.reshape(1, -1))[0, 1]

        return tsv_path.stem, float(prob)
    except Exception as e:
        return tsv_path.stem, 0.5


def extract_sequences_fast(tsv_path: Path) -> pd.DataFrame:
    """Fast sequence extraction - only reads necessary columns."""
    try:
        # Read only necessary columns
        df = pd.read_csv(tsv_path, sep='\t', usecols=['junction_aa', 'v_call', 'j_call'])

        # Filter valid sequences
        df = df.dropna(subset=['junction_aa'])
        df = df[df['junction_aa'].str.len() >= 8]
        df = df[df['junction_aa'].str.isalpha()]

        return df[['junction_aa', 'v_call', 'j_call']]
    except Exception as e:
        return pd.DataFrame(columns=['junction_aa', 'v_call', 'j_call'])


def main():
    print("=" * 60)
    print("Quick Submission Generator v5")
    print("=" * 60)

    # =========================================================================
    # Task A: Generate predictions for test datasets
    # =========================================================================
    print("\n[Task A] Generating predictions...")
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

        feature_names = model_data['feature_names']
        kmer_set = set([f.replace('kmer_', '') for f in feature_names if f.startswith('kmer_')])
        scaler_mean = model_data.get('scaler_mean', np.zeros(len(feature_names)))
        scaler_scale = model_data.get('scaler_scale', np.ones(len(feature_names)))

        # Get all test files
        tsv_files = sorted(test_dir.glob('*.tsv'))
        print(f"  {test_name}: {len(tsv_files)} files")

        # Prepare args for parallel processing
        args_list = [
            (f, feature_names, kmer_set, scaler_mean, scaler_scale, model_path)
            for f in tsv_files
        ]

        # Process in parallel
        with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = {executor.submit(process_test_file, args): args[0] for args in args_list}
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"    Predicting"):
                rep_id, prob = future.result()
                predictions.append({
                    'ID': rep_id,
                    'dataset': test_name,
                    'label_positive_probability': prob,
                    'junction_aa': '-999.0',
                    'v_call': '-999.0',
                    'j_call': '-999.0'
                })

    print(f"  Total Task A predictions: {len(predictions)}")

    # =========================================================================
    # Task B: Extract top sequences from training datasets
    # =========================================================================
    print("\n[Task B] Extracting top sequences...")
    sequences_rows = []

    for ds_id in range(1, 9):
        ds_name = f'train_dataset_{ds_id}'
        ds_dir = TRAIN_ROOT / ds_name
        if not ds_dir.exists():
            print(f"  Warning: {ds_dir} not found")
            continue

        print(f"\n  {ds_name}...")

        # Load metadata
        meta_path = ds_dir / 'metadata.csv'
        if not meta_path.exists():
            print(f"    Warning: metadata not found")
            continue

        meta_df = pd.read_csv(meta_path)
        meta_df['filepath'] = meta_df['filename'].apply(lambda x: ds_dir / x)

        # Separate positive and negative samples
        pos_files = meta_df[meta_df['label_positive'] == 1]['filepath'].tolist()
        neg_files = meta_df[meta_df['label_positive'] == 0]['filepath'].tolist()

        print(f"    Positive: {len(pos_files)}, Negative: {len(neg_files)}")

        # Collect all sequences with their frequencies
        pos_seq_counts = Counter()
        neg_seq_counts = Counter()

        # Process positive files
        for filepath in tqdm(pos_files, desc="    Scanning positive"):
            df = extract_sequences_fast(filepath)
            for _, row in df.iterrows():
                key = (row['junction_aa'], str(row.get('v_call', '')), str(row.get('j_call', '')))
                pos_seq_counts[key] += 1

        # Process negative files
        for filepath in tqdm(neg_files, desc="    Scanning negative"):
            df = extract_sequences_fast(filepath)
            for _, row in df.iterrows():
                key = (row['junction_aa'], str(row.get('v_call', '')), str(row.get('j_call', '')))
                neg_seq_counts[key] += 1

        # Calculate scores: positive frequency - negative frequency (normalized)
        all_seqs = set(pos_seq_counts.keys()) | set(neg_seq_counts.keys())
        pos_total = sum(pos_seq_counts.values()) or 1
        neg_total = sum(neg_seq_counts.values()) or 1

        scored_seqs = []
        for seq_key in all_seqs:
            pos_freq = pos_seq_counts.get(seq_key, 0) / pos_total
            neg_freq = neg_seq_counts.get(seq_key, 0) / neg_total
            score = pos_freq - neg_freq  # Higher = more associated with positive
            scored_seqs.append((score, seq_key))

        # Sort by absolute score and take top K
        scored_seqs.sort(key=lambda x: abs(x[0]), reverse=True)
        top_seqs = scored_seqs[:TOP_K]

        # Create rows for Task B
        for i, (score, (junction, v_call, j_call)) in enumerate(top_seqs):
            sequences_rows.append({
                'ID': f'{ds_name}_seq_top_{i+1}',
                'dataset': ds_name,
                'label_positive_probability': -999.0,
                'junction_aa': junction,
                'v_call': v_call if v_call != 'nan' else '-999.0',
                'j_call': j_call if j_call != 'nan' else '-999.0'
            })

        print(f"    {len(top_seqs)} unique sequences")

        gc.collect()

    # =========================================================================
    # Create final submission
    # =========================================================================
    print("\n[Creating submission]...")

    # Combine Task A and Task B
    task_a_df = pd.DataFrame(predictions)
    task_b_df = pd.DataFrame(sequences_rows)

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

    print("\n[DONE] Submission ready for Kaggle!")


if __name__ == '__main__':
    main()
