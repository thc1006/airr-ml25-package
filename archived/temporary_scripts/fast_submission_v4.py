#!/usr/bin/env python3
"""
Fast Submission Generator v4
============================
Uses pre-trained checkpoints from checkpoints_v4/
Optimized for speed with parallel processing
"""

import os
import gc
import json
import pickle
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set
from collections import defaultdict, Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ============================================================================
# Configuration
# ============================================================================
TRAIN_ROOT = Path('./data/train_datasets/train_datasets')
TEST_ROOT = Path('./data/test_datasets/test_datasets')
SUBMISSION_DIR = Path('./submissions')
CHECKPOINT_DIR = Path('./checkpoints_v4')

TOP_K = 50000
NUM_WORKERS = 8  # Use all CPU cores

TEST_TO_TRAIN = {
    'test_dataset_1': 1, 'test_dataset_2': 2, 'test_dataset_3': 3,
    'test_dataset_4': 4, 'test_dataset_5': 5, 'test_dataset_6': 6,
    'test_dataset_7_1': 7, 'test_dataset_7_2': 7,
    'test_dataset_8_1': 8, 'test_dataset_8_2': 8, 'test_dataset_8_3': 8,
}

# ============================================================================
# Feature Extraction (Same as training)
# ============================================================================
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

            # Clonality
            from scipy.stats import entropy
            shannon = entropy(freqs)
            max_ent = np.log(len(counts)) if len(counts) > 1 else 1
            features['clonality'] = 1 - (shannon / max_ent) if max_ent > 0 else 0
            features['shannon_entropy'] = shannon
            features['gini_simpson'] = 1 - np.sum(freqs ** 2)

            # D50
            cumsum = np.cumsum(np.sort(freqs)[::-1])
            features['d50'] = (np.searchsorted(cumsum, 0.5) + 1) / len(freqs)

            # Length stats
            lengths = seqs.str.len()
            features['mean_cdr3_length'] = lengths.mean()
            features['std_cdr3_length'] = lengths.std() if len(lengths) > 1 else 0

            # Top clone frequency
            features['top_clone_freq'] = freqs[0] if len(freqs) > 0 else 0
            features['unique_ratio'] = len(counts) / len(seqs)

    # Build feature vector
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

        # Load model and predict
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        prob = model_data['model'].predict_proba(features_scaled.reshape(1, -1))[0, 1]

        return tsv_path.stem, float(prob)
    except Exception as e:
        return tsv_path.stem, 0.5


def process_train_file_for_sequences(args):
    """Process single train file for Task B sequence extraction."""
    tsv_path, label = args

    try:
        df = pd.read_csv(tsv_path, sep='\t', usecols=['junction_aa', 'v_call', 'j_call'])
        sequences = []

        for _, row in df.iterrows():
            junction = row.get('junction_aa')
            if pd.isna(junction):
                continue
            junction = str(junction)
            if len(junction) < 8 or not junction.isalpha():
                continue

            v_call = str(row.get('v_call', '')).split('*')[0] if pd.notna(row.get('v_call')) else ''
            j_call = str(row.get('j_call', '')).split('*')[0] if pd.notna(row.get('j_call')) else ''

            sequences.append((junction, v_call, j_call, label))

        return sequences
    except:
        return []


# ============================================================================
# Main Pipeline
# ============================================================================
def main():
    start_time = datetime.now()
    print("=" * 80)
    print("  AIRR-ML-25 Fast Submission Generator v4")
    print(f"  Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Workers: {NUM_WORKERS}")
    print("=" * 80)

    # Load feature names and kmer set
    with open(CHECKPOINT_DIR / 'feature_names.json') as f:
        feature_names = json.load(f)

    kmer_set = set()
    for name in feature_names:
        if name.startswith('kmer_'):
            kmer_set.add(name[5:])

    print(f"\nLoaded {len(feature_names)} features, {len(kmer_set)} k-mers")

    # Load all models and scalers
    print("\nLoading models...")
    models = {}
    for ds in range(1, 9):
        model_path = CHECKPOINT_DIR / f'xgb_ds{ds}.pkl'
        if model_path.exists():
            with open(model_path, 'rb') as f:
                models[ds] = pickle.load(f)
            print(f"  DS{ds}: loaded")

    # ========================================================================
    # Task A: Test Predictions
    # ========================================================================
    print("\n" + "=" * 60)
    print("Task A: Test Predictions")
    print("=" * 60)

    predictions = []

    for test_name, train_ds in TEST_TO_TRAIN.items():
        test_path = TEST_ROOT / test_name
        if not test_path.exists():
            continue

        if train_ds not in models:
            print(f"  Skipping {test_name}: no model")
            continue

        model_data = models[train_ds]
        scaler = model_data['scaler']

        tsv_files = list(test_path.glob('*.tsv'))
        print(f"\n  {test_name}: {len(tsv_files)} files")

        for tsv_file in tqdm(tsv_files, desc=f"    Processing", leave=False):
            try:
                df = pd.read_csv(tsv_file, sep='\t')
                features = extract_features(df, feature_names, kmer_set)
                features_scaled = scaler.transform(features.reshape(1, -1))
                prob = model_data['model'].predict_proba(features_scaled)[0, 1]
            except:
                prob = 0.5

            predictions.append({
                'ID': tsv_file.stem,
                'dataset': test_name,
                'label_positive_probability': float(prob),
                'junction_aa': -999.0,
                'v_call': -999.0,
                'j_call': -999.0
            })

        probs = [p['label_positive_probability'] for p in predictions if p['dataset'] == test_name]
        print(f"    Range: [{min(probs):.4f}, {max(probs):.4f}]")

    print(f"\n  Total Task A predictions: {len(predictions)}")

    # ========================================================================
    # Task B: Sequence Identification (Optimized)
    # ========================================================================
    print("\n" + "=" * 60)
    print("Task B: Sequence Identification (Parallel)")
    print("=" * 60)

    all_sequences = []

    for ds in range(1, 9):
        ds_name = f'train_dataset_{ds}'
        train_path = TRAIN_ROOT / ds_name
        if not train_path.exists():
            continue

        print(f"\n  {ds_name}...")
        metadata = pd.read_csv(train_path / 'metadata.csv')

        # Prepare args for parallel processing
        args_list = []
        for _, row in metadata.iterrows():
            tsv_path = train_path / row['filename']
            if tsv_path.exists():
                args_list.append((tsv_path, int(row['label_positive'])))

        # Process files in parallel
        seq_scores = defaultdict(lambda: {'score': 0, 'count': 0, 'v': '', 'j': ''})

        with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = [executor.submit(process_train_file_for_sequences, args) for args in args_list]

            for future in tqdm(as_completed(futures), total=len(futures), desc="    Scanning"):
                sequences = future.result()
                for junction, v_call, j_call, label in sequences:
                    weight = 2.0 if label == 1 else 0.3
                    seq_scores[junction]['score'] += weight
                    seq_scores[junction]['count'] += 1
                    if v_call and not seq_scores[junction]['v']:
                        seq_scores[junction]['v'] = v_call
                    if j_call and not seq_scores[junction]['j']:
                        seq_scores[junction]['j'] = j_call

        # Sort and get top K
        sorted_seqs = sorted(seq_scores.items(),
                            key=lambda x: (x[1]['score'], x[1]['count']),
                            reverse=True)[:TOP_K]

        for rank, (junction, info) in enumerate(sorted_seqs, 1):
            all_sequences.append({
                'ID': f'{ds_name}_seq_top_{rank}',
                'dataset': ds_name,
                'label_positive_probability': -999.0,
                'junction_aa': junction,
                'v_call': info['v'] if info['v'] else 'TRBV20-1',
                'j_call': info['j'] if info['j'] else 'TRBJ2-7'
            })

        # Pad if needed
        current_count = len([s for s in all_sequences if s['dataset'] == ds_name])
        while current_count < TOP_K:
            current_count += 1
            all_sequences.append({
                'ID': f'{ds_name}_seq_top_{current_count}',
                'dataset': ds_name,
                'label_positive_probability': -999.0,
                'junction_aa': 'CASSXXX',
                'v_call': 'TRBV20-1',
                'j_call': 'TRBJ2-7'
            })

        print(f"    {len(sorted_seqs)} unique sequences")
        gc.collect()

    # ========================================================================
    # Create Submission
    # ========================================================================
    print("\n" + "=" * 60)
    print("Creating Submission")
    print("=" * 60)

    submission_df = pd.DataFrame(predictions + all_sequences)
    submission_df = submission_df[['ID', 'dataset', 'label_positive_probability', 'junction_aa', 'v_call', 'j_call']]

    SUBMISSION_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = SUBMISSION_DIR / f'submission_v4_{timestamp}.csv'
    submission_df.to_csv(out_path, index=False)

    expected = 4213 + 8 * TOP_K
    print(f"\n  Total rows: {len(submission_df)} (expected: {expected})")
    print(f"  Match: {len(submission_df) == expected}")
    print(f"  File: {out_path}")
    print(f"  Size: {out_path.stat().st_size / 1e6:.2f} MB")

    # Stats
    task_a_probs = [p['label_positive_probability'] for p in predictions]
    print(f"\n  Task A stats:")
    print(f"    Count: {len(task_a_probs)}")
    print(f"    Range: [{min(task_a_probs):.4f}, {max(task_a_probs):.4f}]")
    print(f"    Mean: {np.mean(task_a_probs):.4f}")
    print(f"    Std: {np.std(task_a_probs):.4f}")

    duration = datetime.now() - start_time
    print(f"\n" + "=" * 60)
    print(f"  COMPLETE in {duration}")
    print(f"  Submission: {out_path}")
    print("=" * 60)

    return str(out_path)


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
