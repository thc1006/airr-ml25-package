#!/usr/bin/env python3
"""
Quick Submission Generator
Uses pre-trained checkpoints from checkpoints_fixed/
"""

import os
import gc
import json
import pickle
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ============================================================================
# Configuration
# ============================================================================
TRAIN_ROOT = './data/train_datasets/train_datasets'
TEST_ROOT = './data/test_datasets/test_datasets'
SUBMISSION_DIR = './submissions'
CHECKPOINT_DIR = './checkpoints_fixed'
FEATURE_NAMES_PATH = './checkpoints/feature_names_correct.json'

TOP_K = 50000

TEST_TO_TRAIN = {
    'test_dataset_1': 1,
    'test_dataset_2': 2,
    'test_dataset_3': 3,
    'test_dataset_4': 4,
    'test_dataset_5': 5,
    'test_dataset_6': 6,
    'test_dataset_7_1': 7,
    'test_dataset_7_2': 7,
    'test_dataset_8_1': 8,
    'test_dataset_8_2': 8,
    'test_dataset_8_3': 8,
}

# ============================================================================
# Feature Names
# ============================================================================

def load_feature_names():
    """Load feature names from checkpoint"""
    with open(FEATURE_NAMES_PATH, 'r') as f:
        return json.load(f)

# ============================================================================
# Feature Extraction (Fixed Bug)
# ============================================================================

def extract_features(tsv_path: str, feature_names: List[str]) -> np.ndarray:
    """Extract features with FIXED prefix handling"""
    df = pd.read_csv(tsv_path, sep='\t', usecols=['junction_aa', 'v_call', 'j_call'])
    df = df.dropna(subset=['junction_aa'])

    total = len(df)
    features = []

    # Build frequency dicts
    v_freq = defaultdict(float)
    j_freq = defaultdict(float)
    kmer_freq = defaultdict(float)

    for _, row in df.iterrows():
        seq = str(row['junction_aa'])

        # K-mer features (k=4)
        for i in range(len(seq) - 3):
            kmer = seq[i:i+4]
            kmer_freq[kmer] += 1

        # V gene
        v = str(row.get('v_call', ''))
        if v and v != 'nan':
            v_base = v.split('*')[0].split(',')[0].strip()
            v_freq[v_base] += 1

        # J gene
        j = str(row.get('j_call', ''))
        if j and j != 'nan':
            j_base = j.split('*')[0].split(',')[0].strip()
            j_freq[j_base] += 1

    # Normalize
    for k in v_freq:
        v_freq[k] /= total
    for k in j_freq:
        j_freq[k] /= total
    for k in kmer_freq:
        kmer_freq[k] /= total

    # Build feature vector - FIXED prefix handling
    for name in feature_names:
        if name.startswith('kmer_'):
            kmer = name[5:]  # Remove 'kmer_' prefix
            features.append(kmer_freq.get(kmer, 0.0))
        elif name.startswith('v_'):
            # v_TRBV20-1 -> TRBV20-1
            gene = name[2:]  # Remove 'v_' prefix
            features.append(v_freq.get(gene, 0.0))
        elif name.startswith('j_'):
            # j_TRBJ2-7 -> TRBJ2-7
            gene = name[2:]  # Remove 'j_' prefix
            features.append(j_freq.get(gene, 0.0))
        elif name == 'clonality':
            # Gini coefficient approximation
            if len(kmer_freq) > 1:
                vals = sorted(kmer_freq.values())
                n = len(vals)
                cumsum = np.cumsum(vals)
                gini = (2 * np.sum((np.arange(1, n+1) * vals)) / (n * np.sum(vals))) - (n+1)/n
                features.append(max(0, min(1, gini)))
            else:
                features.append(0.5)
        elif name == 'd50':
            if kmer_freq:
                vals = sorted(kmer_freq.values(), reverse=True)
                total_sum = sum(vals)
                cumsum = 0
                for i, v in enumerate(vals):
                    cumsum += v
                    if cumsum >= total_sum * 0.5:
                        features.append((i + 1) / len(vals))
                        break
                else:
                    features.append(1.0)
            else:
                features.append(1.0)
        else:
            features.append(0.0)

    return np.array(features, dtype=np.float32)


def extract_dataset_features(dataset_path: str, feature_names: List[str], is_test: bool = False) -> tuple:
    """Extract features for all repertoires in a dataset"""
    X_list = []
    ids = []

    if is_test:
        # Test dataset: no metadata, use filenames as IDs
        tsv_files = sorted([f for f in os.listdir(dataset_path) if f.endswith('.tsv')])
        for tsv_file in tqdm(tsv_files, desc=f"  {os.path.basename(dataset_path)}"):
            tsv_path = os.path.join(dataset_path, tsv_file)
            feat = extract_features(tsv_path, feature_names)
            X_list.append(feat)
            ids.append(tsv_file.replace('.tsv', ''))  # repertoire_id = filename without .tsv
    else:
        # Train dataset: use metadata
        meta_path = os.path.join(dataset_path, 'metadata.csv')
        meta = pd.read_csv(meta_path)
        for _, row in tqdm(meta.iterrows(), total=len(meta), desc=f"  {os.path.basename(dataset_path)}"):
            tsv_path = os.path.join(dataset_path, row['filename'])
            if os.path.exists(tsv_path):
                feat = extract_features(tsv_path, feature_names)
                X_list.append(feat)
                ids.append(row['repertoire_id'])

    X = np.vstack(X_list)
    return X, ids

# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 80)
    print("  AIRR-ML-25 Quick Submission Generator")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # Load feature names
    feature_names = load_feature_names()
    print(f"\nLoaded {len(feature_names)} features")

    # Load trained models
    models = {}
    for ds in range(1, 9):
        ckpt_path = f"{CHECKPOINT_DIR}/xgb_ds{ds}.pkl"
        if os.path.exists(ckpt_path):
            with open(ckpt_path, 'rb') as f:
                data = pickle.load(f)
                models[ds] = data['model']
            print(f"  Loaded model for dataset {ds}")
        else:
            print(f"  WARNING: Missing checkpoint for dataset {ds}")

    # ========================================================================
    # Task A: Test Predictions
    # ========================================================================
    print("\n" + "=" * 60)
    print("Task A: Test Predictions")
    print("=" * 60)

    predictions = []
    test_dirs = sorted([d for d in os.listdir(TEST_ROOT) if d.startswith('test_dataset_')])

    for test_name in test_dirs:
        test_path = os.path.join(TEST_ROOT, test_name)
        train_ds = TEST_TO_TRAIN.get(test_name)

        if train_ds is None or train_ds not in models:
            print(f"  Skipping {test_name}: no matching model")
            continue

        model = models[train_ds]
        X_test, test_ids = extract_dataset_features(test_path, feature_names, is_test=True)

        # Predict (model is XGBClassifier)
        probs = model.predict_proba(X_test)[:, 1]  # Probability of positive class

        for rid, prob in zip(test_ids, probs):
            predictions.append({
                'ID': rid,
                'dataset': test_name,
                'label_positive_probability': float(prob),
                'junction_aa': '-999.0',
                'v_call': '-999.0',
                'j_call': '-999.0'
            })

        print(f"  {test_name}: {len(test_ids)} samples, range=[{probs.min():.4f}, {probs.max():.4f}]")

    # ========================================================================
    # Task B: Sequence Identification
    # ========================================================================
    print("\n" + "=" * 60)
    print("Task B: Sequence Identification")
    print("=" * 60)

    sequences = []

    for ds in range(1, 9):
        train_path = os.path.join(TRAIN_ROOT, f'train_dataset_{ds}')
        if not os.path.exists(train_path):
            print(f"  Skipping train_dataset_{ds}: not found")
            continue

        print(f"\n  train_dataset_{ds}...")

        # Collect all sequences with scores
        seq_scores = defaultdict(float)
        meta = pd.read_csv(os.path.join(train_path, 'metadata.csv'))

        # Get positive/negative counts
        pos_count = meta['label_positive'].sum()
        neg_count = len(meta) - pos_count

        for _, row in tqdm(meta.iterrows(), total=len(meta), desc="    Scanning"):
            tsv_path = os.path.join(train_path, row['filename'])
            if not os.path.exists(tsv_path):
                continue

            df = pd.read_csv(tsv_path, sep='\t')
            label = row['label_positive']

            for _, seq_row in df.iterrows():
                junction = str(seq_row.get('junction_aa', ''))
                v_call = str(seq_row.get('v_call', ''))
                j_call = str(seq_row.get('j_call', ''))

                if junction and junction != 'nan' and len(junction) >= 8:
                    key = (junction, v_call.split('*')[0], j_call.split('*')[0])
                    # Score: higher if in positive, lower if in negative
                    if label == 1:
                        seq_scores[key] += 1.0 / pos_count
                    else:
                        seq_scores[key] -= 0.5 / neg_count

        # Select top K
        sorted_seqs = sorted(seq_scores.items(), key=lambda x: -x[1])[:TOP_K]

        dataset_name = f'train_dataset_{ds}'
        for i, ((junction, v_call, j_call), score) in enumerate(sorted_seqs):
            sequences.append({
                'ID': f'{dataset_name}_seq_top_{i+1}',
                'dataset': dataset_name,
                'label_positive_probability': '-999.0',
                'junction_aa': junction,
                'v_call': v_call if v_call != 'nan' else '-999.0',
                'j_call': j_call if j_call != 'nan' else '-999.0'
            })

        print(f"    {len(sorted_seqs)} unique sequences identified")
        gc.collect()

    # ========================================================================
    # Create Submission
    # ========================================================================
    print("\n" + "=" * 60)
    print("Creating Submission File")
    print("=" * 60)

    all_rows = predictions + sequences
    submission_df = pd.DataFrame(all_rows)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = f'{SUBMISSION_DIR}/submission_fixed_{timestamp}.csv'
    submission_df.to_csv(out_path, index=False)

    print(f"\n  Total rows: {len(submission_df)}")
    print(f"  Task A rows: {len(predictions)}")
    print(f"  Task B rows: {len(sequences)}")
    print(f"  Saved to: {out_path}")
    print(f"  File size: {os.path.getsize(out_path) / 1024 / 1024:.2f} MB")

    # Verify
    print("\n  Verification:")
    print(f"    Expected rows: 404213")
    print(f"    Actual rows: {len(submission_df)}")
    print(f"    Match: {len(submission_df) == 404213}")

    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)

if __name__ == '__main__':
    main()
