#!/usr/bin/env python3
"""
Fixed Submission Generator
==========================
Fixes the feature extraction prefix mismatch bug.
Uses correct 490-feature set that covers both TCRBV and TRBV naming conventions.
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
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
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

TEST_DATASETS = list(TEST_TO_TRAIN.keys())
TRAIN_DATASETS = [f'train_dataset_{i}' for i in range(1, 9)]

# ============================================================================
# Feature Extraction (FIXED)
# ============================================================================
def load_feature_names() -> List[str]:
    """Load the correct 490-feature names."""
    with open(FEATURE_NAMES_PATH) as f:
        return json.load(f)


def extract_features(df: pd.DataFrame, feature_names: List[str]) -> np.ndarray:
    """
    Extract features from a repertoire DataFrame.

    FIXED: Properly handles both TCRBV and TRBV naming conventions.
    Features are stored as "v_TRBV20-1" format, and we extract "TRBV20-1" from data.
    """
    features = {}
    total = len(df)
    if total == 0:
        return np.zeros(len(feature_names), dtype=np.float32)

    # === V gene usage ===
    if 'v_call' in df.columns:
        v_counts = df['v_call'].dropna().astype(str).apply(lambda x: x.split('*')[0]).value_counts()
        for gene, count in v_counts.items():
            # Feature name format: "v_TRBV20-1" or "v_TCRBV02-01"
            features[f"v_{gene}"] = count / total

    # === J gene usage ===
    if 'j_call' in df.columns:
        j_counts = df['j_call'].dropna().astype(str).apply(lambda x: x.split('*')[0]).value_counts()
        for gene, count in j_counts.items():
            features[f"j_{gene}"] = count / total

    # === VJ pair usage ===
    if 'v_call' in df.columns and 'j_call' in df.columns:
        df_clean = df[['v_call', 'j_call']].dropna()
        if len(df_clean) > 0:
            v_genes = df_clean['v_call'].astype(str).apply(lambda x: x.split('*')[0])
            j_genes = df_clean['j_call'].astype(str).apply(lambda x: x.split('*')[0])
            pairs = v_genes + "_" + j_genes
            pair_counts = pairs.value_counts()
            for pair, count in pair_counts.items():
                features[f"vj_{pair}"] = count / total

    # === Clonality metrics ===
    if 'junction_aa' in df.columns:
        seqs = df['junction_aa'].dropna()
        if len(seqs) > 0:
            counts = seqs.value_counts()
            freqs = counts.values / counts.sum()

            # Shannon entropy
            from scipy.stats import entropy
            features['shannon_entropy'] = entropy(freqs)

            # Gini-Simpson
            features['gini_simpson'] = 1 - np.sum(freqs ** 2)

            # D50 (number of clones needed to reach 50% of total)
            cumsum = np.cumsum(np.sort(freqs)[::-1])
            features['d50'] = np.searchsorted(cumsum, 0.5) + 1

            # Clonality (normalized entropy)
            max_ent = np.log(len(counts)) if len(counts) > 1 else 1
            features['clonality'] = 1 - (features['shannon_entropy'] / max_ent) if max_ent > 0 else 0

            # Length statistics
            lengths = seqs.str.len()
            features['mean_length'] = lengths.mean()
            features['std_length'] = lengths.std() if len(lengths) > 1 else 0
            features['min_length'] = lengths.min()
            features['max_length'] = lengths.max()

            # Top clone frequency
            features['top_clone_freq'] = freqs[0]

    # === Build feature vector ===
    result = np.zeros(len(feature_names), dtype=np.float32)
    for i, name in enumerate(feature_names):
        if name in features:
            v = features[name]
            if pd.notna(v) and not np.isinf(v):
                result[i] = float(v)

    return result


# ============================================================================
# Training
# ============================================================================
def train_models(feature_names: List[str]) -> Dict:
    """Train XGBoost models for all 8 datasets."""
    print("\n" + "="*60)
    print("Training XGBoost Models")
    print("="*60)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    models = {}

    for ds_id in range(1, 9):
        print(f"\n--- Dataset {ds_id} ---")
        train_path = Path(TRAIN_ROOT) / f'train_dataset_{ds_id}'
        metadata = pd.read_csv(train_path / 'metadata.csv')

        X_list = []
        y_list = []

        for _, row in tqdm(metadata.iterrows(), total=len(metadata), desc=f"  Extracting DS{ds_id}"):
            tsv_path = train_path / row['filename']
            try:
                df = pd.read_csv(tsv_path, sep='\t')
                features = extract_features(df, feature_names)
                X_list.append(features)
                y_list.append(int(row['label_positive']))
            except Exception as e:
                X_list.append(np.zeros(len(feature_names), dtype=np.float32))
                y_list.append(int(row['label_positive']))

        X = np.array(X_list)
        y = np.array(y_list)

        # Check feature variance
        non_zero_features = np.sum(X.sum(axis=0) > 0)
        print(f"  Samples: {len(y)}, Non-zero features: {non_zero_features}/{len(feature_names)}")

        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Train XGBoost with GPU
        model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='auc',
            tree_method='hist',
            device='cuda',
            max_depth=6,
            learning_rate=0.05,
            n_estimators=200,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            verbosity=0,
        )

        model.fit(X_scaled, y, verbose=False)

        # Save
        model_data = {'model': model, 'scaler': scaler, 'feature_names': feature_names}
        model_path = Path(CHECKPOINT_DIR) / f'xgb_ds{ds_id}.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)

        models[ds_id] = model_data
        print(f"  Saved: {model_path}")

        # Quick validation
        train_pred = model.predict_proba(X_scaled)[:, 1]
        print(f"  Train pred range: [{train_pred.min():.4f}, {train_pred.max():.4f}]")

        gc.collect()

    return models


# ============================================================================
# Prediction
# ============================================================================
def predict_test_datasets(models: Dict, feature_names: List[str]) -> pd.DataFrame:
    """Generate predictions for all test datasets."""
    print("\n" + "="*60)
    print("Task A: Test Predictions")
    print("="*60)

    all_predictions = []

    for test_ds in TEST_DATASETS:
        train_id = TEST_TO_TRAIN[test_ds]

        if train_id not in models:
            print(f"  Skipping {test_ds} (no model)")
            continue

        model_data = models[train_id]
        model = model_data['model']
        scaler = model_data['scaler']

        test_path = Path(TEST_ROOT) / test_ds
        tsv_files = list(test_path.glob('*.tsv'))

        predictions = []
        for tsv_file in tqdm(tsv_files, desc=f"  {test_ds}", leave=False):
            rep_id = tsv_file.stem

            try:
                df = pd.read_csv(tsv_file, sep='\t')
                features = extract_features(df, feature_names)
                features_scaled = scaler.transform(features.reshape(1, -1))
                prob = model.predict_proba(features_scaled)[0, 1]
            except Exception as e:
                prob = 0.5

            predictions.append({
                'ID': rep_id,
                'dataset': test_ds,
                'label_positive_probability': prob,
                'junction_aa': -999.0,
                'v_call': -999.0,
                'j_call': -999.0
            })

        pred_df = pd.DataFrame(predictions)
        all_predictions.append(pred_df)

        # Check prediction variance
        probs = pred_df['label_positive_probability']
        print(f"  {test_ds}: {len(pred_df)} samples, range=[{probs.min():.4f}, {probs.max():.4f}]")

    return pd.concat(all_predictions, ignore_index=True)


# ============================================================================
# Task B: Sequence Identification
# ============================================================================
def identify_sequences() -> pd.DataFrame:
    """Identify top-k sequences from each training dataset."""
    print("\n" + "="*60)
    print("Task B: Sequence Identification")
    print("="*60)

    all_sequences = []

    for ds_name in TRAIN_DATASETS:
        print(f"\n  {ds_name}...")
        train_path = Path(TRAIN_ROOT) / ds_name
        metadata = pd.read_csv(train_path / 'metadata.csv')

        # Score sequences: higher weight for positive samples
        seq_scores = defaultdict(lambda: {'score': 0, 'count': 0, 'v': None, 'j': None})

        for _, row in tqdm(metadata.iterrows(), total=len(metadata), desc=f"    Scanning"):
            weight = 2.0 if row['label_positive'] else 0.5

            try:
                df = pd.read_csv(train_path / row['filename'], sep='\t')

                for _, seq_row in df.iterrows():
                    junc = seq_row.get('junction_aa')
                    if pd.isna(junc):
                        continue
                    junc = str(junc)

                    seq_scores[junc]['score'] += weight
                    seq_scores[junc]['count'] += 1

                    v = seq_row.get('v_call')
                    j = seq_row.get('j_call')
                    if pd.notna(v) and not seq_scores[junc]['v']:
                        seq_scores[junc]['v'] = str(v).split('*')[0]
                    if pd.notna(j) and not seq_scores[junc]['j']:
                        seq_scores[junc]['j'] = str(j).split('*')[0]
            except:
                continue

        # Sort by score and get top-k
        sorted_seqs = sorted(seq_scores.items(), key=lambda x: (x[1]['score'], x[1]['count']), reverse=True)[:TOP_K]

        for rank, (junc, info) in enumerate(sorted_seqs, 1):
            all_sequences.append({
                'ID': f'{ds_name}_seq_top_{rank}',
                'dataset': ds_name,
                'label_positive_probability': -999.0,
                'junction_aa': junc,
                'v_call': info['v'] if info['v'] else 'TRBV20-1',
                'j_call': info['j'] if info['j'] else 'TRBJ2-7'
            })

        # Pad if needed
        while len([s for s in all_sequences if s['dataset'] == ds_name]) < TOP_K:
            rank = len([s for s in all_sequences if s['dataset'] == ds_name]) + 1
            all_sequences.append({
                'ID': f'{ds_name}_seq_top_{rank}',
                'dataset': ds_name,
                'label_positive_probability': -999.0,
                'junction_aa': 'CASSXXX',
                'v_call': 'TRBV20-1',
                'j_call': 'TRBJ2-7'
            })

        print(f"    {len(sorted_seqs)} unique sequences identified")
        gc.collect()

    return pd.DataFrame(all_sequences)


# ============================================================================
# Main
# ============================================================================
def main():
    start = datetime.now()
    print(f"""
================================================================================
  AIRR-ML-25 Fixed Submission Generator
  Started: {start.strftime('%Y-%m-%d %H:%M:%S')}

  FIX: Feature extraction prefix mismatch bug
  Using: 490-feature set (both TCRBV and TRBV formats)
================================================================================
""")

    # Load feature names
    feature_names = load_feature_names()
    print(f"Loaded {len(feature_names)} features")

    # Train models
    models = train_models(feature_names)

    # Task A
    task_a = predict_test_datasets(models, feature_names)

    # Task B
    task_b = identify_sequences()

    # Combine
    print("\n" + "="*60)
    print("Generating Submission")
    print("="*60)

    submission = pd.concat([task_a, task_b], ignore_index=True)
    submission = submission[['ID', 'dataset', 'label_positive_probability', 'junction_aa', 'v_call', 'j_call']]

    expected = 4213 + 8 * TOP_K
    print(f"  Rows: {len(submission)} (expected: {expected})")

    # Save
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = f'{SUBMISSION_DIR}/fixed_submission_{ts}.csv'
    submission.to_csv(path, index=False)

    print(f"  Saved: {path}")
    print(f"  Size: {os.path.getsize(path) / 1e6:.2f} MB")

    # Stats
    probs = task_a['label_positive_probability']
    print(f"\n  Probability stats:")
    print(f"    Min: {probs.min():.4f}")
    print(f"    Max: {probs.max():.4f}")
    print(f"    Mean: {probs.mean():.4f}")
    print(f"    Std: {probs.std():.4f}")

    # Per-dataset variance check
    print(f"\n  Per-dataset prediction variance:")
    for ds in TEST_DATASETS:
        ds_probs = task_a[task_a['dataset'] == ds]['label_positive_probability']
        print(f"    {ds}: std={ds_probs.std():.4f}, range=[{ds_probs.min():.3f}, {ds_probs.max():.3f}]")

    duration = datetime.now() - start
    print(f"""
================================================================================
  COMPLETE!
  Duration: {duration}
  Submission: {path}
================================================================================
""")

    return path


if __name__ == "__main__":
    main()
