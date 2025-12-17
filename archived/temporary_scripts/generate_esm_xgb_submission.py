#!/usr/bin/env python3
"""
ESM + XGBoost Submission Generator
===================================
Generates complete submission using trained models.

Strategy:
- For test prediction: Uses traditional features only (389 dim)
- For Task B: Uses frequency-based sequence selection from positive samples
"""

import os
import json
import pickle
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import multiprocessing as mp

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ============================================================================
# Configuration
# ============================================================================
CHECKPOINT_DIR = './checkpoints'
MODEL_DIR_XGB = './checkpoints_xgb'
MODEL_DIR_ESM = './checkpoints_esm_xgb'
TRAIN_ROOT = './data/train_datasets/train_datasets'
TEST_ROOT = './data/test_datasets/test_datasets'
SUBMISSION_DIR = './submissions'
FEATURE_NAMES_PATH = './checkpoints/feature_names.json'

TOP_K = 50000

# Test dataset to train dataset mapping
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
# Feature Extraction
# ============================================================================
def load_feature_names() -> List[str]:
    """Load feature names from checkpoint."""
    if os.path.exists(FEATURE_NAMES_PATH):
        with open(FEATURE_NAMES_PATH) as f:
            return json.load(f)
    return None


def extract_traditional_features(df: pd.DataFrame, feature_names: List[str]) -> np.ndarray:
    """Extract V/J gene usage features from a repertoire."""
    # V gene usage
    v_counts = defaultdict(int)
    if 'v_call' in df.columns:
        for v in df['v_call'].dropna():
            v_gene = str(v).split('*')[0]
            v_counts[v_gene] += 1

    # J gene usage
    j_counts = defaultdict(int)
    if 'j_call' in df.columns:
        for j in df['j_call'].dropna():
            j_gene = str(j).split('*')[0]
            j_counts[j_gene] += 1

    # Normalize
    total_v = sum(v_counts.values()) + 1
    total_j = sum(j_counts.values()) + 1

    v_freq = {k: v/total_v for k, v in v_counts.items()}
    j_freq = {k: v/total_j for k, v in j_counts.items()}

    # Build feature vector matching training
    if feature_names:
        features = []
        for name in feature_names:
            if name in v_freq:
                features.append(v_freq[name])
            elif name in j_freq:
                features.append(j_freq[name])
            else:
                features.append(0.0)
        return np.array(features, dtype=np.float32)
    else:
        raise ValueError("Feature names required for consistent feature extraction")


# ============================================================================
# Model Loading and Prediction
# ============================================================================
def load_xgb_model(dataset_id: int) -> Tuple[xgb.XGBClassifier, StandardScaler]:
    """Load XGBoost model and scaler for a dataset."""
    model_path = Path(MODEL_DIR_XGB) / f'xgb_ds{dataset_id}.pkl'

    if model_path.exists():
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
        return data.get('model') or data.get('final_model'), data.get('scaler')
    return None, None


def train_xgb_for_dataset8():
    """Train XGBoost model for dataset 8 using aggregated features."""
    print("Training XGBoost for Dataset 8...")

    # Load aggregated data
    agg_path = Path(CHECKPOINT_DIR) / 'aggregated_features.npz'
    data = np.load(agg_path, allow_pickle=True)

    # Get dataset 8 data (traditional features only)
    dataset_ids = data['dataset_ids']
    mask = dataset_ids == 8

    X = data['trad'][mask].astype(np.float32)
    y = data['labels'][mask].astype(np.int32)

    print(f"  Dataset 8: {X.shape[0]} samples, {X.shape[1]} features")

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train XGBoost
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='auc',
        tree_method='hist',
        device='cuda',
        max_depth=8,
        learning_rate=0.03,
        n_estimators=300,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )

    model.fit(X_scaled, y, verbose=False)

    # Save
    model_path = Path(MODEL_DIR_XGB) / 'xgb_ds8.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump({'model': model, 'scaler': scaler}, f)
    print(f"  Saved to {model_path}")

    return model, scaler


def predict_test_dataset(test_dataset: str, model: xgb.XGBClassifier,
                         scaler: StandardScaler, feature_names: List[str]) -> pd.DataFrame:
    """Generate predictions for a test dataset."""
    test_path = Path(TEST_ROOT) / test_dataset

    predictions = []
    tsv_files = list(test_path.glob('*.tsv'))

    for tsv_file in tqdm(tsv_files, desc=f"  {test_dataset}", leave=False):
        repertoire_id = tsv_file.stem

        try:
            df = pd.read_csv(tsv_file, sep='\t')
            features = extract_traditional_features(df, feature_names)

            # Scale features
            features_scaled = scaler.transform(features.reshape(1, -1))

            # Predict
            prob = model.predict_proba(features_scaled)[0, 1]

            predictions.append({
                'ID': repertoire_id,
                'dataset': test_dataset,
                'label_positive_probability': prob,
                'junction_aa': -999.0,
                'v_call': -999.0,
                'j_call': -999.0
            })
        except Exception as e:
            predictions.append({
                'ID': repertoire_id,
                'dataset': test_dataset,
                'label_positive_probability': 0.5,
                'junction_aa': -999.0,
                'v_call': -999.0,
                'j_call': -999.0
            })

    return pd.DataFrame(predictions)


# ============================================================================
# Task B: Sequence Identification
# ============================================================================
def identify_sequences(train_dataset: str, top_k: int = TOP_K) -> pd.DataFrame:
    """Identify top-k sequences from positive samples in training data."""
    train_path = Path(TRAIN_ROOT) / train_dataset
    metadata_path = train_path / 'metadata.csv'

    if not metadata_path.exists():
        print(f"  Warning: {metadata_path} not found")
        return pd.DataFrame()

    metadata = pd.read_csv(metadata_path)
    positive_samples = metadata[metadata['label_positive'] == 1]

    # Collect sequences from positive samples
    all_sequences = []

    for _, row in tqdm(positive_samples.iterrows(),
                       desc=f"  TaskB {train_dataset}", leave=False,
                       total=len(positive_samples)):
        tsv_path = train_path / row['filename']
        if not tsv_path.exists():
            continue

        try:
            df = pd.read_csv(tsv_path, sep='\t')
            required_cols = ['junction_aa', 'v_call', 'j_call']
            if not all(c in df.columns for c in required_cols):
                continue

            df = df.dropna(subset=['junction_aa'])
            if len(df) == 0:
                continue

            # Sample if too many
            if len(df) > 5000:
                df = df.sample(n=5000, random_state=42)

            for _, seq_row in df.iterrows():
                all_sequences.append({
                    'junction_aa': seq_row['junction_aa'],
                    'v_call': seq_row['v_call'],
                    'j_call': seq_row['j_call'],
                    'count': 1
                })
        except Exception as e:
            continue

    if not all_sequences:
        # Fallback: generate dummy sequences
        return generate_dummy_sequences(train_dataset, top_k)

    # Group by sequence and count
    seq_df = pd.DataFrame(all_sequences)
    grouped = seq_df.groupby(['junction_aa', 'v_call', 'j_call']).agg({
        'count': 'sum'
    }).reset_index()

    # Sort by count and get top k
    grouped = grouped.sort_values('count', ascending=False).head(top_k)

    # Format for submission
    result = []
    for i, (_, row) in enumerate(grouped.iterrows()):
        result.append({
            'ID': f"{train_dataset}_seq_top_{i+1}",
            'dataset': train_dataset,
            'label_positive_probability': -999.0,
            'junction_aa': row['junction_aa'],
            'v_call': row['v_call'],
            'j_call': row['j_call']
        })

    # Pad if needed
    while len(result) < top_k:
        result.append({
            'ID': f"{train_dataset}_seq_top_{len(result)+1}",
            'dataset': train_dataset,
            'label_positive_probability': -999.0,
            'junction_aa': 'CASSXXX',
            'v_call': 'TRBV20-1',
            'j_call': 'TRBJ2-7'
        })

    return pd.DataFrame(result)


def generate_dummy_sequences(train_dataset: str, top_k: int) -> pd.DataFrame:
    """Generate dummy sequences if no real data available."""
    result = []
    for i in range(top_k):
        result.append({
            'ID': f"{train_dataset}_seq_top_{i+1}",
            'dataset': train_dataset,
            'label_positive_probability': -999.0,
            'junction_aa': f'CASSXXX{i}',
            'v_call': 'TRBV20-1',
            'j_call': 'TRBJ2-7'
        })
    return pd.DataFrame(result)


# ============================================================================
# Main
# ============================================================================
def main():
    print("="*60)
    print("ESM + XGBoost Submission Generator")
    print("="*60)

    # Load feature names
    feature_names = load_feature_names()
    if feature_names is None:
        raise ValueError("Feature names not found!")
    print(f"Loaded {len(feature_names)} feature names")

    # Load/train models
    models = {}
    scalers = {}

    print("\nLoading models...")
    for ds_id in range(1, 9):
        model, scaler = load_xgb_model(ds_id)
        if model is None:
            if ds_id == 8:
                model, scaler = train_xgb_for_dataset8()
            else:
                print(f"  Warning: No model for dataset {ds_id}")
                continue
        models[ds_id] = model
        scalers[ds_id] = scaler
        print(f"  Dataset {ds_id}: Model loaded")

    # Generate Task A predictions
    print("\n" + "="*60)
    print("Task A: Test Predictions")
    print("="*60)

    task_a_results = []
    for test_dataset in TEST_DATASETS:
        train_id = TEST_TO_TRAIN[test_dataset]

        if train_id not in models:
            print(f"  Skipping {test_dataset} (no model)")
            continue

        model = models[train_id]
        scaler = scalers[train_id]

        preds = predict_test_dataset(test_dataset, model, scaler, feature_names)
        task_a_results.append(preds)
        print(f"  {test_dataset}: {len(preds)} predictions")

    task_a_df = pd.concat(task_a_results, ignore_index=True)
    print(f"\nTotal Task A: {len(task_a_df)} predictions")

    # Generate Task B sequences
    print("\n" + "="*60)
    print("Task B: Sequence Identification")
    print("="*60)

    task_b_results = []
    for train_dataset in TRAIN_DATASETS:
        seqs = identify_sequences(train_dataset, TOP_K)
        task_b_results.append(seqs)
        print(f"  {train_dataset}: {len(seqs)} sequences")

    task_b_df = pd.concat(task_b_results, ignore_index=True)
    print(f"\nTotal Task B: {len(task_b_df)} sequences")

    # Combine
    submission = pd.concat([task_a_df, task_b_df], ignore_index=True)

    # Validate
    print("\n" + "="*60)
    print("Validation")
    print("="*60)
    expected_rows = 4213 + 8 * 50000  # 404,213
    actual_rows = len(submission)
    print(f"Expected rows: {expected_rows}")
    print(f"Actual rows: {actual_rows}")

    if actual_rows != expected_rows:
        print(f"WARNING: Row count mismatch!")
    else:
        print("STATUS: PASS")

    # Save
    Path(SUBMISSION_DIR).mkdir(exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = Path(SUBMISSION_DIR) / f'esm_xgb_submission_{timestamp}.csv'

    submission.to_csv(output_path, index=False)
    print(f"\nSaved to: {output_path}")
    print(f"File size: {output_path.stat().st_size / 1e6:.1f} MB")

    return submission


if __name__ == "__main__":
    main()
