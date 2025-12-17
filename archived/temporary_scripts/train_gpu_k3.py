#!/usr/bin/env python3
"""
GPU-Optimized Training with k=3 k-mers + XGBoost
================================================
使用 k=3 特徵 (8,000 維度) 而非 k=4 (160,000 維度)
以確保可以在 GPU 上訓練而不會 OOM

關鍵優化：
1. k=3 k-mers: 20^3 = 8,000 特徵 (vs k=4 的 160,000)
2. GPU XGBoost: 使用 tree_method='hist', device='cuda'
3. 額外特徵: V/J gene usage, CDR3 length 統計
4. 記憶體高效: 使用 sparse matrices
"""

import os
import sys
import gc
import warnings
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
from scipy import sparse
from tqdm import tqdm
from collections import Counter
from itertools import product

# XGBoost with GPU support
import xgboost as xgb

warnings.filterwarnings('ignore')

# =============================================================================
# Configuration
# =============================================================================

K = 3  # k-mer size (REDUCED from 4 to 3 for GPU memory)
TRAIN_DIR = "./data/train_datasets/train_datasets/train_dataset_8"
TEST_DIRS = [
    "./data/test_datasets/test_datasets/test_dataset_8_1",
    "./data/test_datasets/test_datasets/test_dataset_8_2",
    "./data/test_datasets/test_datasets/test_dataset_8_3",
]
OUT_DIR = "./results_k4"  # Keep same output directory for consistency
TOP_K = 50000

# Verify paths exist before training
for test_dir in TEST_DIRS:
    assert os.path.exists(test_dir), f"Test directory not found: {test_dir}"
assert os.path.exists(TRAIN_DIR), f"Training directory not found: {TRAIN_DIR}"

# XGBoost GPU parameters
XGB_PARAMS = {
    'tree_method': 'hist',  # GPU-optimized histogram method
    'device': 'cuda',       # Use GPU
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth': 6,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'gamma': 0.1,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': 42,
}

NUM_BOOST_ROUND = 200

# =============================================================================
# k-mer Utilities
# =============================================================================

AA = 'ACDEFGHIKLMNPQRSTVWY'

def generate_kmer_index(k: int = K) -> Dict[str, int]:
    """Generate k-mer to index mapping."""
    kmers = [''.join(p) for p in product(AA, repeat=k)]
    return {kmer: i for i, kmer in enumerate(kmers)}

def extract_kmers(seq: str, k: int = K) -> List[str]:
    """Extract all k-mers from a sequence."""
    return [seq[i:i+k] for i in range(len(seq) - k + 1)]

def kmer_features(sequences: List[str], kmer_index: Dict[str, int], k: int = K) -> sparse.csr_matrix:
    """
    Convert sequences to k-mer count features (sparse matrix).

    Returns:
        sparse.csr_matrix of shape (1, vocab_size) containing k-mer counts
    """
    vocab_size = len(kmer_index)
    kmer_counts = Counter()

    for seq in sequences:
        if pd.isna(seq) or not isinstance(seq, str):
            continue
        for kmer in extract_kmers(seq, k):
            if kmer in kmer_index:
                kmer_counts[kmer] += 1

    # Build sparse matrix
    indices = []
    data = []
    for kmer, count in kmer_counts.items():
        indices.append(kmer_index[kmer])
        data.append(count)

    row = sparse.csr_matrix(
        (data, ([0] * len(data), indices)),
        shape=(1, vocab_size),
        dtype=np.float32
    )

    return row

# =============================================================================
# Additional Features
# =============================================================================

def extract_vj_features(df: pd.DataFrame) -> Dict[str, float]:
    """Extract V and J gene usage features."""
    features = {}

    # V gene usage
    if 'v_call' in df.columns:
        v_genes = df['v_call'].dropna()
        v_counts = v_genes.value_counts()
        for gene, count in v_counts.head(20).items():  # Top 20 V genes
            features[f'v_{gene}'] = count / len(df)

    # J gene usage
    if 'j_call' in df.columns:
        j_genes = df['j_call'].dropna()
        j_counts = j_genes.value_counts()
        for gene, count in j_counts.head(20).items():  # Top 20 J genes
            features[f'j_{gene}'] = count / len(df)

    return features

def extract_cdr3_features(sequences: List[str]) -> Dict[str, float]:
    """Extract CDR3 length and composition features."""
    lengths = [len(seq) for seq in sequences if isinstance(seq, str) and not pd.isna(seq)]

    features = {}
    if lengths:
        features['cdr3_mean_length'] = np.mean(lengths)
        features['cdr3_std_length'] = np.std(lengths)
        features['cdr3_min_length'] = np.min(lengths)
        features['cdr3_max_length'] = np.max(lengths)
        features['cdr3_median_length'] = np.median(lengths)

    return features

# =============================================================================
# Data Loading
# =============================================================================

def load_repertoire(file_path: str, kmer_index: Dict[str, int]) -> Tuple[sparse.csr_matrix, Dict[str, float]]:
    """
    Load a single repertoire and extract features.

    Returns:
        - k-mer features (sparse matrix)
        - additional features (dict)
    """
    try:
        df = pd.read_csv(file_path, sep='\t')
        sequences = df['junction_aa'].dropna().tolist()

        # k-mer features
        kmer_feats = kmer_features(sequences, kmer_index, K)

        # Additional features
        vj_feats = extract_vj_features(df)
        cdr3_feats = extract_cdr3_features(sequences)

        # Combine additional features
        add_feats = {**vj_feats, **cdr3_feats}

        return kmer_feats, add_feats

    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        vocab_size = len(kmer_index)
        return sparse.csr_matrix((1, vocab_size), dtype=np.float32), {}

def load_dataset(metadata_path: str, data_dir: str, kmer_index: Dict[str, int]) -> Tuple[sparse.csr_matrix, np.ndarray, List[str], pd.DataFrame]:
    """Load entire dataset."""
    metadata = pd.read_csv(metadata_path)

    kmer_features_list = []
    add_features_list = []
    labels = []
    filenames = []

    print(f"Loading {len(metadata)} repertoires...")
    for _, row in tqdm(metadata.iterrows(), total=len(metadata)):
        file_path = os.path.join(data_dir, row['filename'])
        kmer_feats, add_feats = load_repertoire(file_path, kmer_index)

        kmer_features_list.append(kmer_feats)
        add_features_list.append(add_feats)
        labels.append(int(row['label_positive']))
        filenames.append(row['filename'])

    # Stack k-mer features
    X_kmer = sparse.vstack(kmer_features_list)

    # Convert additional features to DataFrame and fill missing
    add_features_df = pd.DataFrame(add_features_list).fillna(0)

    # Combine k-mer and additional features
    X_combined = sparse.hstack([X_kmer, add_features_df.values])

    y = np.array(labels)

    return X_combined, y, filenames, add_features_df

# =============================================================================
# Training
# =============================================================================

def train_model(X: sparse.csr_matrix, y: np.ndarray) -> xgb.Booster:
    """Train XGBoost model with GPU."""
    print("\n=== Training XGBoost with GPU ===")
    print(f"Feature shape: {X.shape}")
    print(f"Label distribution: {np.bincount(y)}")

    # Create DMatrix (XGBoost's internal data structure)
    dtrain = xgb.DMatrix(X, label=y)

    # Train
    print(f"\nTraining with {NUM_BOOST_ROUND} rounds...")
    print(f"Using GPU: {XGB_PARAMS['device']}")

    model = xgb.train(
        XGB_PARAMS,
        dtrain,
        num_boost_round=NUM_BOOST_ROUND,
        verbose_eval=20
    )

    return model

# =============================================================================
# Prediction
# =============================================================================

def predict_test_set(model: xgb.Booster, test_dir: str, kmer_index: Dict[str, int], dataset_name: str) -> pd.DataFrame:
    """Generate predictions for test set."""
    metadata_path = os.path.join(test_dir, 'metadata.csv')

    kmer_features_list = []
    add_features_list = []
    repertoire_ids = []

    # Check if metadata.csv exists (Dataset 1-7) or not (Dataset 8)
    if os.path.exists(metadata_path):
        # Standard format with metadata.csv
        metadata = pd.read_csv(metadata_path)
        print(f"\n=== Predicting {dataset_name} ({len(metadata)} repertoires) ===")

        for _, row in tqdm(metadata.iterrows(), total=len(metadata)):
            file_path = os.path.join(test_dir, row['filename'])
            kmer_feats, add_feats = load_repertoire(file_path, kmer_index)

            kmer_features_list.append(kmer_feats)
            add_features_list.append(add_feats)
            repertoire_ids.append(row['repertoire_id'])
    else:
        # Dataset 8 format: no metadata.csv, direct .tsv files
        tsv_files = sorted([f for f in os.listdir(test_dir) if f.endswith('.tsv')])
        print(f"\n=== Predicting {dataset_name} ({len(tsv_files)} repertoires) ===")

        for filename in tqdm(tsv_files):
            file_path = os.path.join(test_dir, filename)
            kmer_feats, add_feats = load_repertoire(file_path, kmer_index)

            kmer_features_list.append(kmer_feats)
            add_features_list.append(add_feats)
            # Use filename without .tsv as ID
            repertoire_ids.append(filename.replace('.tsv', ''))

    # Stack features
    X_kmer = sparse.vstack(kmer_features_list)
    add_features_df = pd.DataFrame(add_features_list).fillna(0)
    X_combined = sparse.hstack([X_kmer, add_features_df.values])

    # Predict
    dtest = xgb.DMatrix(X_combined)
    predictions = model.predict(dtest)

    # Format output
    results = pd.DataFrame({
        'ID': repertoire_ids,
        'dataset': dataset_name,
        'label_positive_probability': predictions,
        'junction_aa': -999.0,
        'v_call': -999.0,
        'j_call': -999.0
    })

    return results

# =============================================================================
# Sequence Identification
# =============================================================================

def identify_important_sequences(
    model: xgb.Booster,
    train_dir: str,
    kmer_index: Dict[str, int],
    dataset_name: str,
    top_k: int = TOP_K
) -> pd.DataFrame:
    """Identify top-k important sequences using feature importance."""
    print(f"\n=== Identifying important sequences for {dataset_name} ===")

    # Get feature importance (gain)
    importance_dict = model.get_score(importance_type='gain')

    # Filter for k-mer features only (first len(kmer_index) features)
    kmer_importance = {}
    vocab_size = len(kmer_index)
    for feat, score in importance_dict.items():
        feat_idx = int(feat.replace('f', ''))
        if feat_idx < vocab_size:
            kmer_importance[feat_idx] = score

    # Sort k-mers by importance
    sorted_kmers = sorted(kmer_importance.items(), key=lambda x: x[1], reverse=True)

    # Map back to k-mer strings
    idx_to_kmer = {v: k for k, v in kmer_index.items()}
    important_kmers = [idx_to_kmer[idx] for idx, _ in sorted_kmers[:top_k]]

    print(f"Top 10 important {K}-mers: {important_kmers[:10]}")

    # Load training data to find sequences containing these k-mers
    metadata_path = os.path.join(train_dir, 'metadata.csv')
    metadata = pd.read_csv(metadata_path)

    # Collect sequences containing important k-mers
    important_sequences = []

    for _, row in tqdm(metadata.iterrows(), total=len(metadata), desc="Scanning sequences"):
        file_path = os.path.join(train_dir, row['filename'])
        try:
            df = pd.read_csv(file_path, sep='\t')

            for _, seq_row in df.iterrows():
                seq = seq_row['junction_aa']
                if pd.isna(seq) or not isinstance(seq, str):
                    continue

                # Check if sequence contains any important k-mer
                seq_kmers = set(extract_kmers(seq, K))
                if seq_kmers & set(important_kmers):
                    important_sequences.append({
                        'junction_aa': seq,
                        'v_call': seq_row.get('v_call', -999.0),
                        'j_call': seq_row.get('j_call', -999.0),
                        'label': row['label_positive']
                    })
        except Exception as e:
            continue

    # Convert to DataFrame and select top-k
    seq_df = pd.DataFrame(important_sequences)

    # Prioritize sequences from positive class
    pos_seqs = seq_df[seq_df['label'] == 1]
    neg_seqs = seq_df[seq_df['label'] == 0]

    # Take 70% from positive, 30% from negative
    n_pos = min(int(top_k * 0.7), len(pos_seqs))
    n_neg = min(top_k - n_pos, len(neg_seqs))

    selected = pd.concat([
        pos_seqs.sample(n=n_pos, random_state=42) if n_pos > 0 else pd.DataFrame(),
        neg_seqs.sample(n=n_neg, random_state=42) if n_neg > 0 else pd.DataFrame()
    ]).reset_index(drop=True)

    # Format output
    selected['ID'] = [f'{dataset_name}_seq_top_{i+1}' for i in range(len(selected))]
    selected['dataset'] = dataset_name
    selected['label_positive_probability'] = -999.0

    result = selected[['ID', 'dataset', 'label_positive_probability', 'junction_aa', 'v_call', 'j_call']].head(top_k)

    print(f"Selected {len(result)} important sequences")

    return result

# =============================================================================
# Main Pipeline
# =============================================================================

def main():
    print("=" * 80)
    print("GPU-Optimized Training with k=3 k-mers")
    print("=" * 80)

    # Check GPU
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
                              capture_output=True, text=True, timeout=5)
        print(f"\nGPU Info:\n{result.stdout}")
    except:
        print("\n⚠️  Could not query GPU info")

    # Generate k-mer index
    print(f"\nGenerating {K}-mer index...")
    kmer_index = generate_kmer_index(K)
    print(f"Vocabulary size: {len(kmer_index):,}")
    print(f"Expected memory: ~{len(kmer_index) * 4 / 1e9:.2f} GB (much less than k=4's 10.35GB)")

    # Load training data
    print(f"\n=== Loading Training Data ===")
    metadata_path = os.path.join(TRAIN_DIR, 'metadata.csv')
    X_train, y_train, filenames, add_feats_df = load_dataset(metadata_path, TRAIN_DIR, kmer_index)

    print(f"\nTraining data loaded:")
    print(f"  - Samples: {X_train.shape[0]}")
    print(f"  - Features: {X_train.shape[1]:,} ({len(kmer_index):,} k-mers + {add_feats_df.shape[1]} additional)")
    print(f"  - Labels: {np.bincount(y_train)}")

    # Train model
    model = train_model(X_train, y_train)

    # Save model
    model_path = os.path.join(OUT_DIR, 'model_k3_gpu.json')
    model.save_model(model_path)
    print(f"\n✅ Model saved: {model_path}")

    # Generate predictions for test sets
    all_predictions = []
    for test_dir in TEST_DIRS:
        dataset_name = os.path.basename(test_dir).replace('test_dataset_', 'test_dataset_')
        preds = predict_test_set(model, test_dir, kmer_index, dataset_name)
        all_predictions.append(preds)

    # Save predictions
    final_preds = pd.concat(all_predictions, ignore_index=True)
    pred_path = os.path.join(OUT_DIR, 'train_dataset_8_test_predictions.tsv')
    final_preds.to_csv(pred_path, sep='\t', index=False)
    print(f"\n✅ Predictions saved: {pred_path}")
    print(f"   Total predictions: {len(final_preds)}")

    # Identify important sequences
    train_dataset_name = os.path.basename(TRAIN_DIR).replace('train_dataset_', 'train_dataset_')
    important_seqs = identify_important_sequences(model, TRAIN_DIR, kmer_index, train_dataset_name, TOP_K)

    seq_path = os.path.join(OUT_DIR, 'train_dataset_8_important_sequences.tsv')
    important_seqs.to_csv(seq_path, sep='\t', index=False)
    print(f"\n✅ Important sequences saved: {seq_path}")
    print(f"   Total sequences: {len(important_seqs)}")

    print("\n" + "=" * 80)
    print("✅ Dataset 8 Training Complete!")
    print("=" * 80)

if __name__ == '__main__':
    main()
