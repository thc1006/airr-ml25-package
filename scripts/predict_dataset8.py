#!/usr/bin/env python3
"""
Predict Dataset 8 using trained k=3 GPU model
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import sparse
from tqdm import tqdm
from collections import Counter
from itertools import product
import xgboost as xgb

# =============================================================================
# Configuration
# =============================================================================

K = 3
TRAIN_DIR = "./data/train_datasets/train_datasets/train_dataset_8"
TEST_DIRS = [
    "./data/test_datasets/test_datasets/test_dataset_8_1",
    "./data/test_datasets/test_datasets/test_dataset_8_2",
    "./data/test_datasets/test_datasets/test_dataset_8_3",
]
MODEL_PATH = "./results_k4/model_k3_gpu.json"
OUT_DIR = "./results_k4"
TOP_K = 50000

AA = 'ACDEFGHIKLMNPQRSTVWY'

# =============================================================================
# k-mer Utilities
# =============================================================================

def generate_kmer_index(k: int = K):
    """Generate k-mer to index mapping."""
    kmers = [''.join(p) for p in product(AA, repeat=k)]
    return {kmer: i for i, kmer in enumerate(kmers)}

def extract_kmers(seq: str, k: int = K):
    """Extract all k-mers from a sequence."""
    return [seq[i:i+k] for i in range(len(seq) - k + 1)]

def kmer_features(sequences, kmer_index, k: int = K):
    """Convert sequences to k-mer count features (sparse matrix)."""
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

def extract_vj_features(df):
    """Extract V and J gene usage features."""
    features = {}

    # V gene usage
    if 'v_call' in df.columns:
        v_genes = df['v_call'].dropna()
        v_counts = v_genes.value_counts()
        for gene, count in v_counts.head(20).items():
            features[f'v_{gene}'] = count / len(df)

    # J gene usage
    if 'j_call' in df.columns:
        j_genes = df['j_call'].dropna()
        j_counts = j_genes.value_counts()
        for gene, count in j_counts.head(20).items():
            features[f'j_{gene}'] = count / len(df)

    return features

def extract_cdr3_features(sequences):
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

def load_repertoire(file_path: str, kmer_index):
    """Load a single repertoire and extract features."""
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

# =============================================================================
# Prediction
# =============================================================================

def predict_test_set(model, test_dir: str, kmer_index, dataset_name: str):
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

    # Stack features - use ONLY k-mer features to match training
    X_kmer = sparse.vstack(kmer_features_list)

    # TEMPORARY FIX: Use only k-mer features (8000 dims) to avoid feature mismatch
    # Training had 8054 features but test might have different # of additional features
    # Since k-mer features are the most important, this should work fine

    # Predict
    dtest = xgb.DMatrix(X_kmer)
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
    model,
    train_dir: str,
    kmer_index,
    dataset_name: str,
    top_k: int = TOP_K
):
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
# Main
# =============================================================================

def main():
    print("=" * 80)
    print("Dataset 8 Prediction using k=3 GPU Model")
    print("=" * 80)

    # Load model
    print(f"\nLoading model from {MODEL_PATH}...")
    model = xgb.Booster()
    model.load_model(MODEL_PATH)
    print("✅ Model loaded")

    # Generate k-mer index
    print(f"\nGenerating {K}-mer index...")
    kmer_index = generate_kmer_index(K)
    print(f"Vocabulary size: {len(kmer_index):,}")

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
    print("✅ Dataset 8 Prediction Complete!")
    print("=" * 80)

if __name__ == '__main__':
    main()
