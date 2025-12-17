#!/usr/bin/env python3
"""
AIRR-ML-25 Ultimate Deep Learning Pipeline V2
Optimized for GB10 with 128GB unified memory

Strategy:
1. Phase 1: K-mer baseline (fast, sets foundation) - target 0.68
2. Phase 2: ESM-2 embeddings + MIL (deep learning) - target 0.75
3. Phase 3: Ensemble (XGBoost + MIL) - target 0.80+
"""

import os
import sys
import gc
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict, Counter
import pickle
import hashlib
import time

# ML libraries
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

# Deep learning
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler

# ============================================================================
# Configuration
# ============================================================================
class Config:
    # Paths
    TRAIN_ROOT = './data/train_datasets'
    TEST_ROOT = './data/test_datasets'
    OUTPUT_DIR = './outputs'
    CACHE_DIR = './cache'

    # K-mer parameters
    KMER_K = [3, 4]
    TOP_KMERS = 5000  # Select top k-mers

    # ESM-2 parameters (for Phase 2)
    ESM_LAYER = 15
    ESM_DIM = 1280
    REDUCED_DIM = 128
    MAX_SEQS_SAMPLE = 500  # Sample per repertoire for ESM-2

    # MIL parameters
    HIDDEN_DIM = 256
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-4
    EPOCHS = 30
    PATIENCE = 7

    # Task B
    TOP_K_SEQUENCES = 50000

    # Device
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

config = Config()

os.makedirs(config.OUTPUT_DIR, exist_ok=True)
os.makedirs(config.CACHE_DIR, exist_ok=True)

# ============================================================================
# Utility Functions
# ============================================================================
def get_cache_path(name):
    return os.path.join(config.CACHE_DIR, f"{name}.pkl")

def save_cache(data, name):
    with open(get_cache_path(name), 'wb') as f:
        pickle.dump(data, f)

def load_cache(name):
    path = get_cache_path(name)
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    return None

# ============================================================================
# Data Loading
# ============================================================================
def load_metadata(dataset_dir):
    """Load metadata.csv from dataset directory"""
    metadata_path = Path(dataset_dir) / 'metadata.csv'
    if metadata_path.exists():
        df = pd.read_csv(metadata_path)
        # Convert label_positive to 0/1
        if 'label_positive' in df.columns:
            df['label'] = df['label_positive'].apply(lambda x: 1 if x in [True, 'True', 1] else 0)
        return df
    return None

def load_repertoire_sequences(tsv_path):
    """Load sequences from TSV file"""
    df = pd.read_csv(tsv_path, sep='\t')
    return df

def get_dataset_info():
    """Get information about all datasets"""
    info = {'train': {}, 'test': {}}

    # Training datasets
    for i in range(1, 9):
        dataset_dir = Path(config.TRAIN_ROOT) / f'train_dataset_{i}'
        if dataset_dir.exists():
            metadata = load_metadata(dataset_dir)
            tsv_files = list(dataset_dir.glob('*.tsv'))
            info['train'][i] = {
                'dir': dataset_dir,
                'n_repertoires': len(tsv_files),
                'metadata': metadata
            }

    # Test datasets
    for test_dir in Path(config.TEST_ROOT).iterdir():
        if test_dir.is_dir() and test_dir.name.startswith('test_dataset_'):
            name = test_dir.name
            tsv_files = list(test_dir.glob('*.tsv'))
            info['test'][name] = {
                'dir': test_dir,
                'n_repertoires': len(tsv_files)
            }

    return info

# ============================================================================
# Phase 1: K-mer Feature Extraction
# ============================================================================
def compute_kmer_counts(sequences, k_values=[3, 4]):
    """Compute k-mer counts for sequences"""
    counts = Counter()
    for seq in sequences:
        if pd.isna(seq) or len(seq) < max(k_values):
            continue
        for k in k_values:
            for i in range(len(seq) - k + 1):
                kmer = seq[i:i+k]
                if 'X' not in kmer and '*' not in kmer:  # Skip unknown/stop
                    counts[kmer] += 1
    return counts

def build_kmer_vocabulary(dataset_info, k_values=[3, 4], top_k=5000):
    """Build vocabulary of top k-mers across all training data"""
    print("Building k-mer vocabulary...")

    cached = load_cache('kmer_vocab')
    if cached is not None:
        print(f"  Loaded from cache: {len(cached)} k-mers")
        return cached

    global_counts = Counter()

    for dataset_id, dataset in dataset_info['train'].items():
        print(f"  Processing Dataset {dataset_id}...")
        tsv_files = list(dataset['dir'].glob('*.tsv'))

        for tsv_file in tqdm(tsv_files[:50], desc=f"Dataset {dataset_id}"):  # Sample 50 repertoires
            df = load_repertoire_sequences(tsv_file)
            if 'junction_aa' in df.columns:
                counts = compute_kmer_counts(df['junction_aa'].tolist(), k_values)
                global_counts.update(counts)

    # Select top k-mers
    vocab = [kmer for kmer, _ in global_counts.most_common(top_k)]
    print(f"  Built vocabulary: {len(vocab)} k-mers")

    save_cache(vocab, 'kmer_vocab')
    return vocab

def extract_kmer_features(sequences, vocab):
    """Extract k-mer frequency features using vocabulary"""
    counts = Counter()
    total = 0

    for seq in sequences:
        if pd.isna(seq):
            continue
        for k in [3, 4]:
            if len(seq) >= k:
                for i in range(len(seq) - k + 1):
                    kmer = seq[i:i+k]
                    if kmer in vocab:
                        counts[kmer] += 1
                        total += 1

    if total == 0:
        return np.zeros(len(vocab))

    features = np.array([counts.get(kmer, 0) / total for kmer in vocab])
    return features

def extract_all_kmer_features(dataset_info, vocab):
    """Extract k-mer features for all repertoires"""
    print("\nExtracting k-mer features...")

    all_features = {'train': {}, 'test': {}}
    all_labels = {}

    # Training data
    for dataset_id, dataset in dataset_info['train'].items():
        cache_name = f'kmer_train_{dataset_id}'
        cached = load_cache(cache_name)

        if cached is not None:
            all_features['train'][dataset_id] = cached['features']
            all_labels[dataset_id] = cached['labels']
            print(f"  Dataset {dataset_id}: loaded from cache")
            continue

        print(f"  Processing Train Dataset {dataset_id}...")
        metadata = dataset['metadata']
        tsv_files = list(dataset['dir'].glob('*.tsv'))

        features_list = []
        labels_list = []
        rep_ids_list = []

        for tsv_file in tqdm(tsv_files, desc=f"Dataset {dataset_id}"):
            rep_id = tsv_file.stem
            df = load_repertoire_sequences(tsv_file)

            if 'junction_aa' not in df.columns:
                continue

            # Extract features
            feats = extract_kmer_features(df['junction_aa'].tolist(), vocab)
            features_list.append(feats)
            rep_ids_list.append(rep_id)

            # Get label
            label = None
            if metadata is not None:
                match = metadata[metadata['repertoire_id'] == rep_id]
                if len(match) > 0 and 'label' in match.columns:
                    label = match['label'].values[0]
            labels_list.append(label)

        features_array = np.vstack(features_list)
        all_features['train'][dataset_id] = {
            'features': features_array,
            'rep_ids': rep_ids_list
        }
        all_labels[dataset_id] = labels_list

        save_cache({
            'features': all_features['train'][dataset_id],
            'labels': labels_list
        }, cache_name)

        print(f"    {len(features_list)} repertoires, {features_array.shape[1]} features")

    # Test data
    for dataset_name, dataset in dataset_info['test'].items():
        cache_name = f'kmer_test_{dataset_name}'
        cached = load_cache(cache_name)

        if cached is not None:
            all_features['test'][dataset_name] = cached
            print(f"  {dataset_name}: loaded from cache")
            continue

        print(f"  Processing {dataset_name}...")
        tsv_files = list(dataset['dir'].glob('*.tsv'))

        features_list = []
        rep_ids_list = []

        for tsv_file in tqdm(tsv_files, desc=dataset_name):
            rep_id = tsv_file.stem
            df = load_repertoire_sequences(tsv_file)

            if 'junction_aa' not in df.columns:
                continue

            feats = extract_kmer_features(df['junction_aa'].tolist(), vocab)
            features_list.append(feats)
            rep_ids_list.append(rep_id)

        features_array = np.vstack(features_list)
        all_features['test'][dataset_name] = {
            'features': features_array,
            'rep_ids': rep_ids_list
        }

        save_cache(all_features['test'][dataset_name], cache_name)
        print(f"    {len(features_list)} repertoires")

    return all_features, all_labels

# ============================================================================
# Phase 1: Logistic Regression Baseline
# ============================================================================
def train_logreg_baseline(all_features, all_labels):
    """Train logistic regression baseline with LODO CV"""
    print("\n" + "="*60)
    print("Phase 1: Logistic Regression Baseline (LODO CV)")
    print("="*60)

    results = []
    oof_preds = {}

    for val_dataset in sorted(all_features['train'].keys()):
        # Prepare train/val
        X_train, y_train = [], []
        for dataset_id in all_features['train'].keys():
            if dataset_id != val_dataset:
                feats = all_features['train'][dataset_id]['features']
                labels = all_labels[dataset_id]
                for f, l in zip(feats, labels):
                    if l is not None:
                        X_train.append(f)
                        y_train.append(l)

        X_val = all_features['train'][val_dataset]['features']
        y_val = [l for l in all_labels[val_dataset] if l is not None]
        rep_ids_val = all_features['train'][val_dataset]['rep_ids']

        if len(X_train) == 0 or len(y_val) == 0:
            continue

        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_val_filtered = np.array([f for f, l in zip(X_val, all_labels[val_dataset]) if l is not None])

        # Standardize
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val_filtered)

        # Train
        model = LogisticRegression(max_iter=1000, C=0.1, solver='lbfgs')
        model.fit(X_train_scaled, y_train)

        # Predict
        y_pred = model.predict_proba(X_val_scaled)[:, 1]
        auc = roc_auc_score(y_val, y_pred)

        results.append({
            'val_dataset': val_dataset,
            'auc': auc,
            'n_train': len(X_train),
            'n_val': len(y_val)
        })

        oof_preds[val_dataset] = y_pred
        print(f"  Fold {val_dataset}: AUC = {auc:.4f} (n={len(y_val)})")

    mean_auc = np.mean([r['auc'] for r in results])
    print(f"\n  LODO CV Mean AUC: {mean_auc:.4f}")

    return results, oof_preds, mean_auc

# ============================================================================
# Phase 2: ESM-2 + MIL (Optional - requires fair-esm)
# ============================================================================
class AttentionMIL(nn.Module):
    """Attention-based MIL aggregation"""

    def __init__(self, input_dim=128, hidden_dim=256, dropout=0.3):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Gated attention
        self.attn_v = nn.Linear(hidden_dim, hidden_dim // 2)
        self.attn_u = nn.Linear(hidden_dim, hidden_dim // 2)
        self.attn_w = nn.Linear(hidden_dim // 2, 1)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x, mask=None):
        # x: (batch, n_seqs, input_dim)
        h = self.encoder(x)  # (batch, n_seqs, hidden_dim)

        # Gated attention
        v = torch.tanh(self.attn_v(h))
        u = torch.sigmoid(self.attn_u(h))
        a = self.attn_w(v * u)  # (batch, n_seqs, 1)

        if mask is not None:
            a = a.masked_fill(~mask.unsqueeze(-1), float('-inf'))

        attn = F.softmax(a, dim=1)

        # Aggregate
        z = (h * attn).sum(dim=1)  # (batch, hidden_dim)

        # Classify
        logits = self.classifier(z)

        return logits, attn.squeeze(-1)

def try_load_esm():
    """Try to load ESM-2 model"""
    try:
        import esm
        print("Loading ESM-2 model...")
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        model = model.to(config.DEVICE).eval()
        batch_converter = alphabet.get_batch_converter()
        print("  ESM-2 loaded successfully")
        return model, alphabet, batch_converter
    except ImportError:
        print("  ESM-2 not available (install with: pip install fair-esm)")
        return None, None, None
    except Exception as e:
        print(f"  ESM-2 loading failed: {e}")
        return None, None, None

# ============================================================================
# XGBoost Training
# ============================================================================
def train_xgboost(all_features, all_labels):
    """Train XGBoost with LODO CV"""
    print("\n" + "="*60)
    print("Phase 2: XGBoost (LODO CV)")
    print("="*60)

    try:
        import xgboost as xgb
    except ImportError:
        print("  XGBoost not available")
        return None, None, 0.0

    results = []
    oof_preds = {}

    for val_dataset in sorted(all_features['train'].keys()):
        # Prepare data
        X_train, y_train = [], []
        for dataset_id in all_features['train'].keys():
            if dataset_id != val_dataset:
                feats = all_features['train'][dataset_id]['features']
                labels = all_labels[dataset_id]
                for f, l in zip(feats, labels):
                    if l is not None:
                        X_train.append(f)
                        y_train.append(l)

        X_val = all_features['train'][val_dataset]['features']
        y_val = [l for l in all_labels[val_dataset] if l is not None]

        if len(X_train) == 0 or len(y_val) == 0:
            continue

        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_val_filtered = np.array([f for f, l in zip(X_val, all_labels[val_dataset]) if l is not None])

        # Train XGBoost
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'max_depth': 6,
            'learning_rate': 0.05,
            'n_estimators': 300,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'tree_method': 'hist',
            'verbosity': 0
        }

        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val_filtered, np.array(y_val))],
            verbose=False
        )

        y_pred = model.predict_proba(X_val_filtered)[:, 1]
        auc = roc_auc_score(y_val, y_pred)

        results.append({
            'val_dataset': val_dataset,
            'auc': auc,
            'n_train': len(X_train),
            'n_val': len(y_val)
        })

        oof_preds[val_dataset] = y_pred
        print(f"  Fold {val_dataset}: AUC = {auc:.4f}")

    mean_auc = np.mean([r['auc'] for r in results])
    print(f"\n  LODO CV Mean AUC: {mean_auc:.4f}")

    return results, oof_preds, mean_auc

# ============================================================================
# Final Model Training & Prediction
# ============================================================================
def train_final_model(all_features, all_labels):
    """Train final XGBoost model on all data"""
    print("\n" + "="*60)
    print("Training Final Model")
    print("="*60)

    try:
        import xgboost as xgb
    except ImportError:
        print("  Using LogisticRegression (XGBoost not available)")
        return train_final_logreg(all_features, all_labels)

    # Combine all training data
    X_train, y_train = [], []
    for dataset_id in all_features['train'].keys():
        feats = all_features['train'][dataset_id]['features']
        labels = all_labels[dataset_id]
        for f, l in zip(feats, labels):
            if l is not None:
                X_train.append(f)
                y_train.append(l)

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    print(f"  Training samples: {len(X_train)}")
    print(f"  Positive ratio: {y_train.mean():.3f}")

    # Train XGBoost
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': 6,
        'learning_rate': 0.05,
        'n_estimators': 500,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'tree_method': 'hist',
        'verbosity': 0
    }

    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, verbose=True)

    return model

def train_final_logreg(all_features, all_labels):
    """Fallback: Train final LogisticRegression"""
    X_train, y_train = [], []
    for dataset_id in all_features['train'].keys():
        feats = all_features['train'][dataset_id]['features']
        labels = all_labels[dataset_id]
        for f, l in zip(feats, labels):
            if l is not None:
                X_train.append(f)
                y_train.append(l)

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = LogisticRegression(max_iter=1000, C=0.1)
    model.fit(X_train_scaled, y_train)

    return model, scaler

def predict_test(model, all_features):
    """Generate predictions for test data"""
    print("\n" + "="*60)
    print("Generating Test Predictions")
    print("="*60)

    predictions = {}

    for dataset_name, data in all_features['test'].items():
        X_test = data['features']
        rep_ids = data['rep_ids']

        y_pred = model.predict_proba(X_test)[:, 1]

        for rep_id, prob in zip(rep_ids, y_pred):
            predictions[(dataset_name, rep_id)] = prob

        print(f"  {dataset_name}: {len(rep_ids)} predictions")

    return predictions

# ============================================================================
# Task B: Important Sequence Selection
# ============================================================================
def select_important_sequences(dataset_info, vocab, model, top_k=50000):
    """Select important sequences for Task B based on k-mer importance"""
    print("\n" + "="*60)
    print("Task B: Selecting Important Sequences")
    print("="*60)

    task_b_results = {}

    # Get feature importance from model
    try:
        importance = model.feature_importances_
    except:
        importance = np.ones(len(vocab)) / len(vocab)

    # Create k-mer importance dict
    kmer_importance = {kmer: imp for kmer, imp in zip(vocab, importance)}

    for dataset_id, dataset in dataset_info['train'].items():
        print(f"\n  Processing Dataset {dataset_id}...")

        metadata = dataset['metadata']
        if metadata is None:
            continue

        # Only use positive samples
        positive_reps = metadata[metadata['label'] == 1]['repertoire_id'].tolist()

        all_sequences = []
        all_scores = []

        for rep_id in tqdm(positive_reps, desc=f"Dataset {dataset_id}"):
            tsv_path = dataset['dir'] / f"{rep_id}.tsv"
            if not tsv_path.exists():
                continue

            df = load_repertoire_sequences(tsv_path)

            if 'junction_aa' not in df.columns:
                continue

            for _, row in df.iterrows():
                seq = row['junction_aa']
                v_call = row.get('v_call', '')
                j_call = row.get('j_call', '')

                if pd.isna(seq):
                    continue

                # Score sequence by k-mer importance
                score = 0
                count = 0
                for k in [3, 4]:
                    if len(seq) >= k:
                        for i in range(len(seq) - k + 1):
                            kmer = seq[i:i+k]
                            if kmer in kmer_importance:
                                score += kmer_importance[kmer]
                                count += 1

                if count > 0:
                    score /= count
                    all_sequences.append({
                        'junction_aa': seq,
                        'v_call': v_call if not pd.isna(v_call) else '',
                        'j_call': j_call if not pd.isna(j_call) else ''
                    })
                    all_scores.append(score)

        # Deduplicate and select top-k
        seq_scores = defaultdict(list)
        for seq, score in zip(all_sequences, all_scores):
            key = (seq['junction_aa'], seq['v_call'], seq['j_call'])
            seq_scores[key].append(score)

        unique_seqs = []
        unique_scores = []
        for key, scores in seq_scores.items():
            unique_seqs.append({
                'junction_aa': key[0],
                'v_call': key[1],
                'j_call': key[2]
            })
            unique_scores.append(np.mean(scores))

        # Sort and select top-k
        sorted_indices = np.argsort(unique_scores)[::-1][:top_k]
        task_b_results[dataset_id] = [unique_seqs[i] for i in sorted_indices]

        print(f"    Selected {len(task_b_results[dataset_id])} sequences")

    return task_b_results

# ============================================================================
# Submission Generation
# ============================================================================
def generate_submission(test_predictions, task_b_results, output_path):
    """Generate final submission file"""
    print("\n" + "="*60)
    print("Generating Submission File")
    print("="*60)

    rows = []

    # Task A: Test predictions
    for (dataset_name, rep_id), prob in test_predictions.items():
        rows.append({
            'ID': rep_id,
            'dataset': dataset_name,
            'label_positive_probability': prob,
            'junction_aa': -999.0,
            'v_call': -999.0,
            'j_call': -999.0
        })

    # Task B: Important sequences
    for dataset_id, sequences in task_b_results.items():
        for rank, seq in enumerate(sequences):
            rows.append({
                'ID': f'train_dataset_{dataset_id}_seq_top_{rank+1}',
                'dataset': f'train_dataset_{dataset_id}',
                'label_positive_probability': -999.0,
                'junction_aa': seq['junction_aa'],
                'v_call': seq['v_call'] if seq['v_call'] else -999.0,
                'j_call': seq['j_call'] if seq['j_call'] else -999.0
            })

    df = pd.DataFrame(rows)

    # Verify row count
    expected_rows = len(test_predictions) + sum(len(seqs) for seqs in task_b_results.values())
    print(f"  Task A predictions: {len(test_predictions)}")
    print(f"  Task B sequences: {sum(len(seqs) for seqs in task_b_results.values())}")
    print(f"  Total rows: {len(df)}")

    df.to_csv(output_path, index=False)
    print(f"  Saved to: {output_path}")

    return df

# ============================================================================
# Main Pipeline
# ============================================================================
def main():
    print("="*60)
    print("AIRR-ML-25 Ultimate Deep Learning Pipeline V2")
    print("="*60)

    start_time = time.time()

    print(f"\nDevice: {config.DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA: {torch.version.cuda}")

    # Get dataset info
    dataset_info = get_dataset_info()
    print(f"\nDatasets:")
    print(f"  Training: {len(dataset_info['train'])} datasets")
    for i, info in sorted(dataset_info['train'].items()):
        print(f"    Dataset {i}: {info['n_repertoires']} repertoires")
    print(f"  Test: {len(dataset_info['test'])} datasets")

    # Phase 1: Build k-mer vocabulary
    vocab = build_kmer_vocabulary(dataset_info, config.KMER_K, config.TOP_KMERS)

    # Extract k-mer features
    all_features, all_labels = extract_all_kmer_features(dataset_info, vocab)

    # Phase 1: Logistic Regression baseline
    lr_results, lr_preds, lr_auc = train_logreg_baseline(all_features, all_labels)

    # Phase 2: XGBoost
    xgb_results, xgb_preds, xgb_auc = train_xgboost(all_features, all_labels)

    # Train final model
    final_model = train_final_model(all_features, all_labels)

    # Generate test predictions
    test_predictions = predict_test(final_model, all_features)

    # Task B: Select important sequences
    task_b_results = select_important_sequences(dataset_info, vocab, final_model, config.TOP_K_SEQUENCES)

    # Generate submission
    submission_path = os.path.join(config.OUTPUT_DIR, 'submission_ultimate_v2.csv')
    generate_submission(test_predictions, task_b_results, submission_path)

    # Summary
    elapsed = time.time() - start_time
    print("\n" + "="*60)
    print("Pipeline Complete!")
    print("="*60)
    print(f"  LogReg CV AUC: {lr_auc:.4f}")
    print(f"  XGBoost CV AUC: {xgb_auc:.4f}")
    print(f"  Time elapsed: {elapsed/60:.1f} minutes")
    print(f"  Submission: {submission_path}")

    return {
        'logreg_auc': lr_auc,
        'xgboost_auc': xgb_auc,
        'submission_path': submission_path
    }

if __name__ == '__main__':
    main()
