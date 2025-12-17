#!/usr/bin/env python3
"""
🏆 AIRR-ML-25 Championship FAST - Optimized Pipeline
=====================================================
Focus: Speed + Robustness + Generalization

Key Design:
- Simple, interpretable features (CDR3 length, AA composition, k-mer TF-IDF)
- PCA dimensionality reduction
- Small neural network with heavy regularization
- Fast execution for rapid iteration

Target: Generate valid predictions with variability
"""

import os
import gc
import json
import random
import warnings
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import Counter
import hashlib

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tqdm import tqdm
import pickle

warnings.filterwarnings('ignore')

# ============================================================================
# Configuration
# ============================================================================
TRAIN_ROOT = './data/train_datasets/train_datasets'
TEST_ROOT = './data/test_datasets/test_datasets'
CHECKPOINT_DIR = './checkpoints_fast'
SUBMISSION_DIR = './submissions'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

print(f"🚀 Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")

AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'

# ============================================================================
# Feature Extraction (Fast & Robust)
# ============================================================================

def extract_simple_features(sequences: List[str]) -> np.ndarray:
    """
    Extract simple, robust features for a repertoire.

    Features:
    - CDR3 length statistics (8)
    - AA composition (20)
    - Simple positional features (40)
    - K-mer hashed features (512)

    Total: ~580 features
    """
    features = []

    if len(sequences) == 0:
        return np.zeros(580, dtype=np.float32)

    # 1. CDR3 length statistics
    lengths = np.array([len(s) for s in sequences])
    features.extend([
        len(sequences),
        lengths.mean(),
        lengths.std(),
        lengths.min(),
        lengths.max(),
        np.percentile(lengths, 25),
        np.percentile(lengths, 50),
        np.percentile(lengths, 75),
    ])

    # 2. AA composition (global)
    aa_counts = Counter(''.join(sequences).upper())
    total = max(sum(aa_counts.values()), 1)
    for aa in AMINO_ACIDS:
        features.append(aa_counts.get(aa, 0) / total)

    # 3. Positional AA frequencies (first 10 and last 10 positions)
    max_pos = 10
    for pos in range(max_pos):
        pos_counts = Counter()
        for seq in sequences:
            if pos < len(seq):
                pos_counts[seq[pos].upper()] += 1
        total_pos = max(sum(pos_counts.values()), 1)
        for aa in AMINO_ACIDS:
            features.append(pos_counts.get(aa, 0) / total_pos)

    for pos in range(1, max_pos + 1):
        pos_counts = Counter()
        for seq in sequences:
            if pos <= len(seq):
                pos_counts[seq[-pos].upper()] += 1
        total_pos = max(sum(pos_counts.values()), 1)
        for aa in AMINO_ACIDS:
            features.append(pos_counts.get(aa, 0) / total_pos)

    # 4. K-mer hashed features (3-mers, hashed to 512 dims)
    hash_dim = 512
    kmer_counts = np.zeros(hash_dim, dtype=np.float32)

    for seq in sequences:
        seq = seq.upper()
        for i in range(len(seq) - 2):
            kmer = seq[i:i+3]
            if all(c in AMINO_ACIDS for c in kmer):
                h = int(hashlib.md5(kmer.encode()).hexdigest(), 16) % hash_dim
                kmer_counts[h] += 1

    # Normalize
    total_kmers = max(kmer_counts.sum(), 1)
    kmer_freq = kmer_counts / total_kmers
    features.extend(kmer_freq.tolist())

    return np.array(features, dtype=np.float32)


def load_repertoire_sequences(data_dir: str, filename: str, max_seqs: int = 5000) -> List[str]:
    """Load sequences from a repertoire file."""
    file_path = Path(data_dir) / filename
    df = pd.read_csv(file_path, sep='\t', usecols=['junction_aa'])
    sequences = df['junction_aa'].dropna().astype(str).tolist()

    # Filter valid sequences
    valid_chars = set(AMINO_ACIDS + 'X*-.')
    sequences = [s for s in sequences if len(s) >= 5 and len(s) <= 30 and all(c.upper() in valid_chars for c in s)]

    if len(sequences) > max_seqs:
        random.seed(42)
        sequences = random.sample(sequences, max_seqs)

    return sequences


# ============================================================================
# Neural Network (Simple & Robust)
# ============================================================================

class SimpleClassifier(nn.Module):
    """Small neural network with heavy regularization."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.5):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ============================================================================
# Training
# ============================================================================

def train_model(X_train: np.ndarray, y_train: np.ndarray,
                X_val: np.ndarray, y_val: np.ndarray,
                n_epochs: int = 200, lr: float = 1e-3,
                weight_decay: float = 1e-2, patience: int = 20) -> Tuple[nn.Module, StandardScaler, float]:
    """Train a model on the given data."""

    # Normalize
    scaler = StandardScaler()
    X_train_norm = scaler.fit_transform(X_train)
    X_val_norm = scaler.transform(X_val)

    # Convert to tensors
    X_train_t = torch.from_numpy(X_train_norm).float().to(DEVICE)
    y_train_t = torch.from_numpy(y_train).float().to(DEVICE)
    X_val_t = torch.from_numpy(X_val_norm).float().to(DEVICE)
    y_val_t = torch.from_numpy(y_val).float().to(DEVICE)

    # Model
    input_dim = X_train.shape[1]
    model = SimpleClassifier(input_dim).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    best_auc = 0
    best_state = None
    no_improve = 0

    for epoch in range(n_epochs):
        # Train
        model.train()
        optimizer.zero_grad()
        logits = model(X_train_t)
        loss = criterion(logits, y_train_t)
        loss.backward()
        optimizer.step()

        # Eval
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_probs = torch.sigmoid(val_logits).cpu().numpy()

        try:
            val_auc = roc_auc_score(y_val, val_probs)
        except:
            val_auc = 0.5

        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            break

    if best_state:
        model.load_state_dict(best_state)
        model = model.to(DEVICE)

    return model, scaler, best_auc


# ============================================================================
# Main Pipeline
# ============================================================================

def main():
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║  🏆 AIRR-ML-25 Championship FAST Pipeline 🏆                    ║
    ║                                                                  ║
    ║  Architecture: Simple Features + Small NN + Heavy Regularization║
    ║  Focus: Speed + Robustness + Prediction Variability             ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    train_root = Path(TRAIN_ROOT)
    test_root = Path(TEST_ROOT)

    models = {}
    scalers = {}
    results = []

    # ========== Phase 1: Train models ==========
    print("\n" + "="*70)
    print("🏆 PHASE 1: TRAINING")
    print("="*70)

    for dataset_dir in sorted(train_root.iterdir()):
        if not dataset_dir.is_dir():
            continue

        dataset_name = dataset_dir.name
        metadata_path = dataset_dir / 'metadata.csv'

        if not metadata_path.exists():
            continue

        print(f"\n📂 {dataset_name}")

        # Load metadata
        metadata = pd.read_csv(metadata_path)

        # Extract features
        print("  Extracting features...")
        features_list = []
        labels_list = []

        for idx, row in tqdm(metadata.iterrows(), total=len(metadata), leave=False):
            seqs = load_repertoire_sequences(str(dataset_dir), row['filename'])
            if seqs:
                feat = extract_simple_features(seqs)
                features_list.append(feat)
                labels_list.append(row.get('label_positive', 0))

        X = np.vstack(features_list)
        y = np.array(labels_list, dtype=np.float32)

        # Split
        n_val = max(1, int(len(X) * 0.2))
        idx = np.random.permutation(len(X))
        val_idx, train_idx = idx[:n_val], idx[n_val:]

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        print(f"  Train: {len(X_train)} | Val: {len(X_val)} | Features: {X.shape[1]}")

        # Train
        model, scaler, val_auc = train_model(X_train, y_train, X_val, y_val)

        models[dataset_name] = model
        scalers[dataset_name] = scaler
        results.append({'dataset': dataset_name, 'val_auc': val_auc})

        print(f"  ✅ Val AUC: {val_auc:.4f}")

        # Save
        torch.save({
            'model_state_dict': model.state_dict(),
            'input_dim': X.shape[1],
            'val_auc': val_auc
        }, os.path.join(CHECKPOINT_DIR, f'{dataset_name}_model.pt'))

        with open(os.path.join(CHECKPOINT_DIR, f'{dataset_name}_scaler.pkl'), 'wb') as f:
            pickle.dump(scaler, f)

        gc.collect()
        torch.cuda.empty_cache()

    # Summary
    print("\n" + "="*70)
    print("📈 TRAINING SUMMARY")
    print("="*70)
    for r in results:
        print(f"  {r['dataset']}: {r['val_auc']:.4f}")
    mean_auc = np.mean([r['val_auc'] for r in results])
    print(f"\n🎯 Mean Val AUC: {mean_auc:.4f}")

    # ========== Phase 2: Predictions ==========
    print("\n" + "="*70)
    print("🔮 PHASE 2: TEST PREDICTIONS")
    print("="*70)

    test_to_train = {
        'test_dataset_1': 'train_dataset_1',
        'test_dataset_2': 'train_dataset_2',
        'test_dataset_3': 'train_dataset_3',
        'test_dataset_4': 'train_dataset_4',
        'test_dataset_5': 'train_dataset_5',
        'test_dataset_6': 'train_dataset_6',
        'test_dataset_7_1': 'train_dataset_7',
        'test_dataset_7_2': 'train_dataset_7',
        'test_dataset_8_1': 'train_dataset_8',
        'test_dataset_8_2': 'train_dataset_8',
        'test_dataset_8_3': 'train_dataset_8',
    }

    all_predictions = []

    for test_dir in sorted(test_root.iterdir()):
        if not test_dir.is_dir():
            continue

        test_name = test_dir.name
        train_name = test_to_train.get(test_name, 'train_dataset_1')

        if train_name not in models:
            train_name = 'train_dataset_1'

        model = models[train_name]
        scaler = scalers[train_name]
        model.eval()

        tsv_files = sorted(test_dir.glob('*.tsv'))
        print(f"\n📂 {test_name}: {len(tsv_files)} repertoires")

        for tsv_file in tqdm(tsv_files, leave=False):
            rep_id = tsv_file.stem

            try:
                seqs = load_repertoire_sequences(str(test_dir), tsv_file.name)
                if not seqs:
                    prob = 0.5
                else:
                    feat = extract_simple_features(seqs)
                    feat_norm = scaler.transform(feat.reshape(1, -1))
                    feat_t = torch.from_numpy(feat_norm).float().to(DEVICE)

                    with torch.no_grad():
                        logit = model(feat_t)
                        prob = torch.sigmoid(logit).item()

                all_predictions.append({
                    'ID': rep_id,
                    'dataset': test_name,
                    'label_positive_probability': prob,
                    'junction_aa': -999.0,
                    'v_call': -999.0,
                    'j_call': -999.0
                })
            except Exception as e:
                all_predictions.append({
                    'ID': rep_id,
                    'dataset': test_name,
                    'label_positive_probability': 0.5,
                    'junction_aa': -999.0,
                    'v_call': -999.0,
                    'j_call': -999.0
                })

    task_a_df = pd.DataFrame(all_predictions)
    print(f"\n✅ Task A: {len(task_a_df)} predictions")

    # Check variability
    print("\n📊 Prediction variability:")
    for ds in sorted(task_a_df['dataset'].unique()):
        probs = task_a_df[task_a_df['dataset']==ds]['label_positive_probability']
        print(f"  {ds}: min={probs.min():.4f}, max={probs.max():.4f}, std={probs.std():.4f}")

    # ========== Phase 3: Task B ==========
    print("\n" + "="*70)
    print("🧬 PHASE 3: TASK B - SEQUENCE IDENTIFICATION")
    print("="*70)

    all_sequences = []

    for dataset_dir in sorted(train_root.iterdir()):
        if not dataset_dir.is_dir():
            continue

        dataset_name = dataset_dir.name
        metadata = pd.read_csv(dataset_dir / 'metadata.csv')
        positive_samples = metadata[metadata['label_positive'] == True]

        # Collect sequences from positive samples
        seq_counts = Counter()
        seq_info = {}

        for idx, row in positive_samples.iterrows():
            try:
                df = pd.read_csv(dataset_dir / row['filename'], sep='\t')
                for _, r in df.iterrows():
                    seq = str(r['junction_aa'])
                    if 5 <= len(seq) <= 30:
                        seq_counts[seq] += 1
                        if seq not in seq_info:
                            seq_info[seq] = {
                                'v_call': r.get('v_call', 'TRBV20-1'),
                                'j_call': r.get('j_call', 'TRBJ2-7')
                            }
            except:
                continue

        # Top 50000 sequences
        top_seqs = seq_counts.most_common(50000)

        for rank, (junction_aa, count) in enumerate(top_seqs, 1):
            info = seq_info.get(junction_aa, {})
            all_sequences.append({
                'ID': f'{dataset_name}_seq_top_{rank}',
                'dataset': dataset_name,
                'label_positive_probability': -999.0,
                'junction_aa': junction_aa,
                'v_call': info.get('v_call', 'TRBV20-1') if pd.notna(info.get('v_call')) else 'TRBV20-1',
                'j_call': info.get('j_call', 'TRBJ2-7') if pd.notna(info.get('j_call')) else 'TRBJ2-7'
            })

        print(f"  {dataset_name}: {len(top_seqs)} sequences")

    task_b_df = pd.DataFrame(all_sequences)
    print(f"\n✅ Task B: {len(task_b_df)} sequences")

    # ========== Phase 4: Generate submission ==========
    print("\n" + "="*70)
    print("📝 PHASE 4: SUBMISSION")
    print("="*70)

    submission = pd.concat([task_a_df, task_b_df], ignore_index=True)
    submission = submission[['ID', 'dataset', 'label_positive_probability', 'junction_aa', 'v_call', 'j_call']]

    expected = 4213 + 8 * 50000
    print(f"  Expected: {expected}")
    print(f"  Actual: {len(submission)}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(SUBMISSION_DIR, f'fast_submission_{timestamp}.csv')
    submission.to_csv(output_path, index=False)

    print(f"\n💾 Saved: {output_path}")
    print(f"   Size: {os.path.getsize(output_path) / 1e6:.2f} MB")

    print("\n" + "="*70)
    print("🏆 FAST PIPELINE COMPLETE!")
    print("="*70)
    print(f"\n📈 Mean Val AUC: {mean_auc:.4f}")
    print(f"\nSubmit:")
    print(f"  kaggle competitions submit -c adaptive-immune-profiling-challenge-2025 \\")
    print(f"    -f {output_path} -m 'Fast: Simple features + Small NN'")


if __name__ == '__main__':
    main()
