#!/usr/bin/env python3
"""
🏆 AIRR-ML-25 Championship V2 - K-mer Neural Network
=====================================================
Improved architecture using k-mer features + neural network

Key Improvements over V1:
1. K-mer features (k=3,4,5) instead of raw AA encoding
2. Per-repertoire aggregation (sum, mean, max, std)
3. Deep MLP with residual connections
4. Better regularization and training

Target: Achieve variability in predictions and beat baseline 0.72866
"""

import os
import gc
import json
import random
import warnings
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict, Counter
from itertools import product

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import pickle

warnings.filterwarnings('ignore')

# ============================================================================
# Configuration
# ============================================================================
class Config:
    """Hyperparameters optimized for AIRR-ML-25"""

    # Paths
    TRAIN_ROOT = './data/train_datasets/train_datasets'
    TEST_ROOT = './data/test_datasets/test_datasets'
    CHECKPOINT_DIR = './checkpoints_v2'
    SUBMISSION_DIR = './submissions'

    # K-mer configuration - smaller k values to reduce memory
    K_VALUES = [3, 4]  # Multi-scale k-mers (k=5 too large)

    # Feature hashing - reduce k-mer dimension using hashing trick
    USE_FEATURE_HASHING = True
    HASH_DIM = 4096  # Fixed dimension for each k value

    # Feature extraction
    MAX_SEQS_PER_REP = 5000  # Reduced for memory efficiency

    # Model architecture
    HIDDEN_DIMS = [512, 256, 128]  # Deeper MLP
    DROPOUT = 0.4

    # Training
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-3
    NUM_EPOCHS = 100
    EARLY_STOPPING = 15

    # Task B
    TOP_K_SEQUENCES = 50000

    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Random seed
    SEED = 42

config = Config()

# Set random seeds
random.seed(config.SEED)
np.random.seed(config.SEED)
torch.manual_seed(config.SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(config.SEED)

print(f"🚀 Device: {config.DEVICE}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ============================================================================
# K-mer Feature Extraction
# ============================================================================

AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'

def hash_kmer(kmer: str, hash_dim: int) -> int:
    """Hash a k-mer to a fixed dimension using MurmurHash-like hashing."""
    # Simple hash function
    h = hash(kmer) % hash_dim
    return h

def count_kmers_hashed(sequence: str, k: int, hash_dim: int) -> np.ndarray:
    """Count k-mers using feature hashing."""
    counts = np.zeros(hash_dim, dtype=np.float32)
    seq = sequence.upper()
    for i in range(len(seq) - k + 1):
        kmer = seq[i:i+k]
        # Only count valid k-mers (all standard AAs)
        if all(c in AMINO_ACIDS for c in kmer):
            idx = hash_kmer(kmer, hash_dim)
            counts[idx] += 1
    return counts

def extract_kmer_features(sequences: List[str], k_values: List[int],
                           use_hashing: bool = True, hash_dim: int = 4096) -> np.ndarray:
    """
    Extract multi-scale k-mer features for a repertoire.

    Returns aggregated statistics: sum, mean, max, std for each k-mer.
    Uses feature hashing for memory efficiency.
    """
    all_features = []

    for k in k_values:
        # Use feature hashing
        vocab_size = hash_dim

        # Count k-mers for each sequence using hashing
        kmer_matrix = np.zeros((len(sequences), vocab_size), dtype=np.float32)
        for i, seq in enumerate(sequences):
            kmer_matrix[i] = count_kmers_hashed(seq, k, hash_dim)

        # Normalize by sequence length
        seq_lengths = np.array([max(len(s) - k + 1, 1) for s in sequences]).reshape(-1, 1)
        kmer_freq = kmer_matrix / seq_lengths

        # Aggregate across sequences - only keep most useful stats
        feat_mean = kmer_freq.mean(axis=0)
        feat_std = kmer_freq.std(axis=0)
        feat_max = kmer_freq.max(axis=0)

        all_features.extend([feat_mean, feat_std, feat_max])

    return np.concatenate(all_features)

def compute_cdr3_stats(sequences: List[str]) -> np.ndarray:
    """Compute CDR3 sequence statistics."""
    lengths = np.array([len(s) for s in sequences])

    stats = [
        len(sequences),  # Number of sequences
        lengths.mean(),
        lengths.std(),
        lengths.min(),
        lengths.max(),
        np.percentile(lengths, 25),
        np.percentile(lengths, 50),
        np.percentile(lengths, 75),
    ]

    # AA composition for the whole repertoire
    aa_counts = Counter(''.join(sequences))
    total = sum(aa_counts.values())
    for aa in AMINO_ACIDS:
        stats.append(aa_counts.get(aa, 0) / max(total, 1))

    return np.array(stats, dtype=np.float32)


# ============================================================================
# Feature Extraction Pipeline
# ============================================================================

def extract_repertoire_features(data_dir: str, filename: str,
                                 max_seqs: int = 5000,
                                 k_values: List[int] = [3, 4],
                                 hash_dim: int = 4096) -> np.ndarray:
    """Extract all features for a single repertoire."""
    file_path = Path(data_dir) / filename
    df = pd.read_csv(file_path, sep='\t', usecols=['junction_aa'])
    sequences = df['junction_aa'].dropna().astype(str).tolist()

    # Filter valid sequences (only keep those with valid AA characters)
    valid_chars = set(AMINO_ACIDS + 'X*-.')
    sequences = [s for s in sequences if len(s) >= 5 and all(c.upper() in valid_chars for c in s)]

    if len(sequences) == 0:
        return None

    # Sample if too many
    if len(sequences) > max_seqs:
        random.seed(42)
        sequences = random.sample(sequences, max_seqs)

    # Extract features with memory-efficient hashing
    kmer_features = extract_kmer_features(sequences, k_values, use_hashing=True, hash_dim=hash_dim)
    cdr3_stats = compute_cdr3_stats(sequences)

    return np.concatenate([kmer_features, cdr3_stats])


def extract_all_features(data_dir: str, metadata: pd.DataFrame,
                          max_seqs: int = 5000) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Extract features for all repertoires in a dataset."""
    features_list = []
    labels_list = []
    ids_list = []

    for idx, row in tqdm(metadata.iterrows(), total=len(metadata), desc="Extracting features"):
        feat = extract_repertoire_features(data_dir, row['filename'], max_seqs)
        if feat is not None:
            features_list.append(feat)
            labels_list.append(row.get('label_positive', 0))
            ids_list.append(row.get('repertoire_id', row['filename'].replace('.tsv', '')))

    return np.vstack(features_list), np.array(labels_list), ids_list


# ============================================================================
# Neural Network Model
# ============================================================================

class ResidualBlock(nn.Module):
    """Residual block with skip connection."""
    def __init__(self, dim: int, dropout: float = 0.3):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(F.gelu(x + self.layers(x)))


class KmerClassifier(nn.Module):
    """Deep MLP classifier for k-mer features."""

    def __init__(self, input_dim: int, hidden_dims: List[int] = [512, 256, 128],
                 dropout: float = 0.4):
        super().__init__()

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_dims[0], dropout)
            for _ in range(2)
        ])

        # Dimension reduction
        layers = []
        prev_dim = hidden_dims[0]
        for dim in hidden_dims[1:]:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.LayerNorm(dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            prev_dim = dim

        self.reduction = nn.Sequential(*layers)

        # Output layer
        self.classifier = nn.Linear(hidden_dims[-1], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)

        for block in self.residual_blocks:
            x = block(x)

        x = self.reduction(x)
        logit = self.classifier(x)

        return logit.squeeze(-1)


# ============================================================================
# Dataset
# ============================================================================

class FeatureDataset(Dataset):
    """Dataset for pre-computed features."""

    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.from_numpy(features).float()
        self.labels = torch.from_numpy(labels).float()

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# ============================================================================
# Training Functions
# ============================================================================

def train_epoch(model: nn.Module, dataloader: DataLoader,
                optimizer: torch.optim.Optimizer, criterion: nn.Module,
                device: torch.device, scaler: GradScaler) -> Tuple[float, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    for features, labels in dataloader:
        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        with autocast():
            logits = model(features)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_preds.extend(probs)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    auc = roc_auc_score(all_labels, all_preds) if len(set(all_labels)) > 1 else 0.5

    return avg_loss, auc


def evaluate(model: nn.Module, dataloader: DataLoader,
             criterion: nn.Module, device: torch.device) -> Tuple[float, float, List, List]:
    """Evaluate model."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for features, labels in dataloader:
            features = features.to(device)
            labels = labels.to(device)

            logits = model(features)
            loss = criterion(logits, labels)

            total_loss += loss.item()

            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.extend(probs)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    auc = roc_auc_score(all_labels, all_preds) if len(set(all_labels)) > 1 else 0.5

    return avg_loss, auc, all_preds, all_labels


# ============================================================================
# Main Training Pipeline
# ============================================================================

def train_single_dataset(dataset_name: str, train_features: np.ndarray,
                          train_labels: np.ndarray, val_features: np.ndarray,
                          val_labels: np.ndarray, config: Config) -> Tuple[nn.Module, float, StandardScaler]:
    """Train model on a single dataset."""

    print(f"\n{'='*60}")
    print(f"🎯 Training on {dataset_name}")
    print(f"{'='*60}")
    print(f"Train: {len(train_features)} | Val: {len(val_features)}")
    print(f"Feature dim: {train_features.shape[1]}")

    # Normalize features
    scaler = StandardScaler()
    train_features_norm = scaler.fit_transform(train_features)
    val_features_norm = scaler.transform(val_features)

    # Create datasets
    train_dataset = FeatureDataset(train_features_norm, train_labels)
    val_dataset = FeatureDataset(val_features_norm, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE,
                               shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE,
                             shuffle=False, num_workers=4, pin_memory=True)

    # Initialize model
    input_dim = train_features.shape[1]
    model = KmerClassifier(input_dim, config.HIDDEN_DIMS, config.DROPOUT).to(config.DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE,
                                   weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.NUM_EPOCHS)
    criterion = nn.BCEWithLogitsLoss()
    scaler_amp = GradScaler()

    # Training loop
    best_auc = 0
    patience_counter = 0
    best_model_state = None

    for epoch in range(config.NUM_EPOCHS):
        train_loss, train_auc = train_epoch(
            model, train_loader, optimizer, criterion, config.DEVICE, scaler_amp
        )
        val_loss, val_auc, _, _ = evaluate(model, val_loader, criterion, config.DEVICE)

        scheduler.step()

        if (epoch + 1) % 10 == 0 or val_auc > best_auc:
            print(f"Epoch {epoch+1:3d} | Train AUC: {train_auc:.4f} | Val AUC: {val_auc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if patience_counter >= config.EARLY_STOPPING:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        model = model.to(config.DEVICE)

    print(f"✅ Best Val AUC: {best_auc:.4f}")

    return model, best_auc, scaler


def main():
    """Full championship pipeline."""

    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║  🏆 AIRR-ML-25 Championship V2 - K-mer Neural Network 🏆        ║
    ║                                                                  ║
    ║  Architecture: Multi-scale K-mer + Deep MLP with Residuals     ║
    ║  Target: Beat baseline 0.72866 → Top position                   ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)

    models = {}
    scalers = {}
    results = []

    train_root = Path(config.TRAIN_ROOT)

    # ========== Step 1: Train models per dataset ==========
    print("\n" + "="*70)
    print("🏆 PHASE 1: FEATURE EXTRACTION & TRAINING")
    print("="*70)

    for dataset_dir in sorted(train_root.iterdir()):
        if not dataset_dir.is_dir():
            continue

        dataset_name = dataset_dir.name
        metadata_path = dataset_dir / 'metadata.csv'

        if not metadata_path.exists():
            continue

        print(f"\n📂 Processing {dataset_name}...")

        # Load metadata
        metadata = pd.read_csv(metadata_path)

        # Extract features
        features, labels, ids = extract_all_features(
            str(dataset_dir), metadata, config.MAX_SEQS_PER_REP
        )

        # Split for validation
        n_val = max(1, int(len(features) * 0.2))
        indices = np.random.permutation(len(features))
        val_idx = indices[:n_val]
        train_idx = indices[n_val:]

        train_features = features[train_idx]
        train_labels = labels[train_idx]
        val_features = features[val_idx]
        val_labels = labels[val_idx]

        # Train model
        model, val_auc, scaler = train_single_dataset(
            dataset_name, train_features, train_labels,
            val_features, val_labels, config
        )

        models[dataset_name] = model
        scalers[dataset_name] = scaler
        results.append({'dataset': dataset_name, 'val_auc': val_auc})

        # Save checkpoint
        os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
        checkpoint_path = os.path.join(config.CHECKPOINT_DIR, f'{dataset_name}_model.pt')
        torch.save({
            'model_state_dict': model.state_dict(),
            'input_dim': features.shape[1],
            'val_auc': val_auc
        }, checkpoint_path)

        scaler_path = os.path.join(config.CHECKPOINT_DIR, f'{dataset_name}_scaler.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)

        print(f"💾 Saved: {checkpoint_path}")

        gc.collect()
        torch.cuda.empty_cache()

    # Summary
    print("\n" + "="*70)
    print("📈 TRAINING SUMMARY")
    print("="*70)
    for r in results:
        print(f"  {r['dataset']}: Val AUC = {r['val_auc']:.4f}")

    mean_auc = np.mean([r['val_auc'] for r in results])
    print(f"\n🎯 Mean Val AUC: {mean_auc:.4f}")

    # ========== Step 2: Generate Task A predictions ==========
    print("\n" + "="*70)
    print("🔮 PHASE 2: TASK A PREDICTIONS")
    print("="*70)

    test_root = Path(config.TEST_ROOT)
    all_predictions = []

    # Mapping test datasets to training datasets
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

    for test_dir in sorted(test_root.iterdir()):
        if not test_dir.is_dir():
            continue

        test_name = test_dir.name
        train_name = test_to_train.get(test_name)

        if train_name is None or train_name not in models:
            print(f"⚠️ No model for {test_name}, using train_dataset_1")
            train_name = 'train_dataset_1'

        model = models[train_name]
        scaler = scalers[train_name]
        model.eval()

        # Get test files
        tsv_files = sorted(test_dir.glob('*.tsv'))
        print(f"\n📂 Predicting {test_name}: {len(tsv_files)} repertoires")

        for tsv_file in tqdm(tsv_files, desc=f"  {test_name}", leave=False):
            repertoire_id = tsv_file.stem

            try:
                # Extract features
                feat = extract_repertoire_features(
                    str(test_dir), tsv_file.name, config.MAX_SEQS_PER_REP
                )

                if feat is None:
                    prob = 0.5
                else:
                    # Normalize and predict
                    feat_norm = scaler.transform(feat.reshape(1, -1))
                    feat_tensor = torch.from_numpy(feat_norm).float().to(config.DEVICE)

                    with torch.no_grad():
                        logit = model(feat_tensor)
                        prob = torch.sigmoid(logit).item()

                all_predictions.append({
                    'ID': repertoire_id,
                    'dataset': test_name,
                    'label_positive_probability': prob,
                    'junction_aa': -999.0,
                    'v_call': -999.0,
                    'j_call': -999.0
                })

            except Exception as e:
                print(f"    ⚠️ Error {repertoire_id}: {e}")
                all_predictions.append({
                    'ID': repertoire_id,
                    'dataset': test_name,
                    'label_positive_probability': 0.5,
                    'junction_aa': -999.0,
                    'v_call': -999.0,
                    'j_call': -999.0
                })

    task_a_df = pd.DataFrame(all_predictions)
    print(f"\n✅ Generated {len(task_a_df)} Task A predictions")

    # Check prediction variability
    print("\n📊 Prediction variability:")
    for ds in sorted(task_a_df['dataset'].unique()):
        probs = task_a_df[task_a_df['dataset']==ds]['label_positive_probability']
        print(f"  {ds}: min={probs.min():.4f}, max={probs.max():.4f}, std={probs.std():.4f}")

    # ========== Step 3: Task B - Sequence identification ==========
    print("\n" + "="*70)
    print("🧬 PHASE 3: TASK B - SEQUENCE IDENTIFICATION")
    print("="*70)

    all_sequences = []

    for dataset_dir in sorted(train_root.iterdir()):
        if not dataset_dir.is_dir():
            continue

        dataset_name = dataset_dir.name
        print(f"\n📂 Processing {dataset_name}...")

        metadata_path = dataset_dir / 'metadata.csv'
        metadata = pd.read_csv(metadata_path)

        # Get positive samples
        positive_samples = metadata[metadata['label_positive'] == True]

        # Collect k-mer scores from positive samples
        kmer_scores = defaultdict(lambda: {'score': 0, 'count': 0})
        sequence_info = {}  # Store v_call, j_call for sequences

        for idx, row in tqdm(positive_samples.iterrows(), total=len(positive_samples),
                              desc=f"  {dataset_name}", leave=False):
            try:
                file_path = dataset_dir / row['filename']
                df = pd.read_csv(file_path, sep='\t')

                sequences = df['junction_aa'].dropna().astype(str).tolist()
                v_calls = df['v_call'].tolist() if 'v_call' in df.columns else [None] * len(sequences)
                j_calls = df['j_call'].tolist() if 'j_call' in df.columns else [None] * len(sequences)

                # Score sequences by their unique k-mers
                for seq, v, j in zip(sequences, v_calls, j_calls):
                    # Use 4-mers for scoring
                    vocab = get_kmer_vocab(4)
                    seq_kmers = set()
                    seq_upper = seq.upper()
                    for i in range(len(seq_upper) - 3):
                        kmer = seq_upper[i:i+4]
                        if kmer in vocab:
                            seq_kmers.add(kmer)

                    # Score based on number of unique k-mers
                    score = len(seq_kmers)
                    kmer_scores[seq]['score'] += score
                    kmer_scores[seq]['count'] += 1

                    if seq not in sequence_info:
                        sequence_info[seq] = {'v_call': v, 'j_call': j}

            except Exception as e:
                continue

        # Sort by average score and select top 50,000
        sorted_seqs = sorted(
            kmer_scores.items(),
            key=lambda x: x[1]['score'] / max(x[1]['count'], 1),
            reverse=True
        )[:config.TOP_K_SEQUENCES]

        # Create rows
        for rank, (junction_aa, info) in enumerate(sorted_seqs, 1):
            seq_info = sequence_info.get(junction_aa, {})
            v_call = seq_info.get('v_call')
            j_call = seq_info.get('j_call')

            all_sequences.append({
                'ID': f'{dataset_name}_seq_top_{rank}',
                'dataset': dataset_name,
                'label_positive_probability': -999.0,
                'junction_aa': junction_aa,
                'v_call': v_call if v_call and pd.notna(v_call) else 'TRBV20-1',
                'j_call': j_call if j_call and pd.notna(j_call) else 'TRBJ2-7'
            })

        print(f"  ✓ Selected {len(sorted_seqs)} sequences")

    task_b_df = pd.DataFrame(all_sequences)
    print(f"\n✅ Generated {len(task_b_df)} Task B sequences")

    # ========== Step 4: Generate submission ==========
    print("\n" + "="*70)
    print("📝 PHASE 4: GENERATING SUBMISSION")
    print("="*70)

    submission = pd.concat([task_a_df, task_b_df], ignore_index=True)
    submission = submission[['ID', 'dataset', 'label_positive_probability',
                              'junction_aa', 'v_call', 'j_call']]

    # Validate
    expected_rows = 4213 + 8 * 50000
    actual_rows = len(submission)

    print(f"  Expected: {expected_rows}")
    print(f"  Actual: {actual_rows}")

    if actual_rows != expected_rows:
        print(f"  ⚠️ Row count mismatch!")
    else:
        print(f"  ✓ Row count matches!")

    # Save
    os.makedirs(config.SUBMISSION_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(config.SUBMISSION_DIR, f'v2_kmer_submission_{timestamp}.csv')
    submission.to_csv(output_path, index=False)

    print(f"\n💾 Saved: {output_path}")
    print(f"   Size: {os.path.getsize(output_path) / 1e6:.2f} MB")

    # Save results
    with open(os.path.join(config.CHECKPOINT_DIR, 'training_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*70)
    print("🏆 CHAMPIONSHIP V2 PIPELINE COMPLETE!")
    print("="*70)
    print(f"\n📈 Mean Val AUC: {mean_auc:.4f}")
    print(f"\nSubmit with:")
    print(f"  kaggle competitions submit -c adaptive-immune-profiling-challenge-2025 \\")
    print(f"    -f {output_path} -m 'V2: K-mer + Deep MLP'")


if __name__ == '__main__':
    main()
