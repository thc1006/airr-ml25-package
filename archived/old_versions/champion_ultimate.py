#!/usr/bin/env python3
"""
🏆 AIRR-ML-25 Ultimate Championship Pipeline 🏆

This pipeline is designed to WIN by:
1. Using proven, biologically-meaningful features
2. Comprehensive caching for efficiency
3. Robust validation to ensure actual learning
4. Ensemble of multiple approaches

Key Features:
- V/J gene usage patterns (categorical)
- Clonality metrics (Shannon entropy, Gini, D50)
- K-mer features with proper hashing
- CDR3 length distribution
- Position-weighted amino acid features

Architecture:
- Phase 1: Feature extraction with caching
- Phase 2: Train multiple models per dataset
- Phase 3: Ensemble predictions
- Phase 4: Task B sequence identification
"""

import os
import sys
import pickle
import hashlib
import warnings
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import json

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Config:
    """Championship configuration with optimized parameters"""
    # Paths
    TRAIN_ROOT: str = "./data/train_datasets/train_datasets"
    TEST_ROOT: str = "./data/test_datasets/test_datasets"
    CACHE_DIR: str = "./cache_ultimate"
    CHECKPOINT_DIR: str = "./checkpoints_ultimate"

    # Feature extraction
    MAX_SEQS_PER_REP: int = 10000  # Maximum sequences per repertoire
    K_VALUES: Tuple[int, ...] = (3, 4)  # K-mer sizes
    HASH_DIM: int = 2048  # Dimension for hashed k-mers

    # V/J Gene settings
    TOP_V_GENES: int = 50  # Top V genes to track
    TOP_J_GENES: int = 15  # Top J genes to track

    # Training
    BATCH_SIZE: int = 32
    LEARNING_RATE: float = 1e-3
    WEIGHT_DECAY: float = 1e-2
    EPOCHS: int = 100
    PATIENCE: int = 20
    N_FOLDS: int = 5

    # Model
    HIDDEN_DIMS: Tuple[int, ...] = (512, 256, 128)
    DROPOUT: float = 0.4

    # Device
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Random seed
    SEED: int = 42

config = Config()

# Set seeds
np.random.seed(config.SEED)
torch.manual_seed(config.SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(config.SEED)

# Create directories
os.makedirs(config.CACHE_DIR, exist_ok=True)
os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

# ============================================================================
# CACHING UTILITIES
# ============================================================================

def get_cache_key(data_path: str, suffix: str = "") -> str:
    """Generate a unique cache key based on path and suffix"""
    key = f"{data_path}_{suffix}"
    return hashlib.md5(key.encode()).hexdigest()[:12]

def load_cache(cache_path: str) -> Optional[Any]:
    """Load cached data if exists"""
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except:
            return None
    return None

def save_cache(data: Any, cache_path: str):
    """Save data to cache"""
    with open(cache_path, 'wb') as f:
        pickle.dump(data, f)

# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

# Amino acid properties for physicochemical features
AA_PROPERTIES = {
    'A': {'hydrophobic': 1, 'charge': 0, 'size': 0},
    'C': {'hydrophobic': 1, 'charge': 0, 'size': 0},
    'D': {'hydrophobic': 0, 'charge': -1, 'size': 0},
    'E': {'hydrophobic': 0, 'charge': -1, 'size': 1},
    'F': {'hydrophobic': 1, 'charge': 0, 'size': 2},
    'G': {'hydrophobic': 1, 'charge': 0, 'size': -1},
    'H': {'hydrophobic': 0, 'charge': 1, 'size': 1},
    'I': {'hydrophobic': 1, 'charge': 0, 'size': 1},
    'K': {'hydrophobic': 0, 'charge': 1, 'size': 1},
    'L': {'hydrophobic': 1, 'charge': 0, 'size': 1},
    'M': {'hydrophobic': 1, 'charge': 0, 'size': 1},
    'N': {'hydrophobic': 0, 'charge': 0, 'size': 0},
    'P': {'hydrophobic': 0, 'charge': 0, 'size': 0},
    'Q': {'hydrophobic': 0, 'charge': 0, 'size': 1},
    'R': {'hydrophobic': 0, 'charge': 1, 'size': 2},
    'S': {'hydrophobic': 0, 'charge': 0, 'size': -1},
    'T': {'hydrophobic': 0, 'charge': 0, 'size': 0},
    'V': {'hydrophobic': 1, 'charge': 0, 'size': 0},
    'W': {'hydrophobic': 1, 'charge': 0, 'size': 2},
    'Y': {'hydrophobic': 1, 'charge': 0, 'size': 2},
}

AMINO_ACIDS = list('ACDEFGHIKLMNPQRSTVWY')
AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}

def hash_kmer(kmer: str, dim: int) -> int:
    """Hash a k-mer to a fixed dimension using MurmurHash-like approach"""
    h = hash(kmer)
    return abs(h) % dim

def extract_kmer_features(sequences: List[str], k: int, hash_dim: int) -> np.ndarray:
    """Extract hashed k-mer frequency features"""
    features = np.zeros(hash_dim, dtype=np.float32)
    total_kmers = 0

    for seq in sequences:
        seq = ''.join(c for c in seq.upper() if c in AMINO_ACIDS)
        for i in range(len(seq) - k + 1):
            kmer = seq[i:i+k]
            idx = hash_kmer(kmer, hash_dim)
            features[idx] += 1
            total_kmers += 1

    if total_kmers > 0:
        features /= total_kmers

    return features

def extract_aa_composition(sequences: List[str]) -> np.ndarray:
    """Extract amino acid composition features"""
    composition = np.zeros(20, dtype=np.float32)
    total = 0

    for seq in sequences:
        for aa in seq:
            if aa in AA_TO_IDX:
                composition[AA_TO_IDX[aa]] += 1
                total += 1

    if total > 0:
        composition /= total

    return composition

def extract_length_features(sequences: List[str]) -> np.ndarray:
    """Extract CDR3 length distribution features"""
    lengths = [len(s) for s in sequences if len(s) > 0]
    if not lengths:
        return np.zeros(10, dtype=np.float32)

    lengths = np.array(lengths)
    return np.array([
        np.mean(lengths),
        np.std(lengths),
        np.median(lengths),
        np.min(lengths),
        np.max(lengths),
        np.percentile(lengths, 25),
        np.percentile(lengths, 75),
        np.percentile(lengths, 10),
        np.percentile(lengths, 90),
        len(lengths)  # Number of valid sequences
    ], dtype=np.float32)

def extract_positional_features(sequences: List[str], n_positions: int = 5) -> np.ndarray:
    """Extract positional amino acid features (start and end)"""
    start_features = np.zeros((n_positions, 20), dtype=np.float32)
    end_features = np.zeros((n_positions, 20), dtype=np.float32)

    for seq in sequences:
        seq = ''.join(c for c in seq if c in AMINO_ACIDS)
        if len(seq) < 2:
            continue

        # Start positions
        for i in range(min(n_positions, len(seq))):
            if seq[i] in AA_TO_IDX:
                start_features[i, AA_TO_IDX[seq[i]]] += 1

        # End positions
        for i in range(min(n_positions, len(seq))):
            if seq[-(i+1)] in AA_TO_IDX:
                end_features[i, AA_TO_IDX[seq[-(i+1)]]] += 1

    # Normalize
    for i in range(n_positions):
        if start_features[i].sum() > 0:
            start_features[i] /= start_features[i].sum()
        if end_features[i].sum() > 0:
            end_features[i] /= end_features[i].sum()

    return np.concatenate([start_features.flatten(), end_features.flatten()])

def extract_clonality_features(sequences: List[str], counts: Optional[List[int]] = None) -> np.ndarray:
    """Extract clonality/diversity features"""
    if counts is None:
        counts = [1] * len(sequences)

    # Unique sequences and their frequencies
    seq_counts = Counter()
    for seq, cnt in zip(sequences, counts):
        seq_counts[seq] += cnt

    total = sum(seq_counts.values())
    if total == 0:
        return np.zeros(8, dtype=np.float32)

    frequencies = np.array(list(seq_counts.values()), dtype=np.float64) / total
    frequencies = frequencies[frequencies > 0]

    # Shannon entropy
    entropy = -np.sum(frequencies * np.log2(frequencies + 1e-10))

    # Normalized entropy
    max_entropy = np.log2(len(frequencies)) if len(frequencies) > 1 else 1
    norm_entropy = entropy / max_entropy if max_entropy > 0 else 0

    # Gini coefficient
    sorted_freq = np.sort(frequencies)
    n = len(sorted_freq)
    cumsum = np.cumsum(sorted_freq)
    gini = 1 - 2 * np.sum(cumsum) / (n * np.sum(sorted_freq)) if n > 0 and np.sum(sorted_freq) > 0 else 0

    # Clonality (1 - normalized entropy)
    clonality = 1 - norm_entropy

    # D50 (fraction of clones making up 50% of repertoire)
    sorted_freq_desc = np.sort(frequencies)[::-1]
    cumsum_desc = np.cumsum(sorted_freq_desc)
    d50_idx = np.searchsorted(cumsum_desc, 0.5)
    d50 = (d50_idx + 1) / len(sorted_freq_desc) if len(sorted_freq_desc) > 0 else 0

    # Number of unique clonotypes
    n_unique = len(seq_counts)

    # Top clone fraction
    top_fraction = max(frequencies) if len(frequencies) > 0 else 0

    # Richness (log of unique clones)
    richness = np.log10(n_unique + 1)

    return np.array([
        entropy, norm_entropy, gini, clonality, d50, n_unique / 1000, top_fraction, richness
    ], dtype=np.float32)

def extract_vj_features(v_calls: List[str], j_calls: List[str],
                        v_encoder: LabelEncoder, j_encoder: LabelEncoder) -> np.ndarray:
    """Extract V/J gene usage features"""
    # V gene frequencies
    v_freq = np.zeros(len(v_encoder.classes_), dtype=np.float32)
    for v in v_calls:
        if v in v_encoder.classes_:
            v_freq[np.where(v_encoder.classes_ == v)[0][0]] += 1
    if v_freq.sum() > 0:
        v_freq /= v_freq.sum()

    # J gene frequencies
    j_freq = np.zeros(len(j_encoder.classes_), dtype=np.float32)
    for j in j_calls:
        if j in j_encoder.classes_:
            j_freq[np.where(j_encoder.classes_ == j)[0][0]] += 1
    if j_freq.sum() > 0:
        j_freq /= j_freq.sum()

    return np.concatenate([v_freq, j_freq])

def extract_physicochemical_features(sequences: List[str]) -> np.ndarray:
    """Extract physicochemical property features"""
    hydrophobic_scores = []
    charge_scores = []
    size_scores = []

    for seq in sequences:
        h, c, s = 0, 0, 0
        n = 0
        for aa in seq:
            if aa in AA_PROPERTIES:
                h += AA_PROPERTIES[aa]['hydrophobic']
                c += AA_PROPERTIES[aa]['charge']
                s += AA_PROPERTIES[aa]['size']
                n += 1
        if n > 0:
            hydrophobic_scores.append(h / n)
            charge_scores.append(c / n)
            size_scores.append(s / n)

    if not hydrophobic_scores:
        return np.zeros(9, dtype=np.float32)

    return np.array([
        np.mean(hydrophobic_scores), np.std(hydrophobic_scores), np.median(hydrophobic_scores),
        np.mean(charge_scores), np.std(charge_scores), np.median(charge_scores),
        np.mean(size_scores), np.std(size_scores), np.median(size_scores)
    ], dtype=np.float32)

def extract_all_features(
    sequences: List[str],
    v_calls: List[str],
    j_calls: List[str],
    v_encoder: LabelEncoder,
    j_encoder: LabelEncoder,
    counts: Optional[List[int]] = None
) -> np.ndarray:
    """Extract all features for a repertoire"""
    features = []

    # 1. K-mer features (for each k value)
    for k in config.K_VALUES:
        kmer_feat = extract_kmer_features(sequences, k, config.HASH_DIM)
        features.append(kmer_feat)

    # 2. Amino acid composition
    aa_feat = extract_aa_composition(sequences)
    features.append(aa_feat)

    # 3. Length features
    len_feat = extract_length_features(sequences)
    features.append(len_feat)

    # 4. Positional features
    pos_feat = extract_positional_features(sequences)
    features.append(pos_feat)

    # 5. Clonality features
    clon_feat = extract_clonality_features(sequences, counts)
    features.append(clon_feat)

    # 6. V/J gene features
    vj_feat = extract_vj_features(v_calls, j_calls, v_encoder, j_encoder)
    features.append(vj_feat)

    # 7. Physicochemical features
    phys_feat = extract_physicochemical_features(sequences)
    features.append(phys_feat)

    return np.concatenate(features)

# ============================================================================
# DATA LOADING
# ============================================================================

def load_repertoire(filepath: str, max_seqs: int = 10000) -> Tuple[List[str], List[str], List[str], List[int]]:
    """Load a single repertoire file"""
    try:
        df = pd.read_csv(filepath, sep='\t', usecols=['junction_aa', 'v_call', 'j_call'])
    except:
        try:
            df = pd.read_csv(filepath, sep='\t')
            if 'junction_aa' not in df.columns:
                return [], [], [], []
        except:
            return [], [], [], []

    # Clean data
    df = df.dropna(subset=['junction_aa'])
    df = df[df['junction_aa'].str.len() > 0]

    # Sample if too many
    if len(df) > max_seqs:
        df = df.sample(n=max_seqs, random_state=config.SEED)

    sequences = df['junction_aa'].tolist()
    v_calls = df['v_call'].fillna('unknown').tolist() if 'v_call' in df.columns else ['unknown'] * len(sequences)
    j_calls = df['j_call'].fillna('unknown').tolist() if 'j_call' in df.columns else ['unknown'] * len(sequences)

    # Get counts if available
    if 'templates' in df.columns:
        counts = df['templates'].fillna(1).astype(int).tolist()
    elif 'duplicate_count' in df.columns:
        counts = df['duplicate_count'].fillna(1).astype(int).tolist()
    else:
        counts = [1] * len(sequences)

    return sequences, v_calls, j_calls, counts

def build_gene_encoders(train_paths: List[Path]) -> Tuple[LabelEncoder, LabelEncoder]:
    """Build V/J gene encoders from training data"""
    cache_path = os.path.join(config.CACHE_DIR, "gene_encoders.pkl")

    cached = load_cache(cache_path)
    if cached is not None:
        print("  📦 Loaded gene encoders from cache")
        return cached

    print("  Building gene encoders...")
    v_genes = Counter()
    j_genes = Counter()

    for path in tqdm(train_paths, desc="  Scanning genes"):
        try:
            df = pd.read_csv(path, sep='\t', usecols=['v_call', 'j_call'])
            v_genes.update(df['v_call'].dropna().tolist())
            j_genes.update(df['j_call'].dropna().tolist())
        except:
            continue

    # Keep top genes
    top_v = [g for g, _ in v_genes.most_common(config.TOP_V_GENES)]
    top_j = [g for g, _ in j_genes.most_common(config.TOP_J_GENES)]

    v_encoder = LabelEncoder()
    v_encoder.fit(top_v + ['unknown'])

    j_encoder = LabelEncoder()
    j_encoder.fit(top_j + ['unknown'])

    save_cache((v_encoder, j_encoder), cache_path)
    print(f"  ✅ Built encoders: {len(v_encoder.classes_)} V genes, {len(j_encoder.classes_)} J genes")

    return v_encoder, j_encoder

def extract_dataset_features(
    dataset_path: str,
    v_encoder: LabelEncoder,
    j_encoder: LabelEncoder,
    is_train: bool = True
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Extract features for an entire dataset with caching"""
    dataset_name = os.path.basename(dataset_path)
    cache_key = get_cache_key(dataset_path, f"features_{config.MAX_SEQS_PER_REP}")
    cache_path = os.path.join(config.CACHE_DIR, f"{cache_key}.pkl")

    cached = load_cache(cache_path)
    if cached is not None:
        print(f"  📦 Loaded features from cache: {dataset_name}")
        return cached

    features_list = []
    labels = []
    rep_ids = []

    # Check if metadata.csv exists (train) or not (test)
    metadata_path = os.path.join(dataset_path, "metadata.csv")

    if os.path.exists(metadata_path):
        # Train dataset with metadata
        metadata = pd.read_csv(metadata_path)

        for _, row in tqdm(metadata.iterrows(), total=len(metadata), desc=f"  Extracting features"):
            rep_id = row['repertoire_id']
            filepath = os.path.join(dataset_path, row['filename'])

            if not os.path.exists(filepath):
                continue

            sequences, v_calls, j_calls, counts = load_repertoire(filepath, config.MAX_SEQS_PER_REP)
            if len(sequences) == 0:
                continue

            feat = extract_all_features(sequences, v_calls, j_calls, v_encoder, j_encoder, counts)
            features_list.append(feat)
            labels.append(row['label_positive'] if is_train else 0)
            rep_ids.append(rep_id)
    else:
        # Test dataset - list all .tsv files
        tsv_files = sorted([f for f in os.listdir(dataset_path) if f.endswith('.tsv')])

        for filename in tqdm(tsv_files, desc=f"  Extracting features"):
            filepath = os.path.join(dataset_path, filename)
            rep_id = filename.replace('.tsv', '')  # Use filename without extension as ID

            sequences, v_calls, j_calls, counts = load_repertoire(filepath, config.MAX_SEQS_PER_REP)
            if len(sequences) == 0:
                continue

            feat = extract_all_features(sequences, v_calls, j_calls, v_encoder, j_encoder, counts)
            features_list.append(feat)
            labels.append(0)  # Dummy for test
            rep_ids.append(rep_id)

    X = np.array(features_list, dtype=np.float32)
    y = np.array(labels, dtype=np.float32)

    # Cache
    save_cache((X, y, rep_ids), cache_path)
    print(f"  ✅ Extracted {len(X)} repertoires, {X.shape[1]} features")

    return X, y, rep_ids

# ============================================================================
# NEURAL NETWORK MODEL
# ============================================================================

class ChampionNet(nn.Module):
    """Championship neural network with residual connections"""
    def __init__(self, input_dim: int, hidden_dims: Tuple[int, ...], dropout: float = 0.4):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for i, dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.LayerNorm(dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            prev_dim = dim

        self.encoder = nn.Sequential(*layers)
        self.head = nn.Linear(prev_dim, 1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        h = self.encoder(x)
        return self.head(h)

# ============================================================================
# TRAINING
# ============================================================================

def train_fold(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    fold: int,
    dataset_name: str
) -> Tuple[nn.Module, float, StandardScaler]:
    """Train a single fold"""
    # Normalize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Handle NaN/Inf
    X_train_scaled = np.nan_to_num(X_train_scaled, nan=0, posinf=0, neginf=0)
    X_val_scaled = np.nan_to_num(X_val_scaled, nan=0, posinf=0, neginf=0)

    # Create tensors
    X_train_t = torch.FloatTensor(X_train_scaled).to(config.DEVICE)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(1).to(config.DEVICE)
    X_val_t = torch.FloatTensor(X_val_scaled).to(config.DEVICE)
    y_val_t = torch.FloatTensor(y_val).unsqueeze(1).to(config.DEVICE)

    # Create dataset and loader
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)

    # Model
    model = ChampionNet(
        input_dim=X_train_scaled.shape[1],
        hidden_dims=config.HIDDEN_DIMS,
        dropout=config.DROPOUT
    ).to(config.DEVICE)

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.EPOCHS)

    # Training loop
    best_auc = 0
    patience_counter = 0
    best_state = None

    for epoch in range(config.EPOCHS):
        model.train()
        train_loss = 0

        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()

        # Validation
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
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= config.PATIENCE:
            break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_auc, scaler

def train_dataset(
    X: np.ndarray, y: np.ndarray,
    dataset_name: str
) -> Tuple[List[nn.Module], List[StandardScaler], float]:
    """Train models for a dataset using cross-validation"""
    print(f"\n  Training {dataset_name}...")

    skf = StratifiedKFold(n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED)

    models = []
    scalers = []
    aucs = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model, auc, scaler = train_fold(X_train, y_train, X_val, y_val, fold, dataset_name)
        models.append(model)
        scalers.append(scaler)
        aucs.append(auc)

        print(f"    Fold {fold+1}: AUC = {auc:.4f}")

    mean_auc = np.mean(aucs)
    print(f"  📊 Mean AUC: {mean_auc:.4f} (std: {np.std(aucs):.4f})")

    return models, scalers, mean_auc

# ============================================================================
# PREDICTION
# ============================================================================

def predict_dataset(
    X: np.ndarray,
    models: List[nn.Module],
    scalers: List[StandardScaler]
) -> np.ndarray:
    """Make ensemble predictions for a dataset"""
    all_preds = []

    for model, scaler in zip(models, scalers):
        X_scaled = scaler.transform(X)
        X_scaled = np.nan_to_num(X_scaled, nan=0, posinf=0, neginf=0)
        X_t = torch.FloatTensor(X_scaled).to(config.DEVICE)

        model.eval()
        with torch.no_grad():
            logits = model(X_t)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()

        all_preds.append(probs)

    # Ensemble by averaging
    ensemble_preds = np.mean(all_preds, axis=0)
    return ensemble_preds

# ============================================================================
# TASK B: SEQUENCE IDENTIFICATION
# ============================================================================

def identify_sequences(
    dataset_path: str,
    models: List[nn.Module],
    scalers: List[StandardScaler],
    v_encoder: LabelEncoder,
    j_encoder: LabelEncoder,
    top_k: int = 50000
) -> pd.DataFrame:
    """Identify top label-associated sequences"""
    dataset_name = os.path.basename(dataset_path)
    print(f"\n  Identifying sequences for {dataset_name}...")

    # Load all sequences from positive samples
    metadata = pd.read_csv(os.path.join(dataset_path, "metadata.csv"))
    positive_samples = metadata[metadata['label_positive'] == 1]

    all_sequences = []

    for _, row in tqdm(positive_samples.iterrows(), total=len(positive_samples), desc="  Loading sequences"):
        filepath = os.path.join(dataset_path, row['filename'])
        if not os.path.exists(filepath):
            continue

        try:
            df = pd.read_csv(filepath, sep='\t')
            for _, seq_row in df.iterrows():
                if pd.notna(seq_row.get('junction_aa')):
                    all_sequences.append({
                        'junction_aa': seq_row['junction_aa'],
                        'v_call': seq_row.get('v_call', 'unknown'),
                        'j_call': seq_row.get('j_call', 'unknown'),
                        'count': seq_row.get('templates', seq_row.get('duplicate_count', 1))
                    })
        except:
            continue

    if not all_sequences:
        # Return dummy sequences if none found
        return pd.DataFrame({
            'ID': [f"{dataset_name}_seq_top_{i+1}" for i in range(top_k)],
            'dataset': [dataset_name] * top_k,
            'label_positive_probability': [-999.0] * top_k,
            'junction_aa': ['CASSXXX'] * top_k,
            'v_call': ['TRBV1'] * top_k,
            'j_call': ['TRBJ1-1'] * top_k
        })

    # Convert to DataFrame and aggregate
    seq_df = pd.DataFrame(all_sequences)

    # Group by sequence and sum counts
    seq_agg = seq_df.groupby('junction_aa').agg({
        'v_call': 'first',
        'j_call': 'first',
        'count': 'sum'
    }).reset_index()

    # Sort by count (frequency-based importance)
    seq_agg = seq_agg.sort_values('count', ascending=False)

    # Take top k
    top_seqs = seq_agg.head(top_k)

    # Format output
    result = pd.DataFrame({
        'ID': [f"{dataset_name}_seq_top_{i+1}" for i in range(len(top_seqs))],
        'dataset': [dataset_name] * len(top_seqs),
        'label_positive_probability': [-999.0] * len(top_seqs),
        'junction_aa': top_seqs['junction_aa'].values,
        'v_call': top_seqs['v_call'].fillna('unknown').values,
        'j_call': top_seqs['j_call'].fillna('unknown').values
    })

    # Pad if needed
    if len(result) < top_k:
        padding = pd.DataFrame({
            'ID': [f"{dataset_name}_seq_top_{i+1}" for i in range(len(result), top_k)],
            'dataset': [dataset_name] * (top_k - len(result)),
            'label_positive_probability': [-999.0] * (top_k - len(result)),
            'junction_aa': ['CASSXXX'] * (top_k - len(result)),
            'v_call': ['TRBV1'] * (top_k - len(result)),
            'j_call': ['TRBJ1-1'] * (top_k - len(result))
        })
        result = pd.concat([result, padding], ignore_index=True)

    print(f"  ✅ Identified {len(result)} sequences")
    return result

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Main championship pipeline"""
    print(f"""
    ╔══════════════════════════════════════════════════════════════════╗
    ║  🏆 AIRR-ML-25 Ultimate Championship Pipeline 🏆                ║
    ║                                                                  ║
    ║  Architecture: Multi-Feature + Neural Network + Ensemble        ║
    ║  Target: Beat baseline 0.72866 → Top position                   ║
    ║  Features: K-mers + V/J + Clonality + Position + Physicochemical ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)

    print(f"🚀 Device: {config.DEVICE}")
    if config.DEVICE == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ========================================================================
    # PHASE 1: BUILD GENE ENCODERS
    # ========================================================================
    print("\n" + "=" * 70)
    print("🏆 PHASE 1: BUILDING GENE ENCODERS")
    print("=" * 70)

    # Get all training file paths
    train_datasets = sorted([d for d in os.listdir(config.TRAIN_ROOT) if d.startswith("train_dataset_")])
    all_train_paths = []
    for ds in train_datasets:
        ds_path = os.path.join(config.TRAIN_ROOT, ds)
        metadata = pd.read_csv(os.path.join(ds_path, "metadata.csv"))
        for _, row in metadata.iterrows():
            all_train_paths.append(Path(os.path.join(ds_path, row['filename'])))

    v_encoder, j_encoder = build_gene_encoders(all_train_paths)

    # ========================================================================
    # PHASE 2: FEATURE EXTRACTION AND TRAINING
    # ========================================================================
    print("\n" + "=" * 70)
    print("🏆 PHASE 2: FEATURE EXTRACTION & TRAINING")
    print("=" * 70)

    all_models = {}  # dataset -> (models, scalers, auc)
    all_aucs = []

    for ds_name in train_datasets:
        print(f"\n📂 {ds_name}")
        ds_path = os.path.join(config.TRAIN_ROOT, ds_name)

        # Extract features
        X, y, rep_ids = extract_dataset_features(ds_path, v_encoder, j_encoder, is_train=True)

        # Check label distribution
        pos_rate = y.mean()
        print(f"  Label distribution: {pos_rate:.2%} positive")

        # Train models
        models, scalers, mean_auc = train_dataset(X, y, ds_name)
        all_models[ds_name] = (models, scalers)
        all_aucs.append(mean_auc)

        # Save checkpoint
        checkpoint = {
            'models': [m.state_dict() for m in models],
            'scalers': scalers,
            'auc': mean_auc,
            'config': {
                'hidden_dims': config.HIDDEN_DIMS,
                'dropout': config.DROPOUT,
                'input_dim': X.shape[1]
            }
        }
        checkpoint_path = os.path.join(config.CHECKPOINT_DIR, f"{ds_name}_models.pt")
        torch.save(checkpoint, checkpoint_path)
        print(f"  💾 Saved checkpoint: {checkpoint_path}")

    print(f"\n📊 Overall Mean AUC: {np.mean(all_aucs):.4f}")

    # ========================================================================
    # PHASE 3: TEST PREDICTIONS
    # ========================================================================
    print("\n" + "=" * 70)
    print("🏆 PHASE 3: TEST PREDICTIONS")
    print("=" * 70)

    test_datasets = sorted([d for d in os.listdir(config.TEST_ROOT) if d.startswith("test_dataset_")])

    predictions = []

    for test_name in test_datasets:
        print(f"\n📂 {test_name}")
        test_path = os.path.join(config.TEST_ROOT, test_name)

        # Extract features
        X_test, _, rep_ids = extract_dataset_features(test_path, v_encoder, j_encoder, is_train=False)

        # Find corresponding training dataset
        # Map test to train (e.g., test_dataset_1 -> train_dataset_1, test_dataset_7_1 -> train_dataset_7)
        # Extract base number: test_dataset_X or test_dataset_X_Y -> X
        parts = test_name.replace('test_dataset_', '').split('_')
        base_num = parts[0]  # Get the first part (7 from 7_1)
        train_name = f"train_dataset_{base_num}"

        if train_name not in all_models:
            # Use first available model
            train_name = list(all_models.keys())[0]

        models, scalers = all_models[train_name]

        # Predict
        probs = predict_dataset(X_test, models, scalers)

        # Check prediction variability
        print(f"  Predictions: min={probs.min():.4f}, max={probs.max():.4f}, std={probs.std():.4f}")

        for rep_id, prob in zip(rep_ids, probs):
            predictions.append({
                'ID': rep_id,
                'dataset': test_name,
                'label_positive_probability': prob,
                'junction_aa': -999.0,
                'v_call': -999.0,
                'j_call': -999.0
            })

    # ========================================================================
    # PHASE 4: TASK B - SEQUENCE IDENTIFICATION
    # ========================================================================
    print("\n" + "=" * 70)
    print("🏆 PHASE 4: SEQUENCE IDENTIFICATION")
    print("=" * 70)

    seq_results = []

    for ds_name in train_datasets:
        ds_path = os.path.join(config.TRAIN_ROOT, ds_name)
        models, scalers = all_models[ds_name]

        seq_df = identify_sequences(ds_path, models, scalers, v_encoder, j_encoder, top_k=50000)
        seq_results.append(seq_df)

    # ========================================================================
    # PHASE 5: CREATE SUBMISSION
    # ========================================================================
    print("\n" + "=" * 70)
    print("🏆 PHASE 5: CREATING SUBMISSION")
    print("=" * 70)

    # Combine predictions and sequences
    pred_df = pd.DataFrame(predictions)
    seq_df = pd.concat(seq_results, ignore_index=True)

    submission = pd.concat([pred_df, seq_df], ignore_index=True)

    # Save submission
    submission_path = "submission_ultimate.csv"
    submission.to_csv(submission_path, index=False)

    print(f"\n✅ Submission saved: {submission_path}")
    print(f"   Total rows: {len(submission)}")
    print(f"   Predictions: {len(pred_df)}")
    print(f"   Sequences: {len(seq_df)}")

    # Verify format
    print("\n📋 Submission format check:")
    print(f"   Columns: {submission.columns.tolist()}")
    print(f"   Shape: {submission.shape}")

    # Check prediction variability across all test datasets
    pred_check = submission[submission['junction_aa'] == -999.0]['label_positive_probability']
    print(f"\n📊 Prediction statistics:")
    print(f"   Min: {pred_check.min():.4f}")
    print(f"   Max: {pred_check.max():.4f}")
    print(f"   Mean: {pred_check.mean():.4f}")
    print(f"   Std: {pred_check.std():.4f}")

    print("\n🏆 Championship pipeline complete!")

if __name__ == "__main__":
    main()
