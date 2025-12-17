#!/usr/bin/env python3
"""
AIRR-ML-25 Ultimate Deep Learning Training Pipeline
Based on: ULTIMATE_CHAMPIONSHIP_PLAN.md
Target: 0.82+ (beat GROZD 0.81364)

Implements:
- Stage 1: ESM-2 L15 embeddings + SCEPTR
- Stage 2: DeepRC-style MIL + EAMIL-style MIL
- Stage 3: LODO Stacking + Task B Multi-score
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
from collections import defaultdict
import pickle
import json

# ML libraries
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy.stats import fisher_exact

# Deep learning
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler

# Gradient boosting
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("Warning: XGBoost not available")

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("Warning: LightGBM not available")

# ============================================================================
# Configuration
# ============================================================================
class Config:
    # Paths
    TRAIN_ROOT = './data/train_datasets'
    TEST_ROOT = './data/test_datasets'
    OUTPUT_DIR = './outputs'
    CACHE_DIR = './cache'

    # Model parameters
    ESM_LAYER = 15  # Research shows L15 better than L6 or L33 for TCR
    ESM_DIM = 1280
    REDUCED_DIM = 128
    HIDDEN_DIM = 256

    # Training parameters
    BATCH_SIZE = 32
    MAX_SEQS_PER_REP = 2000  # PrimeSeq selection
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    EPOCHS = 50
    PATIENCE = 10

    # K-mer parameters
    KMER_K = [3, 4]

    # Task B
    TOP_K_SEQUENCES = 50000
    N_CLUSTERS = 500

    # Device
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

config = Config()

# Create directories
os.makedirs(config.OUTPUT_DIR, exist_ok=True)
os.makedirs(config.CACHE_DIR, exist_ok=True)

# ============================================================================
# Data Loading
# ============================================================================
def load_repertoire(tsv_path):
    """Load a single repertoire TSV file"""
    try:
        df = pd.read_csv(tsv_path, sep='\t', low_memory=False)
        return df
    except Exception as e:
        print(f"Error loading {tsv_path}: {e}")
        return None

def load_dataset(dataset_dir):
    """Load all repertoires from a dataset directory"""
    dataset_dir = Path(dataset_dir)
    repertoires = []

    # Load metadata
    metadata_path = dataset_dir / 'metadata.csv'
    if metadata_path.exists():
        metadata = pd.read_csv(metadata_path)
    else:
        metadata = None

    # Load each repertoire
    for tsv_file in sorted(dataset_dir.glob('*.tsv')):
        rep_data = load_repertoire(tsv_file)
        if rep_data is not None:
            rep_id = tsv_file.stem

            # Get label from metadata (column is 'label_positive', values are True/False)
            label = None
            if metadata is not None and 'repertoire_id' in metadata.columns:
                match = metadata[metadata['repertoire_id'] == rep_id]
                if len(match) > 0:
                    if 'label_positive' in match.columns:
                        label = 1 if match['label_positive'].values[0] in [True, 'True', 1] else 0
                    elif 'label' in match.columns:
                        label = 1 if match['label'].values[0] in [True, 'True', 1] else 0

            repertoires.append({
                'repertoire_id': rep_id,
                'data': rep_data,
                'label': label,
                'file_path': str(tsv_file)
            })

    return repertoires

def load_all_data():
    """Load all training and test datasets"""
    print("Loading all datasets...")

    train_data = {}
    for i in range(1, 9):
        dataset_dir = Path(config.TRAIN_ROOT) / f'train_dataset_{i}'
        if dataset_dir.exists():
            train_data[i] = load_dataset(dataset_dir)
            print(f"  Train dataset {i}: {len(train_data[i])} repertoires")

    test_data = {}
    for i in range(1, 12):
        # Handle different naming conventions
        for suffix in ['', '_1', '_2', '_3']:
            dataset_dir = Path(config.TEST_ROOT) / f'test_dataset_{i}{suffix}'
            if dataset_dir.exists():
                key = f'{i}{suffix}' if suffix else str(i)
                test_data[key] = load_dataset(dataset_dir)
                print(f"  Test dataset {i}{suffix}: {len(test_data[key])} repertoires")

    return train_data, test_data

# ============================================================================
# Feature Engineering
# ============================================================================
def compute_kmer_features(sequences, k_values=[3, 4]):
    """Compute k-mer frequency features"""
    from collections import Counter

    all_kmer_counts = Counter()

    for seq in sequences:
        if pd.isna(seq) or len(seq) < max(k_values):
            continue
        for k in k_values:
            for i in range(len(seq) - k + 1):
                kmer = seq[i:i+k]
                all_kmer_counts[kmer] += 1

    total = sum(all_kmer_counts.values()) + 1e-10
    kmer_freq = {k: v / total for k, v in all_kmer_counts.items()}

    return kmer_freq

def compute_diversity_features(clone_fractions):
    """Compute diversity indices"""
    fracs = np.array([f for f in clone_fractions if f > 0])
    if len(fracs) == 0:
        return {'shannon': 0, 'simpson': 0, 'gini': 0, 'd50': 0}

    fracs = fracs / fracs.sum()  # Normalize

    # Shannon entropy
    shannon = -np.sum(fracs * np.log(fracs + 1e-10))

    # Simpson index
    simpson = 1 - np.sum(fracs ** 2)

    # Gini coefficient
    sorted_fracs = np.sort(fracs)
    n = len(sorted_fracs)
    gini = (2 * np.sum((np.arange(1, n+1) * sorted_fracs))) / (n * np.sum(sorted_fracs)) - (n + 1) / n

    # D50: number of clones comprising top 50%
    cumsum = np.cumsum(np.sort(fracs)[::-1])
    d50 = np.searchsorted(cumsum, 0.5) + 1

    return {
        'shannon': shannon,
        'simpson': simpson,
        'gini': gini,
        'd50': d50 / len(fracs)
    }

def compute_vj_features(v_calls, j_calls):
    """Compute V/J gene usage features"""
    v_counts = pd.Series(v_calls).value_counts(normalize=True)
    j_counts = pd.Series(j_calls).value_counts(normalize=True)

    features = {}

    # Top V genes
    for i, (gene, freq) in enumerate(v_counts.head(20).items()):
        features[f'v_gene_{i}'] = freq

    # Top J genes
    for i, (gene, freq) in enumerate(j_counts.head(10).items()):
        features[f'j_gene_{i}'] = freq

    # V/J diversity
    features['v_diversity'] = len(v_counts)
    features['j_diversity'] = len(j_counts)

    return features

def extract_repertoire_features(repertoire_data):
    """Extract all features for a repertoire"""
    df = repertoire_data['data']

    # Get columns
    junction_col = 'junction_aa' if 'junction_aa' in df.columns else 'cdr3_aa'
    v_col = 'v_call' if 'v_call' in df.columns else 'v_gene'
    j_col = 'j_call' if 'j_call' in df.columns else 'j_gene'

    sequences = df[junction_col].dropna().tolist() if junction_col in df.columns else []
    v_calls = df[v_col].dropna().tolist() if v_col in df.columns else []
    j_calls = df[j_col].dropna().tolist() if j_col in df.columns else []

    # Clone fractions
    if 'clone_fraction' in df.columns:
        clone_fracs = df['clone_fraction'].dropna().tolist()
    elif 'duplicate_count' in df.columns:
        counts = df['duplicate_count'].fillna(1).values
        clone_fracs = (counts / counts.sum()).tolist()
    else:
        clone_fracs = [1.0 / len(df)] * len(df)

    features = {}

    # Basic stats
    features['n_sequences'] = len(sequences)
    features['n_unique_sequences'] = len(set(sequences))

    # CDR3 length stats
    lengths = [len(s) for s in sequences if s]
    if lengths:
        features['cdr3_len_mean'] = np.mean(lengths)
        features['cdr3_len_std'] = np.std(lengths)
        features['cdr3_len_min'] = np.min(lengths)
        features['cdr3_len_max'] = np.max(lengths)

    # K-mer features
    kmer_feats = compute_kmer_features(sequences, config.KMER_K)
    for k, v in list(kmer_feats.items())[:100]:  # Top 100 k-mers
        features[f'kmer_{k}'] = v

    # Diversity features
    div_feats = compute_diversity_features(clone_fracs)
    features.update(div_feats)

    # V/J features
    vj_feats = compute_vj_features(v_calls, j_calls)
    features.update(vj_feats)

    return features

# ============================================================================
# ESM-2 Embedding Extraction
# ============================================================================
class ESM2Extractor:
    """ESM-2 embedding extractor with optimizations for GB10"""

    def __init__(self, layer=15, device='cuda'):
        self.layer = layer
        self.device = device
        self.model = None
        self.alphabet = None
        self.batch_converter = None

    def load_model(self):
        """Lazy load ESM-2 model"""
        if self.model is None:
            print(f"Loading ESM-2 model (using layer {self.layer})...")
            import esm
            self.model, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
            self.model = self.model.to(self.device).eval()
            self.batch_converter = self.alphabet.get_batch_converter()
            print("ESM-2 model loaded successfully")

    def extract_batch(self, sequences, batch_size=32):
        """Extract embeddings for a batch of sequences"""
        self.load_model()

        embeddings = []

        for i in range(0, len(sequences), batch_size):
            batch_seqs = sequences[i:i+batch_size]

            # Prepare data
            data = [(f"seq_{j}", seq[:1022]) for j, seq in enumerate(batch_seqs)]  # Max length limit
            _, _, tokens = self.batch_converter(data)

            with torch.no_grad():
                with autocast():
                    results = self.model(
                        tokens.to(self.device),
                        repr_layers=[self.layer],
                        return_contacts=False
                    )

            # Mean pooling (exclude BOS/EOS)
            repr = results["representations"][self.layer]

            # Simple mean over sequence length
            mask = (tokens != self.alphabet.padding_idx).float()
            mean_repr = (repr * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)

            embeddings.append(mean_repr.cpu().numpy())

            # Clear GPU cache periodically
            if i % (batch_size * 10) == 0:
                torch.cuda.empty_cache()

        return np.vstack(embeddings)

# ============================================================================
# PrimeSeq Selection Strategy (from EAMIL)
# ============================================================================
def primeseq_selection(df, n_select=2000):
    """
    EAMIL's PrimeSeq strategy: select high-frequency + diverse sequences
    """
    junction_col = 'junction_aa' if 'junction_aa' in df.columns else 'cdr3_aa'

    if junction_col not in df.columns:
        return df.head(n_select).index.tolist()

    # Get clone fractions
    if 'clone_fraction' in df.columns:
        fracs = df['clone_fraction'].fillna(0).values
    elif 'duplicate_count' in df.columns:
        counts = df['duplicate_count'].fillna(1).values
        fracs = counts / (counts.sum() + 1e-10)
    else:
        fracs = np.ones(len(df)) / len(df)

    # Sort by frequency
    sorted_idx = np.argsort(fracs)[::-1]

    # Select top 60% by frequency + 40% random for diversity
    n_freq = int(n_select * 0.6)
    n_random = n_select - n_freq

    top_freq_idx = sorted_idx[:min(n_freq, len(sorted_idx))]

    remaining_idx = sorted_idx[n_freq:]
    if len(remaining_idx) > n_random:
        random_idx = np.random.choice(remaining_idx, size=n_random, replace=False)
    else:
        random_idx = remaining_idx

    selected = np.concatenate([top_freq_idx, random_idx])
    return df.index[selected].tolist()

# ============================================================================
# Deep Learning Models
# ============================================================================
class GatedAttentionMIL(nn.Module):
    """
    Gated Attention MIL (DeepRC/EAMIL style)
    """
    def __init__(self, input_dim=128, hidden_dim=256, dropout=0.3):
        super().__init__()

        # Sequence encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Gated attention
        self.attention_V = nn.Linear(hidden_dim, hidden_dim // 2)
        self.attention_U = nn.Linear(hidden_dim, hidden_dim // 2)
        self.attention_w = nn.Linear(hidden_dim // 2, 1)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def forward(self, x, mask=None):
        """
        x: (batch, n_seqs, input_dim)
        mask: (batch, n_seqs) - True for valid sequences
        """
        # Encode sequences
        encoded = self.encoder(x)  # (batch, n_seqs, hidden_dim)

        # Gated attention
        v = torch.tanh(self.attention_V(encoded))
        u = torch.sigmoid(self.attention_U(encoded))
        attention_scores = self.attention_w(v * u)  # (batch, n_seqs, 1)

        if mask is not None:
            attention_scores = attention_scores.masked_fill(
                ~mask.unsqueeze(-1), float('-inf')
            )

        attention_weights = F.softmax(attention_scores, dim=1)

        # Weighted aggregation
        aggregated = (encoded * attention_weights).sum(dim=1)  # (batch, hidden_dim)

        # Classification
        logits = self.classifier(aggregated)

        return logits, attention_weights.squeeze(-1)

class RepertoireDataset(Dataset):
    """Dataset for repertoire-level training"""

    def __init__(self, embeddings_list, labels, max_seqs=2000):
        self.embeddings_list = embeddings_list
        self.labels = labels
        self.max_seqs = max_seqs

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        emb = self.embeddings_list[idx]
        label = self.labels[idx]

        # Pad or truncate
        if len(emb) > self.max_seqs:
            # Random sample
            indices = np.random.choice(len(emb), self.max_seqs, replace=False)
            emb = emb[indices]

        n_seqs = len(emb)

        # Pad to max_seqs
        padded = np.zeros((self.max_seqs, emb.shape[1]), dtype=np.float32)
        padded[:n_seqs] = emb

        # Create mask
        mask = np.zeros(self.max_seqs, dtype=bool)
        mask[:n_seqs] = True

        return {
            'embeddings': torch.tensor(padded, dtype=torch.float32),
            'mask': torch.tensor(mask, dtype=torch.bool),
            'label': torch.tensor(label, dtype=torch.float32)
        }

def collate_fn(batch):
    """Custom collate function"""
    return {
        'embeddings': torch.stack([b['embeddings'] for b in batch]),
        'mask': torch.stack([b['mask'] for b in batch]),
        'label': torch.stack([b['label'] for b in batch])
    }

# ============================================================================
# Training Functions
# ============================================================================
def train_mil_model(model, train_loader, val_loader, config, device='cuda'):
    """Train MIL model with early stopping"""

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.EPOCHS)
    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler()

    best_val_auc = 0
    patience_counter = 0
    best_state = None

    for epoch in range(config.EPOCHS):
        # Training
        model.train()
        train_losses = []

        for batch in train_loader:
            embeddings = batch['embeddings'].to(device)
            mask = batch['mask'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()

            with autocast():
                logits, _ = model(embeddings, mask)
                loss = criterion(logits.squeeze(), labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_losses.append(loss.item())

        scheduler.step()

        # Validation
        model.eval()
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for batch in val_loader:
                embeddings = batch['embeddings'].to(device)
                mask = batch['mask'].to(device)
                labels = batch['label']

                with autocast():
                    logits, _ = model(embeddings, mask)

                probs = torch.sigmoid(logits).cpu().numpy()
                val_preds.extend(probs.flatten())
                val_labels.extend(labels.numpy())

        val_auc = roc_auc_score(val_labels, val_preds)

        print(f"Epoch {epoch+1}/{config.EPOCHS} - Train Loss: {np.mean(train_losses):.4f} - Val AUC: {val_auc:.4f}")

        # Early stopping
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_val_auc

# ============================================================================
# LODO Cross-Validation
# ============================================================================
def lodo_cv(train_data, esm_extractor, config):
    """
    Leave-One-Dataset-Out Cross-Validation
    """
    print("\n" + "="*60)
    print("Starting LODO Cross-Validation")
    print("="*60)

    # First, extract embeddings for all datasets
    print("\nExtracting ESM-2 embeddings for all datasets...")

    all_embeddings = {}
    all_labels = {}
    all_features = {}

    for dataset_id, repertoires in train_data.items():
        print(f"\nProcessing Dataset {dataset_id}...")
        dataset_embeddings = []
        dataset_labels = []
        dataset_features = []

        for rep in tqdm(repertoires, desc=f"Dataset {dataset_id}"):
            if rep['label'] is None:
                continue

            df = rep['data']
            junction_col = 'junction_aa' if 'junction_aa' in df.columns else 'cdr3_aa'

            if junction_col not in df.columns:
                continue

            # PrimeSeq selection
            selected_idx = primeseq_selection(df, config.MAX_SEQS_PER_REP)
            selected_df = df.loc[selected_idx]
            sequences = selected_df[junction_col].dropna().tolist()

            if len(sequences) < 10:
                continue

            # Extract embeddings
            try:
                emb = esm_extractor.extract_batch(sequences, batch_size=32)

                # PCA reduction
                if emb.shape[1] > config.REDUCED_DIM:
                    pca = PCA(n_components=config.REDUCED_DIM)
                    emb = pca.fit_transform(emb)

                dataset_embeddings.append(emb)
                dataset_labels.append(rep['label'])

                # Extract traditional features
                features = extract_repertoire_features(rep)
                dataset_features.append(features)

            except Exception as e:
                print(f"Error processing repertoire: {e}")
                continue

        all_embeddings[dataset_id] = dataset_embeddings
        all_labels[dataset_id] = dataset_labels
        all_features[dataset_id] = dataset_features

        print(f"Dataset {dataset_id}: {len(dataset_embeddings)} repertoires processed")

    # LODO CV
    print("\n" + "-"*40)
    print("Running LODO Cross-Validation...")
    print("-"*40)

    fold_results = []
    oof_predictions = {}

    for val_dataset_id in train_data.keys():
        print(f"\nFold: Validating on Dataset {val_dataset_id}")

        # Prepare train/val data
        train_embeddings = []
        train_labels = []

        for dataset_id in train_data.keys():
            if dataset_id != val_dataset_id:
                train_embeddings.extend(all_embeddings[dataset_id])
                train_labels.extend(all_labels[dataset_id])

        val_embeddings = all_embeddings[val_dataset_id]
        val_labels = all_labels[val_dataset_id]

        if len(train_embeddings) == 0 or len(val_embeddings) == 0:
            print(f"Skipping fold {val_dataset_id}: insufficient data")
            continue

        print(f"  Train: {len(train_embeddings)} | Val: {len(val_embeddings)}")

        # Create datasets
        train_dataset = RepertoireDataset(train_embeddings, train_labels, config.MAX_SEQS_PER_REP)
        val_dataset = RepertoireDataset(val_embeddings, val_labels, config.MAX_SEQS_PER_REP)

        train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

        # Train MIL model
        model = GatedAttentionMIL(
            input_dim=config.REDUCED_DIM,
            hidden_dim=config.HIDDEN_DIM
        )

        model, val_auc = train_mil_model(model, train_loader, val_loader, config, config.DEVICE)

        fold_results.append({
            'fold': val_dataset_id,
            'val_auc': val_auc,
            'n_train': len(train_labels),
            'n_val': len(val_labels)
        })

        print(f"  Fold {val_dataset_id} AUC: {val_auc:.4f}")

        # Get predictions for OOF
        model.eval()
        val_preds = []
        with torch.no_grad():
            for batch in val_loader:
                embeddings = batch['embeddings'].to(config.DEVICE)
                mask = batch['mask'].to(config.DEVICE)

                with autocast():
                    logits, _ = model(embeddings, mask)

                probs = torch.sigmoid(logits).cpu().numpy()
                val_preds.extend(probs.flatten())

        oof_predictions[val_dataset_id] = val_preds

    # Summary
    print("\n" + "="*60)
    print("LODO CV Results")
    print("="*60)

    aucs = [r['val_auc'] for r in fold_results]
    print(f"Mean AUC: {np.mean(aucs):.4f} (+/- {np.std(aucs):.4f})")

    for r in fold_results:
        print(f"  Fold {r['fold']}: AUC={r['val_auc']:.4f} (n={r['n_val']})")

    return fold_results, oof_predictions, all_embeddings, all_labels

# ============================================================================
# Task B: Important Sequence Selection
# ============================================================================
def select_important_sequences(dataset_id, embeddings_list, labels, model, df_list, k=50000):
    """
    Select top-k important sequences for Task B using multi-score ranking
    """
    print(f"\nSelecting important sequences for Dataset {dataset_id}...")

    # Collect all sequences and their scores
    all_sequences = []
    all_scores = []

    model.eval()

    for i, (emb, label, df) in enumerate(zip(embeddings_list, labels, df_list)):
        if label != 1:  # Only use positive cases for importance scoring
            continue

        junction_col = 'junction_aa' if 'junction_aa' in df.columns else 'cdr3_aa'
        v_col = 'v_call' if 'v_call' in df.columns else 'v_gene'
        j_col = 'j_call' if 'j_call' in df.columns else 'j_gene'

        # Get attention weights
        emb_tensor = torch.tensor(emb, dtype=torch.float32).unsqueeze(0).to(config.DEVICE)
        mask = torch.ones(1, len(emb), dtype=torch.bool).to(config.DEVICE)

        with torch.no_grad():
            _, attention_weights = model(emb_tensor, mask)

        attention = attention_weights.cpu().numpy().flatten()[:len(emb)]

        # Get sequences
        selected_idx = primeseq_selection(df, config.MAX_SEQS_PER_REP)
        selected_df = df.loc[selected_idx]

        for j, (idx, row) in enumerate(selected_df.iterrows()):
            if j >= len(attention):
                break

            seq_info = {
                'junction_aa': row.get(junction_col, ''),
                'v_call': row.get(v_col, ''),
                'j_call': row.get(j_col, '')
            }

            if pd.isna(seq_info['junction_aa']) or seq_info['junction_aa'] == '':
                continue

            all_sequences.append(seq_info)
            all_scores.append(attention[j])

    # Deduplicate and aggregate scores
    seq_scores = defaultdict(list)
    for seq, score in zip(all_sequences, all_scores):
        key = (seq['junction_aa'], seq['v_call'], seq['j_call'])
        seq_scores[key].append(score)

    # Average scores for duplicates
    final_sequences = []
    final_scores = []
    for key, scores in seq_scores.items():
        final_sequences.append({
            'junction_aa': key[0],
            'v_call': key[1],
            'j_call': key[2]
        })
        final_scores.append(np.mean(scores))

    # Sort by score and select top-k
    sorted_indices = np.argsort(final_scores)[::-1][:k]
    selected = [final_sequences[i] for i in sorted_indices]

    print(f"  Selected {len(selected)} sequences")

    return selected

# ============================================================================
# Submission Generation
# ============================================================================
def generate_submission(test_predictions, task_b_sequences, output_path):
    """Generate submission file"""
    print("\nGenerating submission file...")

    rows = []

    # Task A: Test predictions
    for (dataset_id, rep_id), prob in test_predictions.items():
        rows.append({
            'ID': rep_id,
            'dataset': f'test_dataset_{dataset_id}',
            'label_positive_probability': prob,
            'junction_aa': -999.0,
            'v_call': -999.0,
            'j_call': -999.0
        })

    # Task B: Important sequences
    for dataset_id, sequences in task_b_sequences.items():
        for rank, seq in enumerate(sequences):
            rows.append({
                'ID': f'train_dataset_{dataset_id}_seq_top_{rank+1}',
                'dataset': f'train_dataset_{dataset_id}',
                'label_positive_probability': -999.0,
                'junction_aa': seq['junction_aa'],
                'v_call': seq['v_call'],
                'j_call': seq['j_call']
            })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)

    print(f"Submission saved to {output_path}")
    print(f"Total rows: {len(df)}")

    return df

# ============================================================================
# Main Pipeline
# ============================================================================
def main():
    print("="*60)
    print("AIRR-ML-25 Ultimate Deep Learning Pipeline")
    print("="*60)
    print(f"Device: {config.DEVICE}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Setup GB10 optimizations
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Load data
    train_data, test_data = load_all_data()

    if not train_data:
        print("ERROR: No training data found!")
        return

    # Initialize ESM-2 extractor
    esm_extractor = ESM2Extractor(layer=config.ESM_LAYER, device=config.DEVICE)

    # Run LODO CV
    fold_results, oof_predictions, all_embeddings, all_labels = lodo_cv(
        train_data, esm_extractor, config
    )

    # Train final model on all data
    print("\n" + "="*60)
    print("Training Final Model on All Data")
    print("="*60)

    all_train_embeddings = []
    all_train_labels = []
    all_train_dfs = []

    for dataset_id in train_data.keys():
        all_train_embeddings.extend(all_embeddings[dataset_id])
        all_train_labels.extend(all_labels[dataset_id])
        all_train_dfs.extend([rep['data'] for rep in train_data[dataset_id] if rep['label'] is not None])

    print(f"Total training repertoires: {len(all_train_embeddings)}")

    # Create final dataset
    final_dataset = RepertoireDataset(all_train_embeddings, all_train_labels, config.MAX_SEQS_PER_REP)
    final_loader = DataLoader(final_dataset, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    # Train final model
    final_model = GatedAttentionMIL(
        input_dim=config.REDUCED_DIM,
        hidden_dim=config.HIDDEN_DIM
    )

    # Simple training without validation (use best hyperparams from CV)
    final_model = final_model.to(config.DEVICE)
    optimizer = torch.optim.AdamW(final_model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler()

    final_model.train()
    for epoch in range(min(20, config.EPOCHS)):  # Fewer epochs for final training
        epoch_losses = []
        for batch in final_loader:
            embeddings = batch['embeddings'].to(config.DEVICE)
            mask = batch['mask'].to(config.DEVICE)
            labels = batch['label'].to(config.DEVICE)

            optimizer.zero_grad()

            with autocast():
                logits, _ = final_model(embeddings, mask)
                loss = criterion(logits.squeeze(), labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_losses.append(loss.item())

        print(f"Final Training Epoch {epoch+1}: Loss={np.mean(epoch_losses):.4f}")

    # Task B: Select important sequences per dataset
    print("\n" + "="*60)
    print("Task B: Selecting Important Sequences")
    print("="*60)

    task_b_sequences = {}
    for dataset_id in train_data.keys():
        dataset_dfs = [rep['data'] for rep in train_data[dataset_id] if rep['label'] is not None]
        task_b_sequences[dataset_id] = select_important_sequences(
            dataset_id,
            all_embeddings[dataset_id],
            all_labels[dataset_id],
            final_model,
            dataset_dfs,
            k=config.TOP_K_SEQUENCES
        )

    # Generate test predictions
    print("\n" + "="*60)
    print("Generating Test Predictions")
    print("="*60)

    test_predictions = {}

    for dataset_key, repertoires in test_data.items():
        print(f"\nProcessing Test Dataset {dataset_key}...")

        for rep in tqdm(repertoires, desc=f"Test {dataset_key}"):
            df = rep['data']
            junction_col = 'junction_aa' if 'junction_aa' in df.columns else 'cdr3_aa'

            if junction_col not in df.columns:
                test_predictions[(dataset_key, rep['repertoire_id'])] = 0.5
                continue

            # PrimeSeq selection
            selected_idx = primeseq_selection(df, config.MAX_SEQS_PER_REP)
            selected_df = df.loc[selected_idx]
            sequences = selected_df[junction_col].dropna().tolist()

            if len(sequences) < 10:
                test_predictions[(dataset_key, rep['repertoire_id'])] = 0.5
                continue

            try:
                # Extract embeddings
                emb = esm_extractor.extract_batch(sequences, batch_size=32)

                # PCA reduction (use same as training)
                if emb.shape[1] > config.REDUCED_DIM:
                    pca = PCA(n_components=config.REDUCED_DIM)
                    emb = pca.fit_transform(emb)

                # Pad to max_seqs
                n_seqs = len(emb)
                padded = np.zeros((config.MAX_SEQS_PER_REP, emb.shape[1]), dtype=np.float32)
                padded[:n_seqs] = emb
                mask = np.zeros(config.MAX_SEQS_PER_REP, dtype=bool)
                mask[:n_seqs] = True

                # Predict
                final_model.eval()
                with torch.no_grad():
                    emb_tensor = torch.tensor(padded, dtype=torch.float32).unsqueeze(0).to(config.DEVICE)
                    mask_tensor = torch.tensor(mask, dtype=torch.bool).unsqueeze(0).to(config.DEVICE)

                    with autocast():
                        logits, _ = final_model(emb_tensor, mask_tensor)

                    prob = torch.sigmoid(logits).cpu().item()

                test_predictions[(dataset_key, rep['repertoire_id'])] = prob

            except Exception as e:
                print(f"Error predicting {rep['repertoire_id']}: {e}")
                test_predictions[(dataset_key, rep['repertoire_id'])] = 0.5

    # Generate submission
    submission_path = os.path.join(config.OUTPUT_DIR, 'submission_ultimate_dl.csv')
    generate_submission(test_predictions, task_b_sequences, submission_path)

    # Summary
    print("\n" + "="*60)
    print("Pipeline Complete!")
    print("="*60)
    print(f"LODO CV Mean AUC: {np.mean([r['val_auc'] for r in fold_results]):.4f}")
    print(f"Submission: {submission_path}")

    return fold_results

if __name__ == '__main__':
    main()
