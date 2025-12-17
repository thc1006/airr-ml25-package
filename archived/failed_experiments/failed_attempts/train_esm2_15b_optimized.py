#!/usr/bin/env python3
"""
train_esm2_15b_optimized.py - ESM-2 15B Optimized Championship Pipeline
=========================================================================

Target: Beat GROZD (0.81364) → Achieve 0.82+
Hardware: NVIDIA DGX Spark (GB10, 128GB unified memory, HugePages disabled)

OPTIMIZATIONS FOR 78GB+ AVAILABLE RAM:
- ESM-2 15B model (facebook/esm2_t48_15B_UR50D) - 15 BILLION parameters
- Larger batch sizes for better GPU utilization
- Layer 33 embeddings (optimal for 48-layer model)
- Optimized memory management

KEY CONFIGURATION NOTES (ADJUST BASED ON 3B RESULTS):
- esm_batch_size: Start with 4, can increase to 8 if memory allows
- train_batch_size: Start with 8, adjust based on GPU memory during MIL training
- train_epochs: 30 with early stopping (patience=5)
- esm_reduced_dim: 512 (can increase to 768 for more expressiveness)

Usage:
    docker run --gpus all --ipc=host \
        --ulimit memlock=-1 --ulimit stack=67108864 \
        --shm-size=64g \
        -v $(pwd):/app -v $(pwd)/data:/app/data \
        -v ~/.cache:/root/.cache \
        -w /app --rm \
        nvcr.io/nvidia/pytorch:25.11-py3 \
        bash -c "pip install transformers scikit-learn tqdm pandas accelerate --quiet && \
                 python -u scripts/train_esm2_15b_optimized.py"
"""

import os
import sys
import gc
import json
import pickle
import warnings
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

# Force unbuffered output
os.environ['PYTHONUNBUFFERED'] = '1'
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

# PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler

# Sklearn imports
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA

# Suppress warnings
warnings.filterwarnings('ignore')


def print_banner():
    """Print championship banner"""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║  🏆 ESM-2 15B OPTIMIZED CHAMPIONSHIP PIPELINE 🏆                             ║
║                                                                              ║
║  Target: Beat GROZD (0.81364) → Achieve 0.82+                                ║
║  Hardware: NVIDIA DGX Spark (GB10, 128GB unified, HugePages OFF)             ║
║                                                                              ║
║  ARCHITECTURE (MAXIMUM POWER):                                               ║
║  ✅ ESM-2 15B (esm2_t48_15B_UR50D) - 15 BILLION parameters!                  ║
║  ✅ Layer 33 embeddings (optimal for 48-layer model)                         ║
║  ✅ Optimized batch sizes for 78GB+ RAM                                      ║
║  ✅ Gated Attention MIL (EAMIL/DeepRC)                                       ║
║  ✅ LODO Cross-Validation with early stopping                                ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """, flush=True)


# =============================================================================
# Configuration - ADJUST THESE BASED ON 3B MODEL RESULTS
# =============================================================================
@dataclass
class Config:
    """Championship Configuration - Optimized for 78GB+ RAM"""
    # Paths
    data_dir: str = '/app/data'
    output_dir: str = '/app/outputs'
    checkpoint_dir: str = '/app/cache/esm2_15b_optimized'

    # Hardware
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_workers: int = 4

    # ==========================================================================
    # ESM-2 15B Settings - OPTIMIZED FOR 78GB+ RAM
    # ==========================================================================
    # Model selection (uncomment the one you want to use):
    esm_model_name: str = 'facebook/esm2_t48_15B_UR50D'  # 15B model - MAXIMUM POWER
    # esm_model_name: str = 'facebook/esm2_t36_3B_UR50D'  # 3B model - fallback

    # Layer selection (adjust based on model):
    # - For 15B (48 layers): use layer 33 (optimal ~2/3 depth)
    # - For 3B (36 layers): use layer 24
    esm_repr_layer: int = 33  # TODO: Change to 24 if using 3B model

    # Batch size for ESM embedding extraction
    # - 15B model: 16 is optimal for 72GB+ unified memory
    # - 3B model: can use 32
    # NOTE: Larger batch = faster but more memory
    # OPTIMIZED: 72GB+ RAM allows batch_size=16 for 15B model!
    esm_batch_size: int = 16  # OPTIMIZED for 72GB+ RAM

    # PCA dimension for embedding reduction
    # - Higher = more information preserved, but slower training
    # - 768 gives more expressiveness for better results
    # OPTIMIZED: Use 768 for maximum expressiveness
    esm_reduced_dim: int = 768  # OPTIMIZED for better representation

    # ==========================================================================
    # PrimeSeq Strategy - Sequence Selection
    # ==========================================================================
    # Number of top sequences per repertoire
    # - More sequences = more information, but slower
    # - 2000 for maximum signal capture with 72GB+ RAM
    # OPTIMIZED: Use 2000 for better signal capture
    primeseq_n_select: int = 2000  # OPTIMIZED for maximum signal

    # Frequency ratio for sequence selection
    # - 0.5 = 50% by frequency, 50% random
    primeseq_freq_ratio: float = 0.5

    # ==========================================================================
    # MIL (Multiple Instance Learning) Settings
    # ==========================================================================
    # Hidden dimension - should match or be close to esm_reduced_dim
    # OPTIMIZED: Match esm_reduced_dim=768 for consistency
    mil_hidden_dim: int = 768  # OPTIMIZED to match esm_reduced_dim

    # Attention dimension - typically half of hidden_dim
    # OPTIMIZED: 384 = half of 768
    mil_attention_dim: int = 384  # OPTIMIZED for 768 hidden_dim

    # Dropout for regularization
    # - Higher = more regularization, less overfitting
    # - 0.3 is good starting point, try 0.4 if overfitting
    mil_dropout: float = 0.3  # TODO: Increase to 0.4 if overfitting

    # ==========================================================================
    # Training Settings - OPTIMIZED FOR 72GB+ RAM
    # ==========================================================================
    # Maximum epochs (will early stop if no improvement)
    # OPTIMIZED: 40 epochs for thorough training
    train_epochs: int = 50  # OPTIMIZED for thorough training

    # Batch size for MIL training (repertoires per batch)
    # - Larger = faster, more stable gradients
    # - 64 is optimal for 72GB+ RAM with unified memory
    # OPTIMIZED: 64 for maximum GPU utilization
    train_batch_size: int = 64  # OPTIMIZED for 72GB+ RAM

    # Learning rate
    # - 1e-4 is good starting point with larger batch size
    train_lr: float = 1e-4

    # Early stopping patience
    # OPTIMIZED: 7 for better convergence
    early_stop_patience: int = 10  # OPTIMIZED for better convergence

    # Test patterns (auto-discovered)
    test_patterns: List[str] = None

    def __post_init__(self):
        if self.test_patterns is None:
            self.test_patterns = [
                'test_dataset_1', 'test_dataset_2', 'test_dataset_3',
                'test_dataset_4', 'test_dataset_5', 'test_dataset_6',
                'test_dataset_7', 'test_dataset_8', 'test_dataset_8_2',
                'test_dataset_8_3', 'test_dataset_8_4'
            ]


# =============================================================================
# Data Loading Utilities
# =============================================================================
def load_repertoire_data(tsv_path: Path, n_select: int = 1000) -> pd.DataFrame:
    """Load and sample repertoire data"""
    # Only use columns that exist: junction_aa, v_call, j_call
    df = pd.read_csv(tsv_path, sep='\t', usecols=['junction_aa', 'v_call', 'j_call'])
    df = df.dropna(subset=['junction_aa'])
    df = df[df['junction_aa'].str.len() >= 5]
    df = df[df['junction_aa'].str.len() <= 30]

    if len(df) > n_select:
        # Random sample since templates column doesn't exist
        df = df.sample(n=n_select, random_state=42)

    return df


def discover_datasets(data_dir: str) -> Dict:
    """Discover all training and test datasets"""
    data_path = Path(data_dir)
    datasets = {'train': {}, 'test': {}}

    # Training datasets
    train_dir = data_path / 'train_datasets'
    if train_dir.exists():
        for dataset_dir in sorted(train_dir.iterdir()):
            if dataset_dir.is_dir() and dataset_dir.name.startswith('train_dataset'):
                metadata_path = dataset_dir / 'metadata.csv'
                if metadata_path.exists():
                    datasets['train'][dataset_dir.name] = {
                        'path': dataset_dir,
                        'metadata': pd.read_csv(metadata_path)
                    }

    # Test datasets
    test_dir = data_path / 'test_datasets'
    if test_dir.exists():
        for dataset_dir in sorted(test_dir.iterdir()):
            if dataset_dir.is_dir() and dataset_dir.name.startswith('test_dataset'):
                datasets['test'][dataset_dir.name] = {
                    'path': dataset_dir
                }

    return datasets


# =============================================================================
# ESM-2 Embedding Extractor
# =============================================================================
class ESMEmbeddingExtractor:
    """Extract embeddings using HuggingFace ESM-2"""

    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.pca = None

    def load_model(self):
        """Load ESM-2 model with retry logic"""
        from transformers import EsmModel, EsmTokenizer

        print(f"📥 Loading ESM-2 from HuggingFace...", flush=True)
        print(f"   Model: {self.config.esm_model_name}", flush=True)

        # Disable XET for stability
        os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'

        max_retries = 10
        for attempt in range(max_retries):
            try:
                print(f"   🔄 Attempt {attempt + 1}/{max_retries}...", flush=True)
                start = time.time()

                self.tokenizer = EsmTokenizer.from_pretrained(self.config.esm_model_name)
                self.model = EsmModel.from_pretrained(self.config.esm_model_name)
                self.model = self.model.to(self.config.device)
                self.model.eval()

                elapsed = time.time() - start
                print(f"✅ ESM-2 loaded on {self.config.device} in {elapsed:.1f}s", flush=True)
                print(f"   Using representation layer: {self.config.esm_repr_layer}", flush=True)
                return

            except Exception as e:
                print(f"   ⚠️ Attempt {attempt + 1} failed: {e}", flush=True)
                if attempt < max_retries - 1:
                    time.sleep(30)
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                else:
                    raise RuntimeError(f"Failed to load ESM-2 after {max_retries} attempts")

    @torch.no_grad()
    def extract_batch(self, sequences: List[str]) -> np.ndarray:
        """Extract embeddings for a batch of sequences"""
        if self.model is None:
            self.load_model()

        # Tokenize
        inputs = self.tokenizer(sequences, return_tensors="pt", padding=True, truncation=True, max_length=50)
        inputs = {k: v.to(self.config.device) for k, v in inputs.items()}

        # Forward pass with FP16 for memory efficiency
        with torch.amp.autocast('cuda', dtype=torch.float16):
            outputs = self.model(**inputs, output_hidden_states=True)

        # Get specified layer embeddings
        hidden_states = outputs.hidden_states[self.config.esm_repr_layer]

        # Mean pooling (exclude special tokens)
        attention_mask = inputs['attention_mask'].unsqueeze(-1)
        masked_embeddings = hidden_states * attention_mask
        sum_embeddings = masked_embeddings.sum(dim=1)
        count = attention_mask.sum(dim=1)
        mean_embeddings = sum_embeddings / count

        return mean_embeddings.float().cpu().numpy()

    def extract_all(self, sequences: List[str], show_progress: bool = True) -> np.ndarray:
        """Extract embeddings for all sequences"""
        all_embeddings = []
        batch_size = self.config.esm_batch_size

        iterator = range(0, len(sequences), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="ESM-2 embedding", file=sys.stdout)

        for i in iterator:
            batch = sequences[i:i+batch_size]
            embeddings = self.extract_batch(batch)
            all_embeddings.append(embeddings)

        return np.vstack(all_embeddings)

    def fit_pca(self, embeddings: np.ndarray):
        """Fit PCA for dimensionality reduction"""
        print(f"Fitting PCA: {embeddings.shape[1]} → {self.config.esm_reduced_dim}", flush=True)
        self.pca = PCA(n_components=self.config.esm_reduced_dim, random_state=42)
        self.pca.fit(embeddings)
        print(f"   Explained variance: {self.pca.explained_variance_ratio_.sum():.4f}", flush=True)

    def transform_pca(self, embeddings: np.ndarray) -> np.ndarray:
        """Apply PCA transformation"""
        if self.pca is None:
            raise ValueError("PCA not fitted yet")
        return self.pca.transform(embeddings)


# =============================================================================
# MIL Model Architecture - Gated Attention
# =============================================================================
class GatedAttentionMIL(nn.Module):
    """Gated Attention MIL for repertoire classification"""

    def __init__(self, input_dim: int, hidden_dim: int, attention_dim: int, dropout: float = 0.3):
        super().__init__()

        # Feature transformation
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Gated attention mechanism
        self.attention_V = nn.Sequential(
            nn.Linear(hidden_dim, attention_dim),
            nn.Tanh()
        )
        self.attention_U = nn.Sequential(
            nn.Linear(hidden_dim, attention_dim),
            nn.Sigmoid()
        )
        self.attention_w = nn.Linear(attention_dim, 1)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x, mask=None):
        # x: (batch, n_instances, input_dim)
        batch_size, n_instances, _ = x.shape

        # Feature extraction
        h = self.feature_extractor(x)  # (batch, n_instances, hidden_dim)

        # Gated attention
        A_V = self.attention_V(h)  # (batch, n_instances, attention_dim)
        A_U = self.attention_U(h)  # (batch, n_instances, attention_dim)
        A = self.attention_w(A_V * A_U)  # (batch, n_instances, 1)

        # Apply mask if provided
        if mask is not None:
            A = A.masked_fill(~mask.unsqueeze(-1), float('-inf'))

        # Softmax attention weights
        A = F.softmax(A, dim=1)  # (batch, n_instances, 1)

        # Weighted aggregation
        M = torch.sum(A * h, dim=1)  # (batch, hidden_dim)

        # Classification
        logits = self.classifier(M)  # (batch, 1)

        return logits, A.squeeze(-1)


# =============================================================================
# Dataset and DataLoader
# =============================================================================
class RepertoireDataset(Dataset):
    """Dataset for repertoire-level classification"""

    def __init__(self, embeddings_dict: Dict[str, np.ndarray],
                 labels_dict: Dict[str, int],
                 max_instances: int = 1000):
        self.rep_ids = list(embeddings_dict.keys())
        self.embeddings = embeddings_dict
        self.labels = labels_dict
        self.max_instances = max_instances

    def __len__(self):
        return len(self.rep_ids)

    def __getitem__(self, idx):
        rep_id = self.rep_ids[idx]
        emb = self.embeddings[rep_id]
        label = self.labels[rep_id]

        # Subsample if too many instances
        if len(emb) > self.max_instances:
            indices = np.random.choice(len(emb), self.max_instances, replace=False)
            emb = emb[indices]

        return {
            'rep_id': rep_id,
            'embeddings': torch.tensor(emb, dtype=torch.float32),
            'label': torch.tensor(label, dtype=torch.float32),
            'n_instances': len(emb)
        }


def collate_fn(batch):
    """Custom collate function for variable-length sequences"""
    max_len = max(b['n_instances'] for b in batch)

    batch_size = len(batch)
    emb_dim = batch[0]['embeddings'].shape[1]

    # Pad embeddings
    padded = torch.zeros(batch_size, max_len, emb_dim)
    mask = torch.zeros(batch_size, max_len, dtype=torch.bool)

    for i, b in enumerate(batch):
        n = b['n_instances']
        padded[i, :n] = b['embeddings']
        mask[i, :n] = True

    return {
        'embeddings': padded,
        'mask': mask,
        'label': torch.stack([b['label'] for b in batch]),
        'rep_id': [b['rep_id'] for b in batch]
    }


# =============================================================================
# Training Functions
# =============================================================================
def train_epoch(model, dataloader, optimizer, scaler, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0

    for batch in dataloader:
        embeddings = batch['embeddings'].to(device)
        mask = batch['mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()

        with torch.amp.autocast('cuda', dtype=torch.float16):
            logits, _ = model(embeddings, mask)
            loss = F.binary_cross_entropy_with_logits(logits.squeeze(), labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    return total_loss / len(dataloader)


@torch.no_grad()
def evaluate(model, dataloader, device):
    """Evaluate model"""
    model.eval()
    all_preds = []
    all_labels = []
    all_rep_ids = []

    for batch in dataloader:
        embeddings = batch['embeddings'].to(device)
        mask = batch['mask'].to(device)
        labels = batch['label']
        rep_ids = batch['rep_id']

        with torch.amp.autocast('cuda', dtype=torch.float16):
            logits, _ = model(embeddings, mask)

        probs = torch.sigmoid(logits).cpu().numpy().flatten()

        all_preds.extend(probs)
        all_labels.extend(labels.numpy())
        all_rep_ids.extend(rep_ids)

    auc = roc_auc_score(all_labels, all_preds)
    return auc, dict(zip(all_rep_ids, all_preds))


# =============================================================================
# Task B: Sequence Selection
# =============================================================================
def select_task_b_sequences(train_embeddings: Dict, train_labels: Dict,
                            train_sequences: Dict, esm_extractor) -> Dict:
    """Select top 50,000 label-associated sequences per dataset"""
    print("\n  Selecting Task B sequences...", flush=True)

    # Group by dataset
    dataset_data = defaultdict(lambda: {'embeddings': [], 'labels': [], 'sequences': [], 'rep_ids': []})

    for rep_id in train_embeddings.keys():
        dataset_id = '_'.join(rep_id.split('_')[:3])
        label = train_labels[rep_id]

        embs = train_embeddings.get(rep_id)
        seqs = train_sequences.get(rep_id)

        if embs is not None and seqs is not None:
            dataset_data[dataset_id]['embeddings'].append(embs)
            dataset_data[dataset_id]['sequences'].extend(seqs)
            dataset_data[dataset_id]['labels'].extend([label] * len(seqs))
            dataset_data[dataset_id]['rep_ids'].extend([rep_id] * len(seqs))

    task_b_results = {}

    for dataset_id, data in dataset_data.items():
        print(f"    Processing {dataset_id}...", flush=True)

        all_embeddings = np.vstack(data['embeddings'])
        all_labels = np.array(data['labels'])
        all_sequences = data['sequences']

        # Train logistic regression
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(all_embeddings)

        lr = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
        lr.fit(X_scaled, all_labels)

        # Get coefficients and select top sequences
        coefficients = lr.coef_[0]
        sequence_scores = X_scaled @ coefficients

        # Get top 50,000 by absolute score
        top_indices = np.argsort(np.abs(sequence_scores))[-50000:]

        task_b_results[dataset_id] = [all_sequences[i] for i in top_indices]
        print(f"      Selected {len(task_b_results[dataset_id])} sequences", flush=True)

    return task_b_results


# =============================================================================
# Main Training Pipeline
# =============================================================================
def main():
    print_banner()

    config = Config()

    # Print system info
    if torch.cuda.is_available():
        print(f"🚀 GPU: {torch.cuda.get_device_name(0)}", flush=True)
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"   Memory: {total_mem:.1f} GB", flush=True)

    # Create directories
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # Discover datasets
    print("\n📁 Discovering datasets...", flush=True)
    datasets = discover_datasets(config.data_dir)
    print(f"   Training: {len(datasets['train'])} datasets", flush=True)
    print(f"   Test: {len(datasets['test'])} datasets", flush=True)

    # Initialize ESM extractor
    esm_extractor = ESMEmbeddingExtractor(config)

    # ==========================================================================
    # Stage 1: Extract ESM-2 Embeddings for Training Data
    # ==========================================================================
    print("\n" + "="*60, flush=True)
    print("Stage 1: Extracting ESM-2 Embeddings", flush=True)
    print("="*60, flush=True)

    checkpoint_path = Path(config.checkpoint_dir) / 'train_embeddings.pkl'

    if checkpoint_path.exists():
        print(f"📂 Loading cached embeddings from {checkpoint_path}...", flush=True)
        with open(checkpoint_path, 'rb') as f:
            cache = pickle.load(f)
            train_embeddings = cache['embeddings']
            train_labels = cache['labels']
            train_sequences = cache['sequences']
            if 'pca' in cache:
                esm_extractor.pca = cache['pca']
                print(f"   ✅ PCA loaded from checkpoint", flush=True)
    else:
        train_embeddings = {}
        train_labels = {}
        train_sequences = {}

        for dataset_id, dataset_info in tqdm(datasets['train'].items(), desc="Datasets"):
            metadata = dataset_info['metadata']
            dataset_path = dataset_info['path']

            print(f"\n  Processing {dataset_id}...", flush=True)

            for _, row in tqdm(metadata.iterrows(), total=len(metadata), desc=f"  {dataset_id}", leave=False):
                rep_id = row['repertoire_id']
                label_val = row['label_positive']
                label = 1 if (label_val is True or str(label_val).lower() == 'true') else 0
                tsv_path = dataset_path / f"{rep_id}.tsv"

                if not tsv_path.exists():
                    continue

                # Load repertoire
                rep_df = load_repertoire_data(tsv_path, n_select=config.primeseq_n_select)
                sequences = rep_df['junction_aa'].tolist()

                if len(sequences) < 10:
                    continue

                # Extract embeddings
                embeddings = esm_extractor.extract_all(sequences, show_progress=False)

                train_embeddings[rep_id] = embeddings
                train_labels[rep_id] = label
                train_sequences[rep_id] = sequences

            # Clear GPU cache periodically
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Fit PCA on all embeddings
        print("\n  Fitting PCA...", flush=True)
        all_emb = np.vstack(list(train_embeddings.values()))
        esm_extractor.fit_pca(all_emb)

        # Apply PCA
        for rep_id in train_embeddings:
            train_embeddings[rep_id] = esm_extractor.transform_pca(train_embeddings[rep_id])

        # Save checkpoint
        print(f"\n  Saving checkpoint to {checkpoint_path}...", flush=True)
        with open(checkpoint_path, 'wb') as f:
            pickle.dump({
                'embeddings': train_embeddings,
                'labels': train_labels,
                'sequences': train_sequences,
                'pca': esm_extractor.pca
            }, f)

    print(f"\n✅ Loaded {len(train_embeddings)} repertoires", flush=True)

    # ==========================================================================
    # Stage 2: LODO Cross-Validation Training
    # ==========================================================================
    print("\n" + "="*60, flush=True)
    print("Stage 2: LODO Cross-Validation Training", flush=True)
    print("="*60, flush=True)

    # Group by dataset
    dataset_reps = defaultdict(list)
    for rep_id in train_embeddings.keys():
        dataset_id = '_'.join(rep_id.split('_')[:3])
        dataset_reps[dataset_id].append(rep_id)

    dataset_ids = sorted(dataset_reps.keys())
    print(f"  Datasets: {dataset_ids}", flush=True)

    fold_models = []
    fold_aucs = []

    for fold_idx, val_dataset in enumerate(dataset_ids):
        print(f"\n  Fold {fold_idx+1}/{len(dataset_ids)}: Validating on {val_dataset}", flush=True)

        # Split
        train_reps = [r for d, reps in dataset_reps.items() for r in reps if d != val_dataset]
        val_reps = dataset_reps[val_dataset]

        print(f"    Train: {len(train_reps)}, Val: {len(val_reps)}", flush=True)

        # Create datasets
        train_emb = {r: train_embeddings[r] for r in train_reps}
        train_lab = {r: train_labels[r] for r in train_reps}
        val_emb = {r: train_embeddings[r] for r in val_reps}
        val_lab = {r: train_labels[r] for r in val_reps}

        train_dataset = RepertoireDataset(train_emb, train_lab)
        val_dataset_obj = RepertoireDataset(val_emb, val_lab)

        train_loader = DataLoader(train_dataset, batch_size=config.train_batch_size,
                                  shuffle=True, collate_fn=collate_fn, num_workers=0)
        val_loader = DataLoader(val_dataset_obj, batch_size=config.train_batch_size,
                                shuffle=False, collate_fn=collate_fn, num_workers=0)

        # Create model
        model = GatedAttentionMIL(
            input_dim=config.esm_reduced_dim,
            hidden_dim=config.mil_hidden_dim,
            attention_dim=config.mil_attention_dim,
            dropout=config.mil_dropout
        ).to(config.device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=config.train_lr, weight_decay=0.01)
        scaler = GradScaler()

        # Training loop
        best_auc = 0
        best_model_state = None
        patience_counter = 0

        for epoch in range(config.train_epochs):
            train_loss = train_epoch(model, train_loader, optimizer, scaler, config.device)
            val_auc, _ = evaluate(model, val_loader, config.device)

            if val_auc > best_auc:
                best_auc = val_auc
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1

            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"    Epoch {epoch+1}: Loss={train_loss:.4f}, Val AUC={val_auc:.4f} (Best: {best_auc:.4f})", flush=True)

            if patience_counter >= config.early_stop_patience:
                print(f"    Early stopping at epoch {epoch+1}", flush=True)
                break

        # Restore best model
        model.load_state_dict(best_model_state)
        fold_models.append(model)
        fold_aucs.append(best_auc)

        print(f"    ✅ Fold {fold_idx+1} Best AUC: {best_auc:.4f}", flush=True)

    mean_auc = np.mean(fold_aucs)
    print(f"\n📊 LODO CV Mean AUC: {mean_auc:.4f} (±{np.std(fold_aucs):.4f})", flush=True)

    # Save LODO models checkpoint
    models_checkpoint_path = Path(config.checkpoint_dir) / 'lodo_models.pkl'
    print(f"\n  Saving LODO models to {models_checkpoint_path}...", flush=True)
    with open(models_checkpoint_path, 'wb') as f:
        pickle.dump({
            'fold_models_state': [m.state_dict() for m in fold_models],
            'fold_aucs': fold_aucs,
            'dataset_ids': dataset_ids,
            'config': {
                'mil_hidden_dim': config.mil_hidden_dim,
                'mil_attention_dim': config.mil_attention_dim,
                'esm_reduced_dim': config.esm_reduced_dim,
            }
        }, f)
    print(f"  ✅ LODO models saved", flush=True)

    # ==========================================================================
    # Stage 3: Test Predictions (Task A)
    # ==========================================================================
    print("\n" + "="*60, flush=True)
    print("Stage 3: Test Predictions (Task A)", flush=True)
    print("="*60, flush=True)

    # Load PCA if needed
    if esm_extractor.pca is None:
        with open(checkpoint_path, 'rb') as f:
            cache = pickle.load(f)
            esm_extractor.pca = cache['pca']

    test_predictions = {}
    rep_to_dataset = {}

    # Test embeddings cache directory
    test_cache_dir = Path(config.checkpoint_dir) / 'test_embeddings'
    test_cache_dir.mkdir(parents=True, exist_ok=True)

    for test_name, test_info in datasets['test'].items():
        print(f"\n  Processing {test_name}...", flush=True)
        test_path = test_info['path']

        # Check for cached test embeddings
        test_cache_path = test_cache_dir / f'{test_name}_embeddings.pkl'
        if test_cache_path.exists():
            print(f"    📂 Loading cached embeddings from {test_cache_path}...", flush=True)
            with open(test_cache_path, 'rb') as f:
                cache = pickle.load(f)
                cached_embeddings = cache['embeddings']
                cached_rep_to_dataset = cache['rep_to_dataset']

            for rep_id, embeddings in cached_embeddings.items():
                rep_to_dataset[rep_id] = cached_rep_to_dataset[rep_id]

                if embeddings is None:
                    test_predictions[rep_id] = 0.5
                    continue

                test_emb = {rep_id: embeddings}
                test_lab = {rep_id: 0}
                test_dataset = RepertoireDataset(test_emb, test_lab)
                test_loader = DataLoader(test_dataset, batch_size=1, collate_fn=collate_fn)

                all_probs = []
                for model in fold_models:
                    _, preds = evaluate(model, test_loader, config.device)
                    all_probs.append(preds[rep_id])

                test_predictions[rep_id] = np.mean(all_probs)

            print(f"    ✅ Loaded {len(cached_embeddings)} cached repertoires", flush=True)
            continue

        # No cache - extract embeddings
        tsv_files = list(test_path.glob('*.tsv'))
        print(f"    Found {len(tsv_files)} repertoires", flush=True)

        dataset_embeddings = {}
        dataset_rep_mapping = {}

        for tsv_file in tqdm(tsv_files, desc=f"    {test_name}", leave=False):
            rep_id = tsv_file.stem
            rep_to_dataset[rep_id] = test_name
            dataset_rep_mapping[rep_id] = test_name

            rep_df = load_repertoire_data(tsv_file, n_select=config.primeseq_n_select)
            sequences = rep_df['junction_aa'].tolist()

            if len(sequences) < 10:
                test_predictions[rep_id] = 0.5
                dataset_embeddings[rep_id] = None
                continue

            embeddings = esm_extractor.extract_all(sequences, show_progress=False)
            embeddings = esm_extractor.transform_pca(embeddings)

            dataset_embeddings[rep_id] = embeddings

            test_emb = {rep_id: embeddings}
            test_lab = {rep_id: 0}
            test_dataset = RepertoireDataset(test_emb, test_lab)
            test_loader = DataLoader(test_dataset, batch_size=1, collate_fn=collate_fn)

            all_probs = []
            for model in fold_models:
                _, preds = evaluate(model, test_loader, config.device)
                all_probs.append(preds[rep_id])

            test_predictions[rep_id] = np.mean(all_probs)

        # Save test embeddings cache
        print(f"    💾 Saving embeddings cache to {test_cache_path}...", flush=True)
        with open(test_cache_path, 'wb') as f:
            pickle.dump({
                'embeddings': dataset_embeddings,
                'rep_to_dataset': dataset_rep_mapping
            }, f)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if len(test_predictions) != 4213:
        print(f"⚠️ WARNING: Expected 4,213 Task A predictions, got {len(test_predictions)}", flush=True)

    print(f"\n✅ Generated {len(test_predictions)} Task A predictions", flush=True)

    # ==========================================================================
    # Stage 4: Task B Sequence Selection
    # ==========================================================================
    print("\n" + "="*60, flush=True)
    print("Stage 4: Task B Sequence Selection", flush=True)
    print("="*60, flush=True)

    task_b_selections = select_task_b_sequences(
        train_embeddings, train_labels, train_sequences, esm_extractor
    )

    # ==========================================================================
    # Stage 5: Generate Submission
    # ==========================================================================
    print("\n" + "="*60, flush=True)
    print("Stage 5: Generating Submission", flush=True)
    print("="*60, flush=True)

    submission_rows = []

    # Task A rows
    for rep_id, prob in test_predictions.items():
        dataset_name = rep_to_dataset.get(rep_id)
        if dataset_name is None:
            raise ValueError(f"No dataset mapping for repertoire {rep_id}")

        submission_rows.append({
            'ID': rep_id,
            'dataset': dataset_name,
            'label_positive_probability': prob,
            'junction_aa': -999.0,
            'v_call': -999.0,
            'j_call': -999.0
        })

    # Task B rows
    for dataset_id, sequences in task_b_selections.items():
        for i, seq in enumerate(sequences):
            submission_rows.append({
                'ID': f"{dataset_id}_seq_{i+1}",
                'dataset': dataset_id,
                'label_positive_probability': -999.0,
                'junction_aa': seq,
                'v_call': -999.0,
                'j_call': -999.0
            })

    # Create submission DataFrame
    submission_df = pd.DataFrame(submission_rows)
    submission_df = submission_df[['ID', 'dataset', 'label_positive_probability', 'junction_aa', 'v_call', 'j_call']]

    # Validate
    expected_rows = 4213 + 8 * 50000
    if len(submission_df) != expected_rows:
        print(f"⚠️ WARNING: Expected {expected_rows} rows, got {len(submission_df)}", flush=True)

    # Save
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = Path(config.output_dir) / f'submission_esm2_15b_{timestamp}.csv'
    submission_df.to_csv(output_path, index=False)

    print(f"\n✅ Submission saved to {output_path}", flush=True)
    print(f"   Total rows: {len(submission_df)}", flush=True)
    print(f"   Task A: {len(test_predictions)} predictions", flush=True)
    print(f"   Task B: {sum(len(s) for s in task_b_selections.values())} sequences", flush=True)

    print("\n" + "="*60, flush=True)
    print("🏆 ESM-2 15B CHAMPIONSHIP PIPELINE COMPLETE!", flush=True)
    print("="*60, flush=True)


if __name__ == '__main__':
    main()
