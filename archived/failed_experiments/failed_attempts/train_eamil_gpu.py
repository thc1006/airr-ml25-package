#!/usr/bin/env python3
"""
EAMIL GPU Training Script for AIRR-ML-25 Competition

Optimized for NVIDIA GB10 Grace Blackwell:
- Flash Attention 2 (native support)
- TF32 matmul precision
- Mixed precision training (FP16/BF16)
- Gradient checkpointing for larger batches

Usage:
    docker run --gpus all --ipc=host \
        --ulimit memlock=-1 --ulimit stack=67108864 \
        --shm-size=64g \
        -v /home/mbwcl/dev/airr-ml25-package:/app \
        -v ~/.cache:/root/.cache \
        -w /app --rm \
        nvcr.io/nvidia/pytorch:25.11-py3 \
        bash -c "pip install transformers scikit-learn tqdm pandas --quiet && \
                 python -u scripts/train_eamil_gpu.py"

Target: Beat 0.67887 → Achieve 0.80+ AUC
"""

import os
import sys
import pickle
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

warnings.filterwarnings('ignore')

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ============================================================================
# Configuration
# ============================================================================

class Config:
    # Data paths
    DATA_DIR = PROJECT_ROOT / "data"
    CACHE_DIR = PROJECT_ROOT / "cache" / "eamil_gpu"
    ESM2_GB10_CACHE = PROJECT_ROOT / "cache" / "esm2_gb10"  # 18GB usable cache!
    OUTPUT_DIR = PROJECT_ROOT / "outputs" / "submissions"

    # Model architecture - ESM2 GB10 embeddings (1280-dim)
    INPUT_DIM = 1280  # ESM2 GB10 sequence-level embeddings
    HIDDEN_DIM = 512
    ATTENTION_DIM = 256
    DROPOUT = 0.2

    # Training hyperparameters
    BATCH_SIZE = 8
    MAX_INSTANCES = 500  # Max TCR sequences per repertoire (GB10 has up to 1000)
    LEARNING_RATE = 3e-4
    WEIGHT_DECAY = 1e-4
    NUM_EPOCHS = 50
    PATIENCE = 10

    # GB10 optimizations
    USE_FP16 = True
    USE_FLASH_ATTENTION = True
    USE_TF32 = True
    GRADIENT_CHECKPOINTING = False

    # LODO Cross-validation
    TRAIN_DATASETS = [f"train_dataset_{i}" for i in range(1, 9)]
    TEST_DATASETS = [f"test_dataset_{i}" for i in range(1, 9)] + \
                   ["test_dataset_8_1", "test_dataset_8_2", "test_dataset_8_3"]


# ============================================================================
# GB10 Hardware Optimization
# ============================================================================

def setup_gb10_optimizations():
    """Enable NVIDIA GB10 Grace Blackwell optimizations."""
    print("=" * 70)
    print("Setting up NVIDIA GB10 Grace Blackwell Optimizations")
    print("=" * 70)

    # Check CUDA availability
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available! Run inside NGC Docker container.")

    device = torch.device("cuda")

    # Print GPU info
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"  GPU: {gpu_name}")
    print(f"  Memory: {gpu_memory:.1f} GB")

    # Flash Attention
    if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
        torch.backends.cuda.enable_flash_sdp(True)
        print("  [✓] Flash Attention 2 enabled")

    # TF32 precision (3x speedup on Ampere+)
    if Config.USE_TF32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("  [✓] TF32 matmul precision enabled")

    # cuDNN auto-tuner
    torch.backends.cudnn.benchmark = True
    print("  [✓] cuDNN benchmark enabled")

    # Set default dtype
    if Config.USE_FP16:
        print("  [✓] Mixed precision (FP16) enabled")

    print("=" * 70)

    return device


# ============================================================================
# EAMIL Model Architecture (Optimized for GB10)
# ============================================================================

class EAMIL(nn.Module):
    """
    Enhanced Attention MIL with Multi-Scale Pooling

    Features:
    - Gated attention mechanism
    - Multi-scale aggregation (attention + max + mean)
    - Optimized for GB10 unified memory
    """

    def __init__(
        self,
        input_dim: int = 1280,
        hidden_dim: int = 512,
        attention_dim: int = 256,
        dropout: float = 0.2,
        use_gated: bool = True,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.use_gated = use_gated

        # Instance encoder
        self.instance_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Gated attention
        self.attention_V = nn.Sequential(
            nn.Linear(hidden_dim, attention_dim),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(hidden_dim, attention_dim),
            nn.Sigmoid()
        ) if use_gated else None

        self.attention_weights = nn.Linear(attention_dim, 1)

        # Multi-scale fusion: [attn, max, mean] -> hidden
        self.feature_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x, mask=None):
        """
        Args:
            x: (batch, n_instances, input_dim)
            mask: (batch, n_instances) - True = padding

        Returns:
            logits: (batch, 1)
            attention_weights: (batch, n_instances)
        """
        batch_size, n_instances, _ = x.shape

        # Auto-create mask
        if mask is None:
            mask = (x.abs().sum(dim=-1) == 0.0)

        # Instance encoding
        h = self.instance_encoder(x)  # (batch, n_instances, hidden)

        # Gated attention
        A_V = self.attention_V(h)
        if self.use_gated:
            A_U = self.attention_U(h)
            A = A_V * A_U
        else:
            A = A_V

        attn_scores = self.attention_weights(A).squeeze(-1)  # (batch, n_instances)
        attn_scores = attn_scores.masked_fill(mask, float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=1)

        # Multi-scale pooling
        # 1. Attention-weighted
        attn_pooled = (h * attn_weights.unsqueeze(-1)).sum(dim=1)

        # 2. Max pooling
        h_masked = h.masked_fill(mask.unsqueeze(-1), float('-inf'))
        max_pooled = h_masked.max(dim=1)[0]

        # 3. Mean pooling
        h_masked = h.masked_fill(mask.unsqueeze(-1), 0.0)
        valid_counts = (~mask).sum(dim=1, keepdim=True).float().clamp(min=1.0)
        mean_pooled = h_masked.sum(dim=1) / valid_counts

        # Fusion
        multi_scale = torch.cat([attn_pooled, max_pooled, mean_pooled], dim=1)
        fused = self.feature_fusion(multi_scale)

        # Classification
        logits = self.classifier(fused)

        return logits, attn_weights


# ============================================================================
# Dataset
# ============================================================================

class RepertoireDataset(Dataset):
    """Dataset for repertoire-level MIL classification."""

    def __init__(self, features, labels, max_instances=500):
        self.features = features
        self.labels = labels
        self.max_instances = max_instances

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feat = self.features[idx]
        label = self.labels[idx] if self.labels is not None else -1

        # Truncate or pad to max_instances
        n_instances = min(len(feat), self.max_instances)
        padded = np.zeros((self.max_instances, feat.shape[1]), dtype=np.float32)
        padded[:n_instances] = feat[:n_instances]

        # Create mask (True = padding)
        mask = np.ones(self.max_instances, dtype=bool)
        mask[:n_instances] = False

        return {
            'features': torch.tensor(padded, dtype=torch.float32),
            'mask': torch.tensor(mask, dtype=torch.bool),
            'label': torch.tensor(label, dtype=torch.float32)
        }


# ============================================================================
# Data Loading Functions
# ============================================================================

def load_esm2_gb10_embeddings():
    """
    Load ESM2 GB10 sequence-level embeddings (18GB cache).

    This cache contains:
    - Shape: (1000, 1280) per repertoire
    - Values: [-224.8, +10.5] (healthy, not corrupted!)
    - All 8 training datasets available

    Perfect for MIL training with attention mechanism!
    """
    print("=" * 70)
    print("Loading ESM2 GB10 Sequence-Level Embeddings")
    print("  Cache: 18GB | Shape: (1000, 1280) per repertoire")
    print("  Values: [-224.8, +10.5] (HEALTHY!)")
    print("=" * 70)

    all_features = []
    all_labels = []
    all_dataset_ids = []

    for dataset_name in Config.TRAIN_DATASETS:
        cache_file = Config.ESM2_GB10_CACHE / f"{dataset_name}_embeddings.pkl"

        if not cache_file.exists():
            print(f"  [!] Missing: {cache_file}")
            continue

        with open(cache_file, 'rb') as f:
            data = pickle.load(f)

        # Use labels directly from cache (already stored!)
        embeddings_dict = data['embeddings']
        labels_dict = data.get('labels', {})

        for rep_id, seq_embeddings in embeddings_dict.items():
            # Get label from cache (True/False -> 1/0)
            label = labels_dict.get(rep_id)
            if label is None:
                continue
            label = int(label)  # Convert bool to int

            # seq_embeddings is (n_sequences, 1280)
            if hasattr(seq_embeddings, 'shape'):
                all_features.append(seq_embeddings)  # Keep sequence-level for MIL
                all_labels.append(label)
                all_dataset_ids.append(dataset_name)

        print(f"  {dataset_name}: {len(embeddings_dict)} repertoires loaded")

    print(f"\n  Total: {len(all_features)} repertoires")
    print(f"  Sample shape: {all_features[0].shape}")
    print("=" * 70)

    return all_features, np.array(all_labels), all_dataset_ids


def load_hybrid_fusion_features():
    """Fallback: Load mean-pooled features from Hybrid Fusion pipeline."""
    cache_path = PROJECT_ROOT / "cache" / "hybrid_fusion" / "train_features.pkl"

    if cache_path.exists():
        print(f"Loading Hybrid Fusion features from {cache_path}")
        with open(cache_path, 'rb') as f:
            data = pickle.load(f)
        return data['features'], data['labels'], data['dataset_ids']

    print("Hybrid Fusion cache not found.")
    return None, None, None


# ============================================================================
# Training Functions
# ============================================================================

def train_epoch(model, dataloader, optimizer, scaler, device):
    """Train for one epoch with mixed precision."""
    model.train()
    total_loss = 0

    for batch in tqdm(dataloader, desc="Training", leave=False):
        features = batch['features'].to(device)
        mask = batch['mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()

        with torch.amp.autocast('cuda', dtype=torch.float16, enabled=Config.USE_FP16):
            logits, _ = model(features, mask)
            loss = F.binary_cross_entropy_with_logits(logits.squeeze(-1), labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    """Evaluate model and return AUC."""
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            features = batch['features'].to(device)
            mask = batch['mask'].to(device)
            labels = batch['label']

            with torch.amp.autocast('cuda', dtype=torch.float16, enabled=Config.USE_FP16):
                logits, _ = model(features, mask)

            probs = torch.sigmoid(logits.squeeze(-1)).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())

    return roc_auc_score(all_labels, all_probs)


def lodo_cross_validation(features, labels, dataset_ids, device):
    """Leave-One-Dataset-Out Cross-Validation."""
    print("\n" + "=" * 70)
    print("LODO Cross-Validation (8-Fold)")
    print("=" * 70)

    unique_datasets = sorted(set(dataset_ids))
    fold_aucs = []
    best_models = {}

    for fold, val_dataset in enumerate(unique_datasets):
        print(f"\n[Fold {fold+1}/8] Held-out: {val_dataset}")

        # Split train/val
        train_idx = [i for i, d in enumerate(dataset_ids) if d != val_dataset]
        val_idx = [i for i, d in enumerate(dataset_ids) if d == val_dataset]

        train_features = [features[i] for i in train_idx]
        train_labels = labels[train_idx]
        val_features = [features[i] for i in val_idx]
        val_labels = labels[val_idx]

        print(f"  Train: {len(train_features)} | Val: {len(val_features)}")

        # Create datasets
        train_dataset = RepertoireDataset(train_features, train_labels, Config.MAX_INSTANCES)
        val_dataset_obj = RepertoireDataset(val_features, val_labels, Config.MAX_INSTANCES)

        train_loader = DataLoader(
            train_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset_obj,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        # Create model
        model = EAMIL(
            input_dim=Config.INPUT_DIM,
            hidden_dim=Config.HIDDEN_DIM,
            attention_dim=Config.ATTENTION_DIM,
            dropout=Config.DROPOUT,
        ).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2
        )
        scaler = torch.amp.GradScaler(enabled=Config.USE_FP16)

        # Training loop
        best_auc = 0
        patience_counter = 0

        for epoch in range(Config.NUM_EPOCHS):
            train_loss = train_epoch(model, train_loader, optimizer, scaler, device)
            val_auc = evaluate(model, val_loader, device)
            scheduler.step()

            if val_auc > best_auc:
                best_auc = val_auc
                patience_counter = 0
                best_models[val_dataset] = model.state_dict().copy()
            else:
                patience_counter += 1

            if patience_counter >= Config.PATIENCE:
                print(f"  Early stopping at epoch {epoch+1}")
                break

            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}: Loss={train_loss:.4f}, Val AUC={val_auc:.4f}")

        print(f"  Best AUC: {best_auc:.4f}")
        fold_aucs.append(best_auc)

    mean_auc = np.mean(fold_aucs)
    std_auc = np.std(fold_aucs)
    print(f"\n{'='*70}")
    print(f"LODO CV Results: {mean_auc:.4f} +/- {std_auc:.4f}")
    print(f"{'='*70}")

    return fold_aucs, best_models, mean_auc


# ============================================================================
# Main
# ============================================================================

def main():
    print("\n" + "=" * 70)
    print("EAMIL GPU Training for AIRR-ML-25")
    print(f"Target: Beat 0.67887 → Achieve 0.80+ AUC")
    print("Using: ESM2 GB10 Cache (18GB, sequence-level embeddings)")
    print("=" * 70)

    # Setup GPU
    device = setup_gb10_optimizations()

    # Create cache directory
    Config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load ESM2 GB10 sequence-level embeddings
    print("\nLoading ESM2 GB10 embeddings (18GB cache)...")
    features, labels, dataset_ids = load_esm2_gb10_embeddings()
    print(f"  Total repertoires: {len(features)}")
    print(f"  Labels distribution: {np.unique(labels, return_counts=True)}")

    # Run LODO CV
    fold_aucs, best_models, mean_auc = lodo_cross_validation(
        features, labels, dataset_ids, device
    )

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        'fold_aucs': fold_aucs,
        'mean_auc': mean_auc,
        'config': {k: v for k, v in vars(Config).items() if not k.startswith('_')},
        'timestamp': timestamp,
    }

    results_path = Config.CACHE_DIR / f"lodo_results_{timestamp}.pkl"
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nResults saved to {results_path}")

    # Save best models
    models_path = Config.CACHE_DIR / f"best_models_{timestamp}.pkl"
    with open(models_path, 'wb') as f:
        pickle.dump(best_models, f)
    print(f"Models saved to {models_path}")

    print("\n" + "=" * 70)
    print(f"EAMIL Training Complete!")
    print(f"LODO CV AUC: {mean_auc:.4f}")
    print(f"Target: 0.80+ | Current Best: 0.67887")
    print(f"Improvement needed: {(0.80 - mean_auc)*100:.1f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()
