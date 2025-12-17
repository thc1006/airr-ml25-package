#!/usr/bin/env python3
"""
Champion Attention MIL Pipeline for AIRR-ML-25 Competition
==========================================================

Optimized for:
- RTX 5080 (16GB VRAM)
- 32GB System RAM
- AMD Ryzen 7 7800X3D

Key Features:
- Attention-based Multiple Instance Learning (DeepRC-style)
- Mixed precision training (FP16)
- Gradient accumulation for large batches
- Aggressive caching to avoid recomputation
- Memory-efficient data loading

Author: Claude Code
Date: 2025-12-15
"""

import os
import sys
import json
import hashlib
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

# ============================================================================
# Configuration
# ============================================================================

@dataclass
class Config:
    """Configuration optimized for RTX 5080 (16GB VRAM)"""
    # Paths
    checkpoint_dir: str = "./checkpoints"
    cache_dir: str = "./cache_attention_mil"
    output_dir: str = "./submissions"
    train_data_root: str = "./data/train_datasets/train_datasets"
    test_data_root: str = "./data/test_datasets/test_datasets"

    # Model architecture
    esm_dim: int = 1280  # ESM embedding dimension
    trad_dim: int = 389  # Traditional feature dimension
    hidden_dim: int = 256  # Attention hidden dimension
    num_heads: int = 4  # Multi-head attention
    dropout: float = 0.3

    # Training - optimized for 16GB VRAM
    batch_size: int = 4  # Small batch size to fit in VRAM
    gradient_accumulation_steps: int = 8  # Effective batch = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    num_epochs: int = 30
    patience: int = 7  # Early stopping

    # Memory optimization
    max_sequences_per_repertoire: int = 500  # Limit sequences
    use_mixed_precision: bool = True
    pin_memory: bool = True
    num_workers: int = 4

    # Reproducibility
    seed: int = 42

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ============================================================================
# Caching System
# ============================================================================

class CacheManager:
    """Manages caching of intermediate results to avoid recomputation"""

    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.cache_dir / "manifest.json"
        self.manifest = self._load_manifest()

    def _load_manifest(self) -> Dict:
        if self.manifest_path.exists():
            with open(self.manifest_path, 'r') as f:
                return json.load(f)
        return {"version": "1.0", "entries": {}}

    def _save_manifest(self):
        with open(self.manifest_path, 'w') as f:
            json.dump(self.manifest, f, indent=2)

    def _compute_hash(self, data: Any) -> str:
        """Compute hash for cache key"""
        if isinstance(data, (np.ndarray,)):
            return hashlib.md5(data.tobytes()[:1000]).hexdigest()[:16]
        return hashlib.md5(str(data).encode()).hexdigest()[:16]

    def get(self, key: str) -> Optional[Any]:
        """Get cached data"""
        cache_path = self.cache_dir / f"{key}.pkl"
        if cache_path.exists() and key in self.manifest["entries"]:
            print(f"  [Cache HIT] Loading {key}")
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        return None

    def set(self, key: str, data: Any, metadata: Dict = None):
        """Cache data"""
        cache_path = self.cache_dir / f"{key}.pkl"
        print(f"  [Cache SAVE] Saving {key}")
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)

        self.manifest["entries"][key] = {
            "path": str(cache_path),
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        self._save_manifest()

    def exists(self, key: str) -> bool:
        """Check if cache exists"""
        cache_path = self.cache_dir / f"{key}.pkl"
        return cache_path.exists() and key in self.manifest["entries"]


# ============================================================================
# Data Loading
# ============================================================================

class AIRRDataset(Dataset):
    """Dataset for AIRR immune repertoire data"""

    def __init__(
        self,
        esm_embeddings: np.ndarray,
        trad_features: np.ndarray,
        labels: np.ndarray,
        repertoire_ids: np.ndarray,
        max_seq: int = 500
    ):
        self.esm_embeddings = esm_embeddings  # (N, seq, dim)
        self.trad_features = trad_features  # (N, feat_dim)
        self.labels = labels  # (N,)
        self.repertoire_ids = repertoire_ids
        self.max_seq = max_seq

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        esm = self.esm_embeddings[idx]  # (seq, dim)
        trad = self.trad_features[idx]  # (feat_dim,)
        label = self.labels[idx]

        # Limit sequences if needed
        if esm.shape[0] > self.max_seq:
            # Random sampling for training diversity
            indices = np.random.choice(esm.shape[0], self.max_seq, replace=False)
            esm = esm[indices]

        return {
            'esm': torch.tensor(esm, dtype=torch.float32),
            'trad': torch.tensor(trad, dtype=torch.float32),
            'label': torch.tensor(label, dtype=torch.float32),
            'idx': idx
        }


def collate_fn(batch):
    """Custom collate function for variable-length sequences"""
    esm_list = [item['esm'] for item in batch]
    trad_list = [item['trad'] for item in batch]
    labels = torch.stack([item['label'] for item in batch])
    indices = [item['idx'] for item in batch]

    # Pad ESM embeddings to max length in batch
    max_len = max(e.shape[0] for e in esm_list)
    esm_dim = esm_list[0].shape[1]

    padded_esm = torch.zeros(len(batch), max_len, esm_dim)
    mask = torch.zeros(len(batch), max_len, dtype=torch.bool)

    for i, esm in enumerate(esm_list):
        seq_len = esm.shape[0]
        padded_esm[i, :seq_len] = esm
        mask[i, :seq_len] = True

    trad = torch.stack(trad_list)

    return {
        'esm': padded_esm,
        'trad': trad,
        'mask': mask,
        'label': labels,
        'idx': indices
    }


# ============================================================================
# Attention MIL Model (DeepRC-inspired)
# ============================================================================

class GatedAttentionMIL(nn.Module):
    """
    Gated Attention-based Multiple Instance Learning

    Inspired by DeepRC (Widrich et al., NeurIPS 2020)
    Modern Hopfield Networks interpretation
    """

    def __init__(
        self,
        esm_dim: int = 1280,
        trad_dim: int = 389,
        hidden_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.3
    ):
        super().__init__()

        # Instance encoder (per-sequence)
        self.instance_encoder = nn.Sequential(
            nn.Linear(esm_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # Gated attention mechanism
        self.attention_V = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        self.attention_U = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        self.attention_W = nn.Linear(hidden_dim, 1)

        # Multi-head self-attention for instances
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Traditional feature encoder
        self.trad_encoder = nn.Sequential(
            nn.Linear(trad_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Classifier (combines bag representation + traditional features)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(64, 1)
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier/Kaiming initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        esm: torch.Tensor,
        trad: torch.Tensor,
        mask: torch.Tensor,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass

        Args:
            esm: (batch, seq, esm_dim) - ESM embeddings per sequence
            trad: (batch, trad_dim) - Traditional features
            mask: (batch, seq) - Valid sequence mask
            return_attention: Whether to return attention weights

        Returns:
            Dictionary with 'logits' and optionally 'attention'
        """
        batch_size, seq_len, _ = esm.shape

        # Encode instances
        h = self.instance_encoder(esm)  # (batch, seq, hidden)

        # Self-attention between instances
        # Create attention mask for padding
        attn_mask = ~mask  # Invert: True where padding
        h_attn, _ = self.self_attention(
            h, h, h,
            key_padding_mask=attn_mask
        )
        h = h + h_attn  # Residual connection

        # Gated attention pooling
        V = self.attention_V(h)  # (batch, seq, hidden)
        U = self.attention_U(h)  # (batch, seq, hidden)

        # Gated attention scores
        attention_scores = self.attention_W(V * U).squeeze(-1)  # (batch, seq)

        # Mask padding positions
        attention_scores = attention_scores.masked_fill(~mask, float('-inf'))

        # Softmax over valid sequences
        attention_weights = F.softmax(attention_scores, dim=1)  # (batch, seq)

        # Weighted sum (bag representation)
        bag_repr = torch.bmm(
            attention_weights.unsqueeze(1),  # (batch, 1, seq)
            h  # (batch, seq, hidden)
        ).squeeze(1)  # (batch, hidden)

        # Encode traditional features
        trad_repr = self.trad_encoder(trad)  # (batch, hidden)

        # Combine and classify
        combined = torch.cat([bag_repr, trad_repr], dim=1)  # (batch, hidden*2)
        logits = self.classifier(combined).squeeze(-1)  # (batch,)

        result = {'logits': logits}
        if return_attention:
            result['attention'] = attention_weights

        return result


# ============================================================================
# Training Loop
# ============================================================================

class Trainer:
    """Training manager with early stopping and checkpointing"""

    def __init__(
        self,
        model: nn.Module,
        config: Config,
        cache_manager: CacheManager,
        dataset_id: int
    ):
        self.model = model.to(config.device)
        self.config = config
        self.cache = cache_manager
        self.dataset_id = dataset_id

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,
            T_mult=2
        )

        # Mixed precision scaler
        self.scaler = GradScaler('cuda') if config.use_mixed_precision else None

        # Loss function with class weighting
        self.criterion = nn.BCEWithLogitsLoss()

        # Tracking
        self.best_auc = 0.0
        self.patience_counter = 0
        self.training_history = []

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        self.optimizer.zero_grad()

        pbar = tqdm(dataloader, desc=f"Training DS{self.dataset_id}")
        for step, batch in enumerate(pbar):
            # Move to device
            esm = batch['esm'].to(self.config.device)
            trad = batch['trad'].to(self.config.device)
            mask = batch['mask'].to(self.config.device)
            labels = batch['label'].to(self.config.device)

            # Forward pass with mixed precision
            if self.config.use_mixed_precision:
                with autocast('cuda'):
                    output = self.model(esm, trad, mask)
                    loss = self.criterion(output['logits'], labels)
                    loss = loss / self.config.gradient_accumulation_steps

                self.scaler.scale(loss).backward()
            else:
                output = self.model(esm, trad, mask)
                loss = self.criterion(output['logits'], labels)
                loss = loss / self.config.gradient_accumulation_steps
                loss.backward()

            # Gradient accumulation
            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                if self.config.use_mixed_precision:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()

                self.optimizer.zero_grad()

            # Track metrics
            total_loss += loss.item() * self.config.gradient_accumulation_steps
            preds = torch.sigmoid(output['logits']).detach().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        # Compute epoch metrics
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        auc = roc_auc_score(all_labels, all_preds) if len(np.unique(all_labels)) > 1 else 0.5

        return {
            'loss': total_loss / len(dataloader),
            'auc': auc
        }

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate on validation set"""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        all_attention = []
        all_indices = []

        for batch in dataloader:
            esm = batch['esm'].to(self.config.device)
            trad = batch['trad'].to(self.config.device)
            mask = batch['mask'].to(self.config.device)
            labels = batch['label'].to(self.config.device)

            if self.config.use_mixed_precision:
                with autocast('cuda'):
                    output = self.model(esm, trad, mask, return_attention=True)
                    loss = self.criterion(output['logits'], labels)
            else:
                output = self.model(esm, trad, mask, return_attention=True)
                loss = self.criterion(output['logits'], labels)

            total_loss += loss.item()
            preds = torch.sigmoid(output['logits']).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            all_attention.append(output['attention'].cpu().numpy())
            all_indices.extend(batch['idx'])

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        auc = roc_auc_score(all_labels, all_preds) if len(np.unique(all_labels)) > 1 else 0.5

        return {
            'loss': total_loss / len(dataloader),
            'auc': auc,
            'predictions': all_preds,
            'labels': all_labels,
            'attention': np.concatenate(all_attention, axis=0),
            'indices': all_indices
        }

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader
    ) -> Dict[str, Any]:
        """Full training loop with early stopping"""

        print(f"\n{'='*60}")
        print(f"Training Dataset {self.dataset_id}")
        print(f"{'='*60}")

        for epoch in range(self.config.num_epochs):
            # Training
            train_metrics = self.train_epoch(train_loader)

            # Validation
            val_metrics = self.evaluate(val_loader)

            # Learning rate scheduling
            self.scheduler.step()

            # Logging
            print(f"Epoch {epoch+1}/{self.config.num_epochs}")
            print(f"  Train: Loss={train_metrics['loss']:.4f}, AUC={train_metrics['auc']:.4f}")
            print(f"  Val:   Loss={val_metrics['loss']:.4f}, AUC={val_metrics['auc']:.4f}")

            # Save history
            self.training_history.append({
                'epoch': epoch + 1,
                'train_loss': train_metrics['loss'],
                'train_auc': train_metrics['auc'],
                'val_loss': val_metrics['loss'],
                'val_auc': val_metrics['auc']
            })

            # Early stopping
            if val_metrics['auc'] > self.best_auc:
                self.best_auc = val_metrics['auc']
                self.patience_counter = 0

                # Save best model
                self._save_checkpoint(epoch, val_metrics)
                print(f"  -> New best AUC: {self.best_auc:.4f}")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config.patience:
                    print(f"  -> Early stopping at epoch {epoch+1}")
                    break

        # Load best model
        self._load_best_checkpoint()

        return {
            'best_auc': self.best_auc,
            'history': self.training_history
        }

    def _save_checkpoint(self, epoch: int, metrics: Dict):
        """Save model checkpoint"""
        checkpoint_path = Path(self.config.cache_dir) / f"model_ds{self.dataset_id}_best.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config
        }, checkpoint_path)

    def _load_best_checkpoint(self):
        """Load best model checkpoint"""
        checkpoint_path = Path(self.config.cache_dir) / f"model_ds{self.dataset_id}_best.pt"
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=self.config.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded best model from epoch {checkpoint['epoch']+1}")


# ============================================================================
# Main Pipeline
# ============================================================================

class ChampionAttentionMIL:
    """Main pipeline for Attention MIL training and inference"""

    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.cache = CacheManager(self.config.cache_dir)
        self.models = {}  # Per-dataset models
        self.results = {}

        # Create output directories
        Path(self.config.cache_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

        set_seed(self.config.seed)

    def load_dataset(self, dataset_id: int) -> Tuple[np.ndarray, ...]:
        """Load dataset from checkpoint cache"""
        cache_key = f"loaded_ds{dataset_id}_v2"  # New version with proper dtype

        # Check cache
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        # Load from npz
        npz_path = Path(self.config.checkpoint_dir) / f"dataset_{dataset_id}.npz"
        print(f"Loading dataset {dataset_id} from {npz_path}")

        data = np.load(npz_path, allow_pickle=True)

        # Convert ESM embeddings from object array to float32
        esm_raw = data['esm_embeddings']
        if esm_raw.dtype == object:
            print(f"  Converting ESM embeddings from object to float32...")
            # Stack the array of arrays into a proper 3D array
            esm_embeddings = np.stack([x.astype(np.float32) for x in esm_raw])
        else:
            esm_embeddings = esm_raw.astype(np.float32)

        # Convert traditional features
        trad_raw = data['trad_features']
        if trad_raw.dtype == object:
            trad_features = np.stack([x.astype(np.float32) for x in trad_raw])
        else:
            trad_features = trad_raw.astype(np.float32)

        result = (
            esm_embeddings,
            trad_features,
            data['labels'],
            data['repertoire_ids'],
            data['dataset_ids']
        )

        # Cache loaded data
        self.cache.set(cache_key, result, {'source': str(npz_path)})

        return result

    def train_single_dataset(self, dataset_id: int) -> Dict[str, Any]:
        """Train model on a single dataset with cross-validation"""

        print(f"\n{'#'*60}")
        print(f"# Dataset {dataset_id}")
        print(f"{'#'*60}")

        # Load data
        esm, trad, labels, rep_ids, ds_ids = self.load_dataset(dataset_id)
        print(f"Data shape: ESM={esm.shape}, Trad={trad.shape}, Labels={labels.shape}")
        print(f"Label distribution: {np.bincount(labels.astype(int))}")

        # Create CV splits
        n_folds = 5
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.config.seed)

        fold_results = []
        all_val_preds = np.zeros(len(labels))
        all_attention = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(esm, labels)):
            print(f"\n--- Fold {fold+1}/{n_folds} ---")

            # Create datasets
            train_dataset = AIRRDataset(
                esm[train_idx], trad[train_idx], labels[train_idx], rep_ids[train_idx],
                max_seq=self.config.max_sequences_per_repertoire
            )
            val_dataset = AIRRDataset(
                esm[val_idx], trad[val_idx], labels[val_idx], rep_ids[val_idx],
                max_seq=self.config.max_sequences_per_repertoire
            )

            # Create dataloaders
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                collate_fn=collate_fn,
                num_workers=self.config.num_workers,
                pin_memory=self.config.pin_memory
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=self.config.num_workers,
                pin_memory=self.config.pin_memory
            )

            # Initialize model
            model = GatedAttentionMIL(
                esm_dim=self.config.esm_dim,
                trad_dim=self.config.trad_dim,
                hidden_dim=self.config.hidden_dim,
                num_heads=self.config.num_heads,
                dropout=self.config.dropout
            )

            # Train
            trainer = Trainer(model, self.config, self.cache, dataset_id)
            result = trainer.train(train_loader, val_loader)
            fold_results.append(result)

            # Get final predictions
            final_metrics = trainer.evaluate(val_loader)
            all_val_preds[val_idx] = final_metrics['predictions']
            all_attention.append((val_idx, final_metrics['attention']))

            # Clear GPU memory
            del model, trainer
            torch.cuda.empty_cache()

        # Compute overall CV score
        cv_auc = roc_auc_score(labels, all_val_preds)
        print(f"\n{'='*40}")
        print(f"Dataset {dataset_id} CV AUC: {cv_auc:.4f}")
        print(f"{'='*40}")

        # Train final model on all data
        print(f"\nTraining final model on all data...")
        full_dataset = AIRRDataset(
            esm, trad, labels, rep_ids,
            max_seq=self.config.max_sequences_per_repertoire
        )
        full_loader = DataLoader(
            full_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory
        )

        final_model = GatedAttentionMIL(
            esm_dim=self.config.esm_dim,
            trad_dim=self.config.trad_dim,
            hidden_dim=self.config.hidden_dim,
            num_heads=self.config.num_heads,
            dropout=self.config.dropout
        ).to(self.config.device)

        # Train with reduced epochs for final model
        optimizer = torch.optim.AdamW(
            final_model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        scaler = GradScaler('cuda') if self.config.use_mixed_precision else None
        criterion = nn.BCEWithLogitsLoss()

        final_model.train()
        for epoch in range(min(15, self.config.num_epochs)):
            epoch_loss = 0.0
            optimizer.zero_grad()

            for step, batch in enumerate(full_loader):
                esm_b = batch['esm'].to(self.config.device)
                trad_b = batch['trad'].to(self.config.device)
                mask_b = batch['mask'].to(self.config.device)
                labels_b = batch['label'].to(self.config.device)

                if self.config.use_mixed_precision:
                    with autocast('cuda'):
                        output = final_model(esm_b, trad_b, mask_b)
                        loss = criterion(output['logits'], labels_b)
                        loss = loss / self.config.gradient_accumulation_steps
                    scaler.scale(loss).backward()

                    if (step + 1) % self.config.gradient_accumulation_steps == 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(final_model.parameters(), 1.0)
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                else:
                    output = final_model(esm_b, trad_b, mask_b)
                    loss = criterion(output['logits'], labels_b)
                    loss.backward()
                    if (step + 1) % self.config.gradient_accumulation_steps == 0:
                        optimizer.step()
                        optimizer.zero_grad()

                epoch_loss += loss.item()

            print(f"  Final model epoch {epoch+1}: Loss={epoch_loss/len(full_loader):.4f}")

        # Save final model
        model_path = Path(self.config.cache_dir) / f"final_model_ds{dataset_id}.pt"
        torch.save({
            'model_state_dict': final_model.state_dict(),
            'config': self.config,
            'cv_auc': cv_auc
        }, model_path)

        self.models[dataset_id] = final_model

        result = {
            'dataset_id': dataset_id,
            'cv_auc': cv_auc,
            'fold_results': fold_results,
            'cv_predictions': all_val_preds,
            'attention': all_attention
        }

        self.results[dataset_id] = result

        # Cache results
        self.cache.set(f"results_ds{dataset_id}", result)

        return result

    def train_all_datasets(self) -> Dict[str, Any]:
        """Train on all 8 datasets"""
        all_results = {}

        for ds_id in range(1, 9):
            result = self.train_single_dataset(ds_id)
            all_results[ds_id] = result

            # Force garbage collection
            import gc
            gc.collect()
            torch.cuda.empty_cache()

        # Summary
        print(f"\n{'='*60}")
        print("Training Summary")
        print(f"{'='*60}")
        for ds_id, res in all_results.items():
            print(f"Dataset {ds_id}: CV AUC = {res['cv_auc']:.4f}")

        mean_auc = np.mean([r['cv_auc'] for r in all_results.values()])
        print(f"\nMean CV AUC: {mean_auc:.4f}")

        return all_results

    @torch.no_grad()
    def predict_test(self, test_dataset_id: int, model_dataset_id: int) -> np.ndarray:
        """Predict on test dataset using trained model"""
        # Load model if not in memory
        if model_dataset_id not in self.models:
            model_path = Path(self.config.cache_dir) / f"final_model_ds{model_dataset_id}.pt"
            checkpoint = torch.load(model_path, map_location=self.config.device)

            model = GatedAttentionMIL(
                esm_dim=self.config.esm_dim,
                trad_dim=self.config.trad_dim,
                hidden_dim=self.config.hidden_dim,
                num_heads=self.config.num_heads,
                dropout=self.config.dropout
            ).to(self.config.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            self.models[model_dataset_id] = model

        model = self.models[model_dataset_id]
        model.eval()

        # Load test data (need to implement this based on test data format)
        # For now, return placeholder
        print(f"Predicting test_dataset_{test_dataset_id} with model from ds{model_dataset_id}")

        return np.array([])  # Placeholder

    def identify_important_sequences(
        self,
        dataset_id: int,
        top_k: int = 50000
    ) -> pd.DataFrame:
        """Identify top-k important sequences using attention weights"""

        if dataset_id not in self.results:
            raise ValueError(f"Dataset {dataset_id} not trained yet")

        # Get attention weights
        attention_data = self.results[dataset_id]['attention']

        # Load original data to get sequence information
        esm, trad, labels, rep_ids, ds_ids = self.load_dataset(dataset_id)

        # Aggregate attention across all repertoires
        # This is a simplified approach - in practice we'd need the original sequences
        print(f"Identifying top {top_k} sequences for dataset {dataset_id}")

        # Placeholder - need to implement sequence extraction
        return pd.DataFrame()


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main entry point"""
    print("="*60)
    print("Champion Attention MIL Pipeline")
    print("="*60)
    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print("="*60)

    # Initialize
    config = Config()
    pipeline = ChampionAttentionMIL(config)

    # Train single dataset first as test
    result = pipeline.train_single_dataset(1)
    print(f"\nDataset 1 Result: CV AUC = {result['cv_auc']:.4f}")

    return pipeline


if __name__ == "__main__":
    main()
