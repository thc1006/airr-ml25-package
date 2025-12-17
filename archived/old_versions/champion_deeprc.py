#!/usr/bin/env python3
"""
🏆 AIRR-ML-25 Championship DeepRC-like Model
============================================
Custom implementation of attention-based multiple instance learning
for immune repertoire classification.

Key Features:
- Adaptive sampling for variable-size repertoires (25K - 560K sequences)
- Attention-based aggregation (NOT averaging!)
- Per-dataset training with cross-dataset generalization
- GPU-optimized for RTX 5080 16GB

Architecture:
    Sequences → AA Embedding → 1D-CNN → Attention Pooling → MLP → Prediction
                                              ↑
                                    (learns important sequences)

Target: Beat 0.84590 (current #1)
"""

import os
import gc
import json
import random
import warnings
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ============================================================================
# Configuration
# ============================================================================
class Config:
    """Hyperparameters optimized for AIRR-ML-25"""

    # Paths
    TRAIN_ROOT = './data/train_datasets/train_datasets'
    TEST_ROOT = './data/test_datasets/test_datasets'
    CHECKPOINT_DIR = './checkpoints_deeprc'
    SUBMISSION_DIR = './submissions'

    # Model architecture
    EMBEDDING_DIM = 64          # AA embedding dimension
    CNN_CHANNELS = [128, 256]   # 1D-CNN channel sizes
    CNN_KERNELS = [5, 9]        # Kernel sizes for capturing motifs
    ATTENTION_DIM = 128         # Attention hidden dimension
    CLASSIFIER_DIMS = [256, 128] # MLP classifier dimensions
    DROPOUT = 0.3

    # Training
    BATCH_SIZE = 8              # Repertoires per batch
    MAX_SEQS_PER_REP = 5000     # Max sequences to sample per repertoire
    MAX_SEQ_LENGTH = 30         # Max CDR3 length (truncate longer)
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    NUM_EPOCHS = 30
    EARLY_STOPPING = 7

    # Task B
    TOP_K_SEQUENCES = 50000

    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Dataset-specific configurations
    DATASET_CONFIGS = {
        'train_dataset_1': {'max_seqs': 5000, 'is_simulated': True},
        'train_dataset_2': {'max_seqs': 5000, 'is_simulated': True},
        'train_dataset_3': {'max_seqs': 5000, 'is_simulated': True},
        'train_dataset_4': {'max_seqs': 5000, 'is_simulated': True},
        'train_dataset_5': {'max_seqs': 5000, 'is_simulated': True},
        'train_dataset_6': {'max_seqs': 5000, 'is_simulated': True},
        'train_dataset_7': {'max_seqs': 8000, 'is_simulated': False},  # Real data, larger
        'train_dataset_8': {'max_seqs': 8000, 'is_simulated': False},  # Real data, larger
    }

config = Config()

print(f"🚀 Device: {config.DEVICE}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ============================================================================
# Amino Acid Encoding
# ============================================================================
AA_VOCAB = {
    'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'K': 9, 'L': 10,
    'M': 11, 'N': 12, 'P': 13, 'Q': 14, 'R': 15, 'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20,
    'X': 21, '*': 22, '-': 23, '.': 24,  # Special characters
}
VOCAB_SIZE = len(AA_VOCAB) + 1  # +1 for padding (0)


def encode_sequence(seq: str, max_len: int = 30) -> np.ndarray:
    """Encode amino acid sequence to integer array."""
    encoded = np.zeros(max_len, dtype=np.int32)
    for i, aa in enumerate(seq[:max_len]):
        encoded[i] = AA_VOCAB.get(aa.upper(), AA_VOCAB['X'])
    return encoded


def encode_sequences_batch(sequences: List[str], max_len: int = 30) -> np.ndarray:
    """Encode batch of sequences efficiently."""
    batch = np.zeros((len(sequences), max_len), dtype=np.int32)
    for i, seq in enumerate(sequences):
        for j, aa in enumerate(seq[:max_len]):
            batch[i, j] = AA_VOCAB.get(aa.upper(), AA_VOCAB['X'])
    return batch


# ============================================================================
# Model Architecture
# ============================================================================

class SequenceEncoder(nn.Module):
    """
    Encode CDR3 sequences using embedding + 1D-CNN.

    Input: (batch, seq_len) - integer encoded sequences
    Output: (batch, hidden_dim) - sequence embeddings
    """

    def __init__(self, vocab_size: int = VOCAB_SIZE, embed_dim: int = 64,
                 cnn_channels: List[int] = [128, 256], cnn_kernels: List[int] = [5, 9],
                 dropout: float = 0.3):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # Multi-scale 1D-CNN for capturing different motif sizes
        self.convs = nn.ModuleList()
        for channels, kernel in zip(cnn_channels, cnn_kernels):
            self.convs.append(nn.Sequential(
                nn.Conv1d(embed_dim, channels, kernel, padding=kernel//2),
                nn.BatchNorm1d(channels),
                nn.ReLU(),
                nn.Dropout(dropout)
            ))

        self.output_dim = sum(cnn_channels)

        # Final projection
        self.projection = nn.Sequential(
            nn.Linear(self.output_dim, self.output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len) integer encoded sequences
        Returns:
            (batch, output_dim) sequence embeddings
        """
        # Embedding: (batch, seq_len) -> (batch, seq_len, embed_dim)
        embedded = self.embedding(x)

        # Transpose for Conv1d: (batch, embed_dim, seq_len)
        embedded = embedded.transpose(1, 2)

        # Apply multi-scale convolutions
        conv_outputs = []
        for conv in self.convs:
            out = conv(embedded)
            # Global max pooling over sequence length
            pooled = F.adaptive_max_pool1d(out, 1).squeeze(-1)
            conv_outputs.append(pooled)

        # Concatenate multi-scale features
        combined = torch.cat(conv_outputs, dim=1)

        # Project
        output = self.projection(combined)

        return output


class AttentionAggregator(nn.Module):
    """
    Attention-based aggregation over sequences in a repertoire.

    This is the KEY component that makes DeepRC work:
    - Instead of averaging, we learn which sequences are important
    - Attention weights can be used for Task B (sequence identification)

    Input: (n_seqs, hidden_dim) - sequence embeddings for one repertoire
    Output: (hidden_dim,) - repertoire embedding
    """

    def __init__(self, input_dim: int, attention_dim: int = 128):
        super().__init__()

        # Attention mechanism (single query attending to all sequences)
        self.attention = nn.Sequential(
            nn.Linear(input_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1)
        )

    def forward(self, seq_embeddings: torch.Tensor,
                return_weights: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            seq_embeddings: (n_seqs, hidden_dim) sequence embeddings
            return_weights: whether to return attention weights
        Returns:
            repertoire_embedding: (hidden_dim,)
            attention_weights: (n_seqs,) if return_weights=True
        """
        # Compute attention scores: (n_seqs, 1)
        attn_scores = self.attention(seq_embeddings)

        # Softmax over sequences: (n_seqs, 1)
        attn_weights = F.softmax(attn_scores, dim=0)

        # Weighted sum: (hidden_dim,)
        repertoire_embedding = (seq_embeddings * attn_weights).sum(dim=0)

        if return_weights:
            return repertoire_embedding, attn_weights.squeeze(-1)
        return repertoire_embedding, None


class RepertoireClassifier(nn.Module):
    """
    Full model: Sequence Encoder + Attention Aggregator + Classifier
    """

    def __init__(self, config: Config):
        super().__init__()

        self.sequence_encoder = SequenceEncoder(
            vocab_size=VOCAB_SIZE,
            embed_dim=config.EMBEDDING_DIM,
            cnn_channels=config.CNN_CHANNELS,
            cnn_kernels=config.CNN_KERNELS,
            dropout=config.DROPOUT
        )

        hidden_dim = self.sequence_encoder.output_dim

        self.attention_aggregator = AttentionAggregator(
            input_dim=hidden_dim,
            attention_dim=config.ATTENTION_DIM
        )

        # MLP Classifier (using LayerNorm instead of BatchNorm for single-sample support)
        layers = []
        input_dim = hidden_dim
        for dim in config.CLASSIFIER_DIMS:
            layers.extend([
                nn.Linear(input_dim, dim),
                nn.LayerNorm(dim),
                nn.ReLU(),
                nn.Dropout(config.DROPOUT)
            ])
            input_dim = dim
        layers.append(nn.Linear(input_dim, 1))

        self.classifier = nn.Sequential(*layers)

        self.config = config

    def forward_single_repertoire(self, sequences: torch.Tensor,
                                   return_attention: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Process a single repertoire.

        Args:
            sequences: (n_seqs, seq_len) integer encoded sequences
            return_attention: whether to return attention weights
        Returns:
            logit: scalar prediction logit
            attention_weights: (n_seqs,) if return_attention=True
        """
        # Encode all sequences: (n_seqs, hidden_dim)
        seq_embeddings = self.sequence_encoder(sequences)

        # Aggregate with attention: (hidden_dim,)
        rep_embedding, attn_weights = self.attention_aggregator(
            seq_embeddings, return_weights=return_attention
        )

        # Classify: (1,)
        logit = self.classifier(rep_embedding.unsqueeze(0)).squeeze()

        return logit, attn_weights

    def forward(self, batch_sequences: List[torch.Tensor],
                return_attention: bool = False) -> Tuple[torch.Tensor, List[Optional[torch.Tensor]]]:
        """
        Process a batch of repertoires.

        Args:
            batch_sequences: list of (n_seqs_i, seq_len) tensors
            return_attention: whether to return attention weights
        Returns:
            logits: (batch_size,) prediction logits
            attention_weights: list of attention weights per repertoire
        """
        logits = []
        all_attn_weights = []

        for sequences in batch_sequences:
            logit, attn_weights = self.forward_single_repertoire(
                sequences, return_attention=return_attention
            )
            logits.append(logit)
            all_attn_weights.append(attn_weights)

        return torch.stack(logits), all_attn_weights


# ============================================================================
# Dataset and DataLoader
# ============================================================================

class RepertoireDataset(Dataset):
    """
    Dataset for immune repertoires with adaptive sampling.
    """

    def __init__(self, data_dir: str, metadata: pd.DataFrame,
                 max_seqs: int = 5000, max_seq_len: int = 30,
                 is_train: bool = True):
        self.data_dir = Path(data_dir)
        self.metadata = metadata.reset_index(drop=True)
        self.max_seqs = max_seqs
        self.max_seq_len = max_seq_len
        self.is_train = is_train

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Dict:
        row = self.metadata.iloc[idx]
        filename = row['filename']
        label = row['label_positive'] if 'label_positive' in row else 0
        repertoire_id = row.get('repertoire_id', filename.replace('.tsv', ''))

        # Load sequences
        file_path = self.data_dir / filename
        df = pd.read_csv(file_path, sep='\t', usecols=['junction_aa'])
        sequences = df['junction_aa'].dropna().astype(str).tolist()

        # Sample sequences if too many
        if len(sequences) > self.max_seqs:
            if self.is_train:
                # Random sampling during training
                sequences = random.sample(sequences, self.max_seqs)
            else:
                # Deterministic sampling during evaluation (first N)
                # But shuffle first to get diverse sample
                random.seed(42)
                random.shuffle(sequences)
                sequences = sequences[:self.max_seqs]

        # Encode sequences
        encoded = encode_sequences_batch(sequences, self.max_seq_len)

        return {
            'sequences': torch.from_numpy(encoded).long(),
            'label': torch.tensor(label, dtype=torch.float32),
            'repertoire_id': repertoire_id,
            'n_seqs': len(sequences)
        }


def collate_fn(batch: List[Dict]) -> Dict:
    """Custom collate function for variable-size repertoires."""
    return {
        'sequences': [item['sequences'] for item in batch],
        'labels': torch.stack([item['label'] for item in batch]),
        'repertoire_ids': [item['repertoire_id'] for item in batch],
        'n_seqs': [item['n_seqs'] for item in batch]
    }


# ============================================================================
# Training Functions
# ============================================================================

def train_epoch(model: RepertoireClassifier, dataloader: DataLoader,
                optimizer: torch.optim.Optimizer, criterion: nn.Module,
                device: torch.device, scaler: GradScaler) -> Tuple[float, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    pbar = tqdm(dataloader, desc="Training", leave=False)
    for batch in pbar:
        sequences = [s.to(device) for s in batch['sequences']]
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        with autocast():
            logits, _ = model(sequences)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_preds.extend(probs)
        all_labels.extend(labels.cpu().numpy())

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / len(dataloader)
    auc = roc_auc_score(all_labels, all_preds)

    return avg_loss, auc


def evaluate(model: RepertoireClassifier, dataloader: DataLoader,
             criterion: nn.Module, device: torch.device) -> Tuple[float, float, List, List]:
    """Evaluate model."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_ids = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            sequences = [s.to(device) for s in batch['sequences']]
            labels = batch['labels'].to(device)

            logits, _ = model(sequences)
            loss = criterion(logits, labels)

            total_loss += loss.item()

            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.extend(probs)
            all_labels.extend(labels.cpu().numpy())
            all_ids.extend(batch['repertoire_ids'])

    avg_loss = total_loss / len(dataloader)
    auc = roc_auc_score(all_labels, all_preds)

    return avg_loss, auc, all_preds, all_ids


# ============================================================================
# Main Training Pipeline
# ============================================================================

def train_single_dataset(dataset_name: str, train_data: pd.DataFrame,
                         val_data: pd.DataFrame, data_dir: str,
                         config: Config) -> Tuple[RepertoireClassifier, float]:
    """Train model on a single dataset with validation."""

    print(f"\n{'='*60}")
    print(f"🎯 Training on {dataset_name}")
    print(f"{'='*60}")
    print(f"Train: {len(train_data)} | Val: {len(val_data)}")

    # Get dataset-specific config
    ds_config = config.DATASET_CONFIGS.get(dataset_name, {'max_seqs': 5000})
    max_seqs = ds_config['max_seqs']

    # Create datasets
    train_dataset = RepertoireDataset(
        data_dir, train_data, max_seqs=max_seqs,
        max_seq_len=config.MAX_SEQ_LENGTH, is_train=True
    )
    val_dataset = RepertoireDataset(
        data_dir, val_data, max_seqs=max_seqs,
        max_seq_len=config.MAX_SEQ_LENGTH, is_train=False
    )

    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE,
        shuffle=True, collate_fn=collate_fn, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.BATCH_SIZE,
        shuffle=False, collate_fn=collate_fn, num_workers=4, pin_memory=True
    )

    # Initialize model
    model = RepertoireClassifier(config).to(config.DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    )
    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler()

    # Training loop
    best_auc = 0
    patience_counter = 0
    best_model_state = None

    for epoch in range(config.NUM_EPOCHS):
        train_loss, train_auc = train_epoch(
            model, train_loader, optimizer, criterion, config.DEVICE, scaler
        )
        val_loss, val_auc, _, _ = evaluate(
            model, val_loader, criterion, config.DEVICE
        )

        scheduler.step(val_auc)

        print(f"Epoch {epoch+1:2d} | Train AUC: {train_auc:.4f} | Val AUC: {val_auc:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1

        if patience_counter >= config.EARLY_STOPPING:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    print(f"✅ Best Val AUC: {best_auc:.4f}")

    return model, best_auc


def train_all_datasets(config: Config) -> Dict[str, RepertoireClassifier]:
    """
    Train models for all datasets using leave-one-out or per-dataset strategy.
    """
    print("\n" + "="*70)
    print("🏆 CHAMPIONSHIP DEEPRC TRAINING")
    print("="*70)

    models = {}
    results = []

    # Strategy: Train one model per dataset (matching train to test)
    train_root = Path(config.TRAIN_ROOT)

    for dataset_dir in sorted(train_root.iterdir()):
        if not dataset_dir.is_dir():
            continue

        dataset_name = dataset_dir.name
        metadata_path = dataset_dir / 'metadata.csv'

        if not metadata_path.exists():
            continue

        metadata = pd.read_csv(metadata_path)

        # Split for validation (80/20)
        n_val = max(1, int(len(metadata) * 0.2))
        val_data = metadata.sample(n=n_val, random_state=42)
        train_data = metadata.drop(val_data.index)

        model, val_auc = train_single_dataset(
            dataset_name, train_data, val_data, str(dataset_dir), config
        )

        models[dataset_name] = model
        results.append({'dataset': dataset_name, 'val_auc': val_auc})

        # Save model checkpoint
        os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
        checkpoint_path = os.path.join(config.CHECKPOINT_DIR, f'{dataset_name}_model.pt')
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config.__dict__,
            'val_auc': val_auc
        }, checkpoint_path)
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

    # Save results
    with open(os.path.join(config.CHECKPOINT_DIR, 'training_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    return models


# ============================================================================
# Prediction Functions
# ============================================================================

def predict_test_dataset(model: RepertoireClassifier, test_dir: str,
                         dataset_name: str, config: Config) -> pd.DataFrame:
    """Generate predictions for a test dataset."""

    test_path = Path(test_dir)
    tsv_files = sorted(test_path.glob('*.tsv'))

    print(f"\n📂 Predicting {dataset_name}: {len(tsv_files)} repertoires")

    predictions = []
    model.eval()

    # Get max_seqs from matching training dataset config
    base_name = dataset_name.replace('test_', 'train_').split('_')[0:2]
    base_name = '_'.join(base_name)
    # Handle test_dataset_7_1, test_dataset_8_2, etc.
    if 'train_dataset_7' in base_name or dataset_name.startswith('test_dataset_7'):
        max_seqs = config.DATASET_CONFIGS['train_dataset_7']['max_seqs']
    elif 'train_dataset_8' in base_name or dataset_name.startswith('test_dataset_8'):
        max_seqs = config.DATASET_CONFIGS['train_dataset_8']['max_seqs']
    else:
        max_seqs = config.MAX_SEQS_PER_REP

    with torch.no_grad():
        for tsv_file in tqdm(tsv_files, desc=f"  {dataset_name}", leave=False):
            repertoire_id = tsv_file.stem

            try:
                # Load sequences
                df = pd.read_csv(tsv_file, sep='\t', usecols=['junction_aa'])
                sequences = df['junction_aa'].dropna().astype(str).tolist()

                if len(sequences) == 0:
                    prob = 0.5
                else:
                    # Sample if needed
                    if len(sequences) > max_seqs:
                        random.seed(42)
                        sequences = random.sample(sequences, max_seqs)

                    # Encode and predict
                    encoded = encode_sequences_batch(sequences, config.MAX_SEQ_LENGTH)
                    seq_tensor = torch.from_numpy(encoded).long().to(config.DEVICE)

                    logit, _ = model.forward_single_repertoire(seq_tensor)
                    prob = torch.sigmoid(logit).item()

                predictions.append({
                    'ID': repertoire_id,
                    'dataset': dataset_name,
                    'label_positive_probability': prob,
                    'junction_aa': -999.0,
                    'v_call': -999.0,
                    'j_call': -999.0
                })

            except Exception as e:
                print(f"    ⚠️ Error {repertoire_id}: {e}")
                predictions.append({
                    'ID': repertoire_id,
                    'dataset': dataset_name,
                    'label_positive_probability': 0.5,
                    'junction_aa': -999.0,
                    'v_call': -999.0,
                    'j_call': -999.0
                })

    return pd.DataFrame(predictions)


def generate_task_a_predictions(models: Dict[str, RepertoireClassifier],
                                 config: Config) -> pd.DataFrame:
    """Generate Task A predictions for all test datasets."""

    print("\n" + "="*70)
    print("🔮 TASK A: GENERATING TEST PREDICTIONS")
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
        preds = predict_test_dataset(model, str(test_dir), test_name, config)
        all_predictions.append(preds)

    predictions_df = pd.concat(all_predictions, ignore_index=True)
    print(f"\n✅ Generated {len(predictions_df)} Task A predictions")

    return predictions_df


# ============================================================================
# Task B: Sequence Identification
# ============================================================================

def identify_important_sequences(models: Dict[str, RepertoireClassifier],
                                  config: Config) -> pd.DataFrame:
    """
    Identify top 50,000 disease-associated sequences per training dataset
    using attention weights.
    """

    print("\n" + "="*70)
    print("🧬 TASK B: IDENTIFYING IMPORTANT SEQUENCES")
    print("="*70)

    all_sequences = []
    train_root = Path(config.TRAIN_ROOT)

    for dataset_dir in sorted(train_root.iterdir()):
        if not dataset_dir.is_dir():
            continue

        dataset_name = dataset_dir.name
        print(f"\n📂 Processing {dataset_name}...")

        if dataset_name not in models:
            print(f"  ⚠️ No model for {dataset_name}")
            continue

        model = models[dataset_name]
        model.eval()

        metadata_path = dataset_dir / 'metadata.csv'
        metadata = pd.read_csv(metadata_path)

        # Collect sequences with attention scores from POSITIVE samples
        positive_samples = metadata[metadata['label_positive'] == True]

        sequence_scores = defaultdict(lambda: {'score': 0, 'count': 0, 'v_call': None, 'j_call': None})

        ds_config = config.DATASET_CONFIGS.get(dataset_name, {'max_seqs': 5000})
        max_seqs = ds_config['max_seqs']

        with torch.no_grad():
            for idx, row in tqdm(positive_samples.iterrows(), total=len(positive_samples),
                                  desc=f"  {dataset_name}", leave=False):
                try:
                    file_path = dataset_dir / row['filename']
                    df = pd.read_csv(file_path, sep='\t')

                    sequences = df['junction_aa'].dropna().astype(str).tolist()
                    v_calls = df['v_call'].tolist() if 'v_call' in df.columns else [None] * len(sequences)
                    j_calls = df['j_call'].tolist() if 'j_call' in df.columns else [None] * len(sequences)

                    # Sample if needed (keep original indices for mapping)
                    if len(sequences) > max_seqs:
                        indices = random.sample(range(len(sequences)), max_seqs)
                        sampled_seqs = [sequences[i] for i in indices]
                        sampled_v = [v_calls[i] for i in indices]
                        sampled_j = [j_calls[i] for i in indices]
                    else:
                        sampled_seqs = sequences
                        sampled_v = v_calls
                        sampled_j = j_calls

                    # Get attention weights
                    encoded = encode_sequences_batch(sampled_seqs, config.MAX_SEQ_LENGTH)
                    seq_tensor = torch.from_numpy(encoded).long().to(config.DEVICE)

                    _, attn_weights = model.forward_single_repertoire(seq_tensor, return_attention=True)
                    attn_weights = attn_weights.cpu().numpy()

                    # Accumulate scores
                    for i, (seq, v, j, w) in enumerate(zip(sampled_seqs, sampled_v, sampled_j, attn_weights)):
                        sequence_scores[seq]['score'] += float(w)
                        sequence_scores[seq]['count'] += 1
                        if v and pd.notna(v) and not sequence_scores[seq]['v_call']:
                            sequence_scores[seq]['v_call'] = str(v)
                        if j and pd.notna(j) and not sequence_scores[seq]['j_call']:
                            sequence_scores[seq]['j_call'] = str(j)

                except Exception as e:
                    continue

        # Sort by average score and select top 50,000
        sorted_seqs = sorted(
            sequence_scores.items(),
            key=lambda x: x[1]['score'] / max(x[1]['count'], 1),
            reverse=True
        )[:config.TOP_K_SEQUENCES]

        # Create rows
        for rank, (junction_aa, info) in enumerate(sorted_seqs, 1):
            all_sequences.append({
                'ID': f'{dataset_name}_seq_top_{rank}',
                'dataset': dataset_name,
                'label_positive_probability': -999.0,
                'junction_aa': junction_aa,
                'v_call': info['v_call'] if info['v_call'] else 'TRBV20-1',
                'j_call': info['j_call'] if info['j_call'] else 'TRBJ2-7'
            })

        print(f"  ✓ Selected {len(sorted_seqs)} sequences")
        gc.collect()

    sequences_df = pd.DataFrame(all_sequences)
    print(f"\n✅ Generated {len(sequences_df)} Task B sequences")

    return sequences_df


# ============================================================================
# Submission Generation
# ============================================================================

def generate_submission(task_a_df: pd.DataFrame, task_b_df: pd.DataFrame,
                        config: Config) -> pd.DataFrame:
    """Generate final submission file."""

    print("\n" + "="*70)
    print("📝 GENERATING SUBMISSION")
    print("="*70)

    submission = pd.concat([task_a_df, task_b_df], ignore_index=True)

    # Ensure correct column order
    submission = submission[['ID', 'dataset', 'label_positive_probability',
                             'junction_aa', 'v_call', 'j_call']]

    # Validate
    expected_rows = 4213 + 8 * 50000  # 404,213
    actual_rows = len(submission)

    print(f"  Expected: {expected_rows}")
    print(f"  Actual: {actual_rows}")

    if actual_rows != expected_rows:
        print(f"  ⚠️ Row count mismatch!")
    else:
        print(f"  ✓ Row count matches!")

    # Check for NaN
    nan_count = submission.isna().sum().sum()
    print(f"  NaN values: {nan_count}")

    # Save
    os.makedirs(config.SUBMISSION_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(config.SUBMISSION_DIR, f'deeprc_submission_{timestamp}.csv')
    submission.to_csv(output_path, index=False)

    print(f"\n💾 Saved: {output_path}")
    print(f"   Size: {os.path.getsize(output_path) / 1e6:.2f} MB")

    return submission


# ============================================================================
# Main
# ============================================================================

def main():
    """Full championship pipeline."""

    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║  🏆 AIRR-ML-25 Championship DeepRC Pipeline 🏆                  ║
    ║                                                                  ║
    ║  Architecture: CNN + Attention-based MIL                        ║
    ║  Target: Beat 0.84590 → Win $5,000 + Nature Methods            ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)

    # Step 1: Train all models
    models = train_all_datasets(config)

    # Step 2: Generate Task A predictions
    task_a_df = generate_task_a_predictions(models, config)

    # Step 3: Identify important sequences (Task B)
    task_b_df = identify_important_sequences(models, config)

    # Step 4: Generate submission
    submission = generate_submission(task_a_df, task_b_df, config)

    print("\n" + "="*70)
    print("🏆 CHAMPIONSHIP PIPELINE COMPLETE!")
    print("="*70)
    print(f"\nSubmit with:")
    print(f"  kaggle competitions submit -c adaptive-immune-profiling-challenge-2025 \\")
    print(f"    -f {config.SUBMISSION_DIR}/deeprc_submission_*.csv -m 'DeepRC attention MIL'")


if __name__ == '__main__':
    main()
