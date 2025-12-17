#!/usr/bin/env python3
"""
AIRR-ML-25 Deep Learning Pipeline with PyTorch
================================================
Memory-efficient deep learning using sequence embeddings instead of k-mers.
Uses Transformer encoder to capture sequence patterns.

Key optimizations:
- Character-level embeddings (20 amino acids)
- Batch processing to fit in GPU memory
- Gradient accumulation for large batch sizes
- Mixed precision training (FP16)
"""

import os
import sys
import gc
import glob
import warnings
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# =============================================================================
# Amino Acid Vocabulary
# =============================================================================

AA_VOCAB = {
    'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5,
    'G': 6, 'H': 7, 'I': 8, 'K': 9, 'L': 10,
    'M': 11, 'N': 12, 'P': 13, 'Q': 14, 'R': 15,
    'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20,
    '<PAD>': 0, '<UNK>': 21
}

MAX_SEQ_LEN = 20  # Maximum CDR3 length to process (reduced for GPU memory)
MAX_REPERTOIRE_SIZE = 1000  # Sample up to N sequences per repertoire (reduced for GPU memory)

# =============================================================================
# Dataset
# =============================================================================

class RepertoireDataset(Dataset):
    """Dataset for immune repertoires."""

    def __init__(self, repertoire_features: List[torch.Tensor], labels: List[int]):
        self.features = repertoire_features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def encode_sequence(seq: str, max_len: int = MAX_SEQ_LEN) -> List[int]:
    """Encode amino acid sequence to indices."""
    encoded = [AA_VOCAB.get(aa, AA_VOCAB['<UNK>']) for aa in seq[:max_len]]
    # Pad if needed
    if len(encoded) < max_len:
        encoded += [AA_VOCAB['<PAD>']] * (max_len - len(encoded))
    return encoded


def load_repertoire_sequences(file_path: str, sample_size: int = MAX_REPERTOIRE_SIZE) -> torch.Tensor:
    """Load and encode sequences from a repertoire file."""
    try:
        df = pd.read_csv(file_path, sep='\t', usecols=['junction_aa'])
        sequences = df['junction_aa'].dropna().tolist()

        # Sample if too large
        if len(sequences) > sample_size:
            sequences = np.random.choice(sequences, sample_size, replace=False).tolist()

        # Encode sequences
        encoded = [encode_sequence(seq) for seq in sequences]

        # Pad repertoire to fixed size
        while len(encoded) < sample_size:
            encoded.append([AA_VOCAB['<PAD>']] * MAX_SEQ_LEN)

        return torch.tensor(encoded[:sample_size], dtype=torch.long)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        # Return dummy repertoire
        return torch.zeros((sample_size, MAX_SEQ_LEN), dtype=torch.long)


# =============================================================================
# Deep Learning Model
# =============================================================================

class TransformerRepertoireClassifier(nn.Module):
    """
    Transformer-based repertoire classifier.

    Architecture:
    1. Embedding layer (amino acid → vector)
    2. Sequence-level Transformer encoder
    3. Mean pooling across sequences
    4. MLP classifier
    """

    def __init__(
        self,
        vocab_size: int = 22,
        embed_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        hidden_dim: int = 128,
        dropout: float = 0.3
    ):
        super().__init__()

        # Embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # Positional encoding
        self.pos_encoder = nn.Parameter(torch.randn(1, MAX_SEQ_LEN, embed_dim))

        # Transformer encoder for sequences
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Repertoire-level aggregation
        self.repertoire_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        """
        Args:
            x: (batch_size, num_sequences, seq_len)

        Returns:
            logits: (batch_size, 1)
        """
        batch_size, num_seqs, seq_len = x.shape

        # Reshape to (batch_size * num_seqs, seq_len)
        x = x.view(-1, seq_len)

        # Embed: (batch_size * num_seqs, seq_len, embed_dim)
        x = self.embedding(x)
        x = x + self.pos_encoder

        # Transformer encode sequences
        x = self.transformer(x)

        # Mean pool over sequence length: (batch_size * num_seqs, embed_dim)
        mask = (x != 0).any(dim=-1).float().unsqueeze(-1)
        x = (x * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)

        # Reshape back: (batch_size, num_seqs, embed_dim)
        x = x.view(batch_size, num_seqs, -1)

        # Aggregate across sequences with attention
        x, _ = self.repertoire_attn(x, x, x)
        x = x.mean(dim=1)  # (batch_size, embed_dim)

        # Classify
        logits = self.classifier(x)

        return logits


# =============================================================================
# Training
# =============================================================================

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 20,
    lr: float = 1e-3,
    device: str = 'cuda'
):
    """Train the model with mixed precision."""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
    scaler = GradScaler()
    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        # Train
        model.train()
        train_loss = 0

        for batch_x, batch_y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device).float().unsqueeze(1)

            optimizer.zero_grad()

            with autocast():
                logits = model(batch_x)
                loss = criterion(logits, batch_y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validate
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device).float().unsqueeze(1)

                with autocast():
                    logits = model(batch_x)
                    loss = criterion(logits, batch_y)

                val_loss += loss.item()

        val_loss /= len(val_loader)

        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')

        scheduler.step()

    # Load best model
    model.load_state_dict(torch.load('best_model.pth'))
    return model


# =============================================================================
# Main Pipeline
# =============================================================================

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Load data
    print("\nLoading training data...")
    train_dir = "./data/train_datasets/train_datasets/train_dataset_1"
    metadata = pd.read_csv(os.path.join(train_dir, 'metadata.csv'))

    repertoire_features = []
    labels = []

    for _, row in tqdm(metadata.iterrows(), total=len(metadata)):
        file_path = os.path.join(train_dir, row['filename'])
        features = load_repertoire_sequences(file_path)
        repertoire_features.append(features)
        labels.append(int(row['label_positive']))

    # Split train/val
    from sklearn.model_selection import train_test_split
    train_idx, val_idx = train_test_split(
        range(len(labels)),
        test_size=0.2,
        stratify=labels,
        random_state=42
    )

    train_features = [repertoire_features[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    val_features = [repertoire_features[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]

    # Create dataloaders
    train_dataset = RepertoireDataset(train_features, train_labels)
    val_dataset = RepertoireDataset(val_features, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)  # Reduced batch size for GPU memory
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

    print(f"\nTrain samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # Create model
    print("\nInitializing model...")
    model = TransformerRepertoireClassifier(
        vocab_size=len(AA_VOCAB),
        embed_dim=64,
        num_heads=4,
        num_layers=2,
        hidden_dim=128,
        dropout=0.3
    )

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Train
    print("\nTraining...")
    model = train_model(
        model,
        train_loader,
        val_loader,
        num_epochs=20,
        lr=1e-3,
        device=device
    )

    print("\n✅ Training complete!")
    print("Model saved to: best_model.pth")


if __name__ == '__main__':
    main()
