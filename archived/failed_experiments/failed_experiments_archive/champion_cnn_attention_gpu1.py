#!/usr/bin/env python3
"""
🏆 AIRR-ML-25 CNN + Attention Model (GPU 1)
================================================================
Alternative architecture for ensemble with ESM-2 model.

Architecture: 1D CNN + Gated Attention (DeepRC-inspired)
- NO ESM-2 (faster, different features)
- Chemical property encoding for amino acids
- 1D CNN for motif extraction
- Gated attention aggregation
- LODO cross-validation

Runs on GPU 1 while ESM-2 model runs on GPU 0.
"""

import os
import gc
import warnings
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')

# ============================================================================
# Configuration - GPU 1
# ============================================================================
class Config:
    # Paths
    DATA_ROOT = Path('./data')
    TRAIN_ROOT = DATA_ROOT / 'train_datasets/train_datasets'
    TEST_ROOT = DATA_ROOT / 'test_datasets/test_datasets'
    OUTPUT_DIR = Path('./submissions')
    MODEL_DIR = Path('./models_cnn_attention')

    # Amino acid encoding
    AA_LIST = 'ACDEFGHIKLMNPQRSTVWY'
    AA_TO_IDX = {aa: i+1 for i, aa in enumerate(AA_LIST)}  # 0 = padding
    VOCAB_SIZE = len(AA_LIST) + 1
    MAX_SEQ_LEN = 25
    MAX_SEQS_PER_REP = 2000  # More sequences since no ESM overhead

    # Chemical properties (normalized)
    AA_PROPERTIES = {
        'A': [0.62, 0.00, 0.50, 0.00], 'C': [0.29, 0.00, 0.35, 0.00],
        'D': [-0.90, -1.00, 0.30, 1.00], 'E': [-0.74, -1.00, 0.40, 1.00],
        'F': [1.19, 0.00, 0.70, 0.00], 'G': [0.48, 0.00, 0.20, 0.00],
        'H': [-0.40, 0.50, 0.55, 1.00], 'I': [1.38, 0.00, 0.60, 0.00],
        'K': [-1.50, 1.00, 0.60, 1.00], 'L': [1.06, 0.00, 0.60, 0.00],
        'M': [0.64, 0.00, 0.60, 0.00], 'N': [-0.78, 0.00, 0.40, 1.00],
        'P': [0.12, 0.00, 0.40, 0.00], 'Q': [-0.85, 0.00, 0.50, 1.00],
        'R': [-2.53, 1.00, 0.65, 1.00], 'S': [-0.18, 0.00, 0.30, 1.00],
        'T': [-0.05, 0.00, 0.40, 1.00], 'V': [1.08, 0.00, 0.50, 0.00],
        'W': [0.81, 0.00, 0.85, 0.00], 'Y': [0.26, 0.00, 0.70, 1.00],
    }  # [hydrophobicity, charge, volume, polarity]
    PROP_DIM = 4

    # Model
    EMBED_DIM = 64
    CNN_CHANNELS = [128, 256, 128]
    KERNEL_SIZES = [3, 5, 7]
    HIDDEN_DIM = 256
    DROPOUT = 0.3

    # Training
    EPOCHS = 40
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 0.01
    PATIENCE = 8

    # Hardware - GPU 1
    DEVICE = 'cuda:1'
    NUM_WORKERS = 4
    USE_AMP = True

    SEED = 42

# Set seeds
torch.manual_seed(Config.SEED)
np.random.seed(Config.SEED)

print(f"🚀 CNN+Attention Model on {Config.DEVICE}")

# ============================================================================
# Amino Acid Encoding
# ============================================================================
def encode_sequence(seq: str, max_len: int = 25) -> Tuple[np.ndarray, np.ndarray]:
    """Encode sequence to indices and chemical properties."""
    seq = seq.upper()[:max_len]

    # Index encoding
    indices = np.zeros(max_len, dtype=np.int64)
    for i, aa in enumerate(seq):
        if aa in Config.AA_TO_IDX:
            indices[i] = Config.AA_TO_IDX[aa]

    # Property encoding
    props = np.zeros((max_len, Config.PROP_DIM), dtype=np.float32)
    for i, aa in enumerate(seq):
        if aa in Config.AA_PROPERTIES:
            props[i] = Config.AA_PROPERTIES[aa]

    return indices, props

def encode_repertoire(sequences: List[str], max_seqs: int = 2000) -> Tuple[np.ndarray, np.ndarray]:
    """Encode all sequences in a repertoire."""
    if len(sequences) > max_seqs:
        np.random.seed(Config.SEED)
        sequences = list(np.random.choice(sequences, max_seqs, replace=False))

    all_indices = []
    all_props = []

    for seq in sequences:
        if len(seq) < 5:
            continue
        idx, props = encode_sequence(seq, Config.MAX_SEQ_LEN)
        all_indices.append(idx)
        all_props.append(props)

    if len(all_indices) == 0:
        return np.zeros((1, Config.MAX_SEQ_LEN), dtype=np.int64), \
               np.zeros((1, Config.MAX_SEQ_LEN, Config.PROP_DIM), dtype=np.float32)

    return np.array(all_indices), np.array(all_props)

# ============================================================================
# Multi-Scale CNN Encoder
# ============================================================================
class MultiScaleCNN(nn.Module):
    """Multi-scale 1D CNN for motif extraction."""

    def __init__(self, embed_dim: int, channels: List[int], kernel_sizes: List[int]):
        super().__init__()

        self.convs = nn.ModuleList()
        for i, (ch, ks) in enumerate(zip(channels, kernel_sizes)):
            in_ch = embed_dim + Config.PROP_DIM if i == 0 else channels[i-1]
            self.convs.append(nn.Sequential(
                nn.Conv1d(in_ch, ch, ks, padding=ks//2),
                nn.BatchNorm1d(ch),
                nn.ReLU(),
                nn.Dropout(0.1)
            ))

        self.output_dim = channels[-1]

    def forward(self, x):
        """x: (batch, seq_len, embed_dim + prop_dim)"""
        x = x.transpose(1, 2)  # (batch, channels, seq_len)

        for conv in self.convs:
            x = conv(x)

        # Global max pooling over sequence
        x = F.adaptive_max_pool1d(x, 1).squeeze(-1)  # (batch, channels)
        return x

# ============================================================================
# Gated Attention
# ============================================================================
class GatedAttention(nn.Module):
    """Gated attention for repertoire aggregation."""

    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()

        self.attention_U = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh()
        )
        self.attention_V = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid()
        )
        self.attention_w = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """x: (batch, n_seqs, dim)"""
        h_U = self.attention_U(x)
        h_V = self.attention_V(x)
        gated = h_U * h_V

        scores = self.attention_w(gated).squeeze(-1)

        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))

        weights = F.softmax(scores, dim=1)
        aggregated = torch.bmm(weights.unsqueeze(1), x).squeeze(1)

        return aggregated, weights

# ============================================================================
# CNN + Attention Classifier
# ============================================================================
class CNNAttentionClassifier(nn.Module):
    """CNN + Gated Attention for repertoire classification."""

    def __init__(self):
        super().__init__()

        # Embedding layer
        self.embedding = nn.Embedding(Config.VOCAB_SIZE, Config.EMBED_DIM, padding_idx=0)

        # Multi-scale CNN
        self.cnn = MultiScaleCNN(
            Config.EMBED_DIM,
            Config.CNN_CHANNELS,
            Config.KERNEL_SIZES
        )

        # Gated attention
        self.attention = GatedAttention(self.cnn.output_dim, Config.HIDDEN_DIM // 2)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.cnn.output_dim, Config.HIDDEN_DIM),
            nn.LayerNorm(Config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT),
            nn.Linear(Config.HIDDEN_DIM, Config.HIDDEN_DIM // 2),
            nn.LayerNorm(Config.HIDDEN_DIM // 2),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT),
            nn.Linear(Config.HIDDEN_DIM // 2, 1)
        )

    def forward(self, indices: torch.Tensor, props: torch.Tensor, mask: torch.Tensor):
        """
        Args:
            indices: (batch, n_seqs, seq_len)
            props: (batch, n_seqs, seq_len, prop_dim)
            mask: (batch, n_seqs)
        """
        batch_size, n_seqs, seq_len = indices.shape

        # Flatten for CNN processing
        indices_flat = indices.view(-1, seq_len)
        props_flat = props.view(-1, seq_len, Config.PROP_DIM)

        # Embed and concatenate with properties
        emb = self.embedding(indices_flat)  # (batch*n_seqs, seq_len, embed_dim)
        x = torch.cat([emb, props_flat], dim=-1)  # (batch*n_seqs, seq_len, embed_dim+prop_dim)

        # CNN encoding
        seq_features = self.cnn(x)  # (batch*n_seqs, cnn_out)

        # Reshape back
        seq_features = seq_features.view(batch_size, n_seqs, -1)  # (batch, n_seqs, cnn_out)

        # Gated attention aggregation
        rep_features, attn_weights = self.attention(seq_features, mask)

        # Classification
        logits = self.classifier(rep_features)

        return logits, attn_weights

# ============================================================================
# Dataset
# ============================================================================
class RepertoireDataset(Dataset):
    def __init__(self, data: List[Dict]):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    max_seqs = max(item['indices'].shape[0] for item in batch)
    batch_size = len(batch)

    indices_padded = torch.zeros(batch_size, max_seqs, Config.MAX_SEQ_LEN, dtype=torch.long)
    props_padded = torch.zeros(batch_size, max_seqs, Config.MAX_SEQ_LEN, Config.PROP_DIM)
    masks = torch.zeros(batch_size, max_seqs, dtype=torch.bool)
    labels = torch.zeros(batch_size)

    for i, item in enumerate(batch):
        n_seqs = item['indices'].shape[0]
        indices_padded[i, :n_seqs] = torch.from_numpy(item['indices'])
        props_padded[i, :n_seqs] = torch.from_numpy(item['props'])
        masks[i, :n_seqs] = True
        labels[i] = item['label']

    return {
        'indices': indices_padded,
        'props': props_padded,
        'masks': masks,
        'labels': labels,
        'repertoire_ids': [item['repertoire_id'] for item in batch]
    }

# ============================================================================
# Data Loading
# ============================================================================
def load_sequences(tsv_path: Path) -> List[str]:
    """Load CDR3 sequences from TSV."""
    try:
        df = pd.read_csv(tsv_path, sep='\t', usecols=['junction_aa'], nrows=Config.MAX_SEQS_PER_REP * 2)
        return df['junction_aa'].dropna().astype(str).tolist()
    except:
        return []

def load_dataset(dataset_id: int) -> List[Dict]:
    """Load and encode one dataset."""
    print(f"\n📂 Loading dataset {dataset_id}...")

    dataset_path = Config.TRAIN_ROOT / f'train_dataset_{dataset_id}'
    metadata = pd.read_csv(dataset_path / 'metadata.csv')

    data = []
    for idx, row in tqdm(metadata.iterrows(), total=len(metadata), desc=f"Dataset {dataset_id}"):
        tsv_path = dataset_path / row['filename']
        sequences = load_sequences(tsv_path)

        if len(sequences) < 10:
            continue

        indices, props = encode_repertoire(sequences, Config.MAX_SEQS_PER_REP)

        data.append({
            'indices': indices,
            'props': props,
            'label': 1 if row['label_positive'] else 0,
            'repertoire_id': row['repertoire_id'],
            'dataset_id': dataset_id
        })

    print(f"  ✓ Loaded {len(data)} repertoires")
    return data

# ============================================================================
# Training
# ============================================================================
def train_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []

    for batch in tqdm(loader, desc="Training", leave=False):
        indices = batch['indices'].to(device)
        props = batch['props'].to(device)
        masks = batch['masks'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        if scaler:
            with autocast():
                logits, _ = model(indices, props, masks)
                loss = criterion(logits.squeeze(), labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits, _ = model(indices, props, masks)
            loss = criterion(logits.squeeze(), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item()
        probs = torch.sigmoid(logits.squeeze()).detach().cpu().numpy()
        all_preds.extend(probs if len(probs.shape) > 0 else [probs.item()])
        all_labels.extend(labels.cpu().numpy())

    auc = roc_auc_score(all_labels, all_preds) if len(set(all_labels)) > 1 else 0.5
    return total_loss / len(loader), auc

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []

    for batch in tqdm(loader, desc="Evaluating", leave=False):
        indices = batch['indices'].to(device)
        props = batch['props'].to(device)
        masks = batch['masks'].to(device)
        labels = batch['labels'].to(device)

        logits, _ = model(indices, props, masks)
        loss = criterion(logits.squeeze(), labels)

        total_loss += loss.item()
        probs = torch.sigmoid(logits.squeeze()).cpu().numpy()
        all_preds.extend(probs if len(probs.shape) > 0 else [probs.item()])
        all_labels.extend(labels.cpu().numpy())

    auc = roc_auc_score(all_labels, all_preds) if len(set(all_labels)) > 1 else 0.5
    return total_loss / len(loader), auc, all_preds

def train_fold(fold_id: int, train_data: List[Dict], val_data: List[Dict], device: str):
    """Train one LODO fold."""
    print(f"\n{'='*60}")
    print(f"🎯 CNN+Attention FOLD {fold_id}: Train={len(train_data)}, Val={len(val_data)}")
    print(f"{'='*60}")

    train_loader = DataLoader(
        RepertoireDataset(train_data), batch_size=Config.BATCH_SIZE,
        shuffle=True, collate_fn=collate_fn, num_workers=Config.NUM_WORKERS, pin_memory=True
    )
    val_loader = DataLoader(
        RepertoireDataset(val_data), batch_size=Config.BATCH_SIZE,
        shuffle=False, collate_fn=collate_fn, num_workers=Config.NUM_WORKERS, pin_memory=True
    )

    model = CNNAttentionClassifier().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)
    scaler = GradScaler() if Config.USE_AMP else None

    best_auc = 0
    patience_counter = 0

    for epoch in range(Config.EPOCHS):
        train_loss, train_auc = train_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_loss, val_auc, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"Epoch {epoch+1:2d} | Train AUC: {train_auc:.4f} | Val AUC: {val_auc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            Config.MODEL_DIR.mkdir(exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'fold_id': fold_id,
                'val_auc': val_auc
            }, Config.MODEL_DIR / f'cnn_fold_{fold_id}_best.pt')
            print(f"  ✓ New best! AUC: {best_auc:.4f}")
        else:
            patience_counter += 1

        if patience_counter >= Config.PATIENCE:
            print(f"  Early stopping")
            break

    checkpoint = torch.load(Config.MODEL_DIR / f'cnn_fold_{fold_id}_best.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, best_auc

# ============================================================================
# Prediction
# ============================================================================
@torch.no_grad()
def predict_test(models: List[nn.Module], device: str) -> pd.DataFrame:
    """Generate test predictions."""
    print("\n📊 Generating test predictions...")
    predictions = []

    for test_dataset in sorted(Config.TEST_ROOT.iterdir()):
        if not test_dataset.name.startswith('test_dataset_'):
            continue

        print(f"  Processing {test_dataset.name}...")
        metadata = pd.read_csv(test_dataset / 'metadata.csv')

        for idx, row in tqdm(metadata.iterrows(), total=len(metadata), leave=False):
            tsv_path = test_dataset / row['filename']
            sequences = load_sequences(tsv_path)

            if len(sequences) < 5:
                prob = 0.5
            else:
                indices, props = encode_repertoire(sequences, Config.MAX_SEQS_PER_REP)

                probs = []
                for model in models:
                    model.eval()
                    idx_t = torch.from_numpy(indices).unsqueeze(0).to(device)
                    props_t = torch.from_numpy(props).unsqueeze(0).to(device)
                    mask = torch.ones(1, indices.shape[0], dtype=torch.bool, device=device)
                    logits, _ = model(idx_t, props_t, mask)
                    probs.append(torch.sigmoid(logits).item())
                prob = np.mean(probs)

            predictions.append({
                'ID': row['repertoire_id'],
                'dataset': test_dataset.name,
                'label_positive_probability': prob
            })

    return pd.DataFrame(predictions)

def generate_task_b() -> pd.DataFrame:
    """Generate Task B predictions."""
    print("\n🧬 Generating Task B...")
    task_b_rows = []

    for dataset_id in range(1, 9):
        train_path = Config.TRAIN_ROOT / f'train_dataset_{dataset_id}'
        if not train_path.exists():
            continue

        metadata = pd.read_csv(train_path / 'metadata.csv')
        all_seqs = []

        for _, row in metadata.iterrows():
            tsv_path = train_path / row['filename']
            if tsv_path.exists():
                df = pd.read_csv(tsv_path, sep='\t', usecols=['junction_aa', 'v_call', 'j_call'], nrows=10000)
                df['label'] = row['label_positive']
                all_seqs.append(df)

        if not all_seqs:
            continue

        combined = pd.concat(all_seqs, ignore_index=True)

        pos_seqs = combined[combined['label'] == True]['junction_aa'].value_counts()
        neg_seqs = combined[combined['label'] == False]['junction_aa'].value_counts()

        all_seq_set = set(pos_seqs.index) | set(neg_seqs.index)
        scores = {}
        for seq in all_seq_set:
            pos_freq = pos_seqs.get(seq, 0) / max(pos_seqs.sum(), 1)
            neg_freq = neg_seqs.get(seq, 0) / max(neg_seqs.sum(), 1)
            scores[seq] = pos_freq - neg_freq

        top_seqs = sorted(scores.items(), key=lambda x: -x[1])[:50000]
        seq_info = combined.drop_duplicates('junction_aa').set_index('junction_aa')

        for rank, (seq, _) in enumerate(top_seqs, 1):
            if pd.isna(seq) or seq == '':
                continue

            v_call = seq_info.loc[seq, 'v_call'] if seq in seq_info.index else 'Unknown'
            j_call = seq_info.loc[seq, 'j_call'] if seq in seq_info.index else 'Unknown'
            if pd.isna(v_call): v_call = 'Unknown'
            if pd.isna(j_call): j_call = 'Unknown'

            task_b_rows.append({
                'ID': f'train_dataset_{dataset_id}_seq_top_{rank}',
                'dataset': f'train_dataset_{dataset_id}',
                'label_positive_probability': -999.0,
                'junction_aa': seq,
                'v_call': str(v_call),
                'j_call': str(j_call)
            })

    return pd.DataFrame(task_b_rows)

# ============================================================================
# Main
# ============================================================================
def main():
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║  🏆 AIRR-ML-25 CNN + Attention Model (GPU 1)                    ║
    ║                                                                  ║
    ║  Architecture: 1D CNN + Gated Attention                         ║
    ║  Features: Chemical property encoding (no ESM-2)                ║
    ║  Purpose: Ensemble with ESM-2 model                             ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)

    device = Config.DEVICE

    # Load all data
    print("\n📥 Loading all training data...")
    all_data = []
    for dataset_id in range(1, 9):
        data = load_dataset(dataset_id)
        all_data.extend(data)
        gc.collect()

    print(f"\n✓ Total: {len(all_data)} repertoires")

    # LODO CV
    print("\n" + "="*60)
    print("🎓 LODO CROSS-VALIDATION (CNN + Attention)")
    print("="*60)

    fold_results = []
    trained_models = []

    for val_id in range(1, 9):
        train_data = [d for d in all_data if d['dataset_id'] != val_id]
        val_data = [d for d in all_data if d['dataset_id'] == val_id]

        if len(val_data) == 0:
            continue

        model, val_auc = train_fold(val_id, train_data, val_data, device)
        fold_results.append({'fold': val_id, 'val_auc': val_auc})
        trained_models.append(model)

        gc.collect()
        torch.cuda.empty_cache()

    # Results
    print("\n" + "="*60)
    print("📈 CNN+Attention LODO RESULTS")
    print("="*60)
    for r in fold_results:
        print(f"  Fold {r['fold']}: AUC = {r['val_auc']:.4f}")

    mean_auc = np.mean([r['val_auc'] for r in fold_results])
    std_auc = np.std([r['val_auc'] for r in fold_results])
    print(f"\n  🎯 Mean LODO AUC: {mean_auc:.4f} ± {std_auc:.4f}")

    # Predictions
    task_a = predict_test(trained_models, device)
    task_b = generate_task_b()

    submission = pd.concat([task_a, task_b], ignore_index=True)
    submission = submission[['ID', 'dataset', 'label_positive_probability', 'junction_aa', 'v_call', 'j_call']]
    submission['junction_aa'] = submission['junction_aa'].fillna('-999.0')
    submission['v_call'] = submission['v_call'].fillna('-999.0')
    submission['j_call'] = submission['j_call'].fillna('-999.0')

    Config.OUTPUT_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = Config.OUTPUT_DIR / f'cnn_attention_submission_{timestamp}.csv'
    submission.to_csv(output_path, index=False)

    print(f"\n✅ Submission: {output_path}")
    print(f"   Rows: {len(submission)}")
    print("\n🏆 CNN+Attention training complete!")

if __name__ == '__main__':
    main()
