#!/usr/bin/env python3
"""
🏆 AIRR-ML-25 BREAKTHROUGH Model
================================================================
Target: Beat SajayR (0.84590) → Achieve 0.85+

Based on 2025 Research Findings:
1. ESM-2 Layer 6 (not Layer 33!) - optimal for TCR CDR3
2. Gated Attention MIL (DeepRC-style) - learns which sequences matter
3. LODO CV - true cross-dataset generalization
4. NO dataset-specific features (V/J gene counts removed)
5. Focus on biological signal only

Key Architecture:
  CDR3 sequences → ESM-2 Layer 6 → Gated Attention Pooling → Classifier

Deadline: Dec 17, 2025 06:59 UTC
"""

import os
import gc
import sys
import warnings
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional

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
# Configuration
# ============================================================================
class Config:
    # Paths
    DATA_ROOT = Path('./data')
    TRAIN_ROOT = DATA_ROOT / 'train_datasets/train_datasets'
    TEST_ROOT = DATA_ROOT / 'test_datasets/test_datasets'
    OUTPUT_DIR = Path('./submissions')
    MODEL_DIR = Path('./models_breakthrough')

    # ESM-2 Settings (CRITICAL: Use Layer 6, not 33!)
    ESM_MODEL = 'facebook/esm2_t33_650M_UR50D'
    ESM_LAYER = 6  # Research shows L6 > L33 for TCR CDR3
    ESM_DIM = 1280
    ESM_BATCH_SIZE = 32
    MAX_SEQ_LEN = 30
    MAX_SEQS_PER_REP = 500  # Reduced for memory efficiency

    # Training Settings
    EPOCHS = 30
    BATCH_SIZE = 8
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 0.01
    PATIENCE = 7

    # Model Architecture
    HIDDEN_DIM = 256
    ATTENTION_HEADS = 4
    DROPOUT = 0.3

    # Hardware
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    NUM_WORKERS = 4
    USE_AMP = True  # Mixed precision

    # Random seed
    SEED = 42

# Set seeds
torch.manual_seed(Config.SEED)
np.random.seed(Config.SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(Config.SEED)

print(f"🚀 Using device: {Config.DEVICE}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ============================================================================
# ESM-2 Feature Extractor (Layer 6 - CRITICAL)
# ============================================================================
class ESM2Extractor:
    """Extract ESM-2 Layer 6 embeddings for CDR3 sequences."""

    def __init__(self, device='cuda'):
        self.device = device
        print(f"\n📦 Loading ESM-2 model (Layer {Config.ESM_LAYER})...")

        try:
            import esm
            self.model, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
            self.model = self.model.to(device).eval()
            self.batch_converter = self.alphabet.get_batch_converter()
            print(f"✓ ESM-2 loaded on {device}")
        except Exception as e:
            print(f"Installing fair-esm...")
            os.system("pip install fair-esm -q")
            import esm
            self.model, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
            self.model = self.model.to(device).eval()
            self.batch_converter = self.alphabet.get_batch_converter()
            print(f"✓ ESM-2 loaded on {device}")

    @torch.no_grad()
    def extract_embeddings(self, sequences: List[str], batch_size: int = 32) -> np.ndarray:
        """Extract Layer 6 embeddings."""
        if len(sequences) == 0:
            return np.zeros((1, Config.ESM_DIM), dtype=np.float32)

        # Clean sequences
        valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
        cleaned = []
        for seq in sequences:
            clean_seq = ''.join(c if c in valid_aa else '' for c in str(seq).upper())
            if len(clean_seq) >= 5:  # Minimum viable CDR3 length
                cleaned.append(clean_seq[:Config.MAX_SEQ_LEN])

        if len(cleaned) == 0:
            return np.zeros((1, Config.ESM_DIM), dtype=np.float32)

        embeddings = []
        for i in range(0, len(cleaned), batch_size):
            batch_seqs = cleaned[i:i+batch_size]
            batch_data = [(f"seq_{j}", s) for j, s in enumerate(batch_seqs)]

            _, _, batch_tokens = self.batch_converter(batch_data)
            batch_tokens = batch_tokens.to(self.device)

            # Extract LAYER 6 (not final layer!)
            results = self.model(batch_tokens, repr_layers=[Config.ESM_LAYER], return_contacts=False)
            reps = results["representations"][Config.ESM_LAYER]

            # Mean pooling over sequence
            for j, seq_len in enumerate([len(s) for s in batch_seqs]):
                seq_emb = reps[j, 1:seq_len+1].mean(0)
                embeddings.append(seq_emb.cpu().numpy())

            del batch_tokens, results, reps
            torch.cuda.empty_cache()

        return np.array(embeddings, dtype=np.float32)

# ============================================================================
# Gated Attention Pooling (DeepRC-style)
# ============================================================================
class GatedAttentionPool(nn.Module):
    """
    Gated Attention for MIL (Multiple Instance Learning).

    Key insight: Not all sequences in a repertoire are disease-associated.
    This module learns to attend to relevant sequences.

    Reference: DeepRC (Nature Communications 2021)
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()

        # Gating mechanism: sigmoid(U*x) * tanh(V*x)
        self.attention_U = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh()
        )
        self.attention_V = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid()
        )

        # Attention score
        self.attention_w = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        Args:
            x: (batch, n_sequences, dim)
            mask: (batch, n_sequences) - True for valid positions

        Returns:
            aggregated: (batch, dim)
            attention_weights: (batch, n_sequences)
        """
        # Gated attention
        h_U = self.attention_U(x)  # (batch, n_seq, hidden)
        h_V = self.attention_V(x)  # (batch, n_seq, hidden)

        gated = h_U * h_V  # Element-wise gating

        # Attention scores
        attention_scores = self.attention_w(gated).squeeze(-1)  # (batch, n_seq)

        # Mask invalid positions
        if mask is not None:
            attention_scores = attention_scores.masked_fill(~mask, float('-inf'))

        # Softmax over sequences
        attention_weights = F.softmax(attention_scores, dim=1)  # (batch, n_seq)

        # Weighted sum
        aggregated = torch.bmm(attention_weights.unsqueeze(1), x).squeeze(1)  # (batch, dim)

        return aggregated, attention_weights

# ============================================================================
# Breakthrough Classifier Model
# ============================================================================
class BreakthroughClassifier(nn.Module):
    """
    Pure ESM-2 + Gated Attention classifier.

    NO dataset-specific features (V/J gene usage removed).
    This forces the model to learn biological signal, not technical bias.
    """

    def __init__(self, esm_dim: int = 1280, hidden_dim: int = 256,
                 num_heads: int = 4, dropout: float = 0.3):
        super().__init__()

        # Project ESM embeddings
        self.input_proj = nn.Sequential(
            nn.Linear(esm_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Gated attention pooling
        self.attention = GatedAttentionPool(hidden_dim, hidden_dim // 2)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        """
        Args:
            x: (batch, n_sequences, esm_dim)
            mask: (batch, n_sequences)

        Returns:
            logits: (batch, 1)
            attention_weights: (batch, n_sequences)
        """
        # Project to hidden dimension
        h = self.input_proj(x)  # (batch, n_seq, hidden)

        # Gated attention aggregation
        aggregated, attn_weights = self.attention(h, mask)  # (batch, hidden)

        # Classify
        logits = self.classifier(aggregated)  # (batch, 1)

        return logits, attn_weights

# ============================================================================
# Dataset for LODO Training
# ============================================================================
class RepertoireDataset(Dataset):
    """Dataset for repertoire-level classification."""

    def __init__(self, data: List[Dict]):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate function for variable-length repertoires."""
    max_len = max(item['embeddings'].shape[0] for item in batch)
    batch_size = len(batch)
    esm_dim = batch[0]['embeddings'].shape[1]

    # Pad embeddings
    padded = torch.zeros(batch_size, max_len, esm_dim)
    masks = torch.zeros(batch_size, max_len, dtype=torch.bool)
    labels = torch.zeros(batch_size)

    for i, item in enumerate(batch):
        seq_len = item['embeddings'].shape[0]
        padded[i, :seq_len] = torch.from_numpy(item['embeddings'])
        masks[i, :seq_len] = True
        labels[i] = item['label']

    return {
        'embeddings': padded,
        'masks': masks,
        'labels': labels,
        'repertoire_ids': [item['repertoire_id'] for item in batch],
        'dataset_ids': [item['dataset_id'] for item in batch]
    }

# ============================================================================
# Data Loading
# ============================================================================
def load_repertoire_sequences(tsv_path: Path, max_seqs: int = 500) -> List[str]:
    """Load CDR3 sequences from repertoire file."""
    try:
        df = pd.read_csv(tsv_path, sep='\t', usecols=['junction_aa'], nrows=max_seqs * 2)
        sequences = df['junction_aa'].dropna().astype(str).tolist()

        # Sample if too many
        if len(sequences) > max_seqs:
            np.random.seed(Config.SEED)
            sequences = list(np.random.choice(sequences, max_seqs, replace=False))

        return sequences
    except Exception as e:
        print(f"❌ Error loading {tsv_path}: {e}")
        return []

def load_dataset(dataset_path: Path, dataset_id: int, esm_extractor: ESM2Extractor) -> List[Dict]:
    """Load one dataset with ESM-2 embeddings."""
    print(f"\n📂 Loading dataset {dataset_id}...")

    metadata = pd.read_csv(dataset_path / 'metadata.csv')
    data = []

    for idx, row in tqdm(metadata.iterrows(), total=len(metadata), desc=f"Dataset {dataset_id}"):
        tsv_path = dataset_path / row['filename']

        if not tsv_path.exists():
            continue

        # Load sequences
        sequences = load_repertoire_sequences(tsv_path, Config.MAX_SEQS_PER_REP)

        if len(sequences) < 10:  # Skip very small repertoires
            continue

        # Extract ESM-2 embeddings
        embeddings = esm_extractor.extract_embeddings(sequences, Config.ESM_BATCH_SIZE)

        data.append({
            'embeddings': embeddings,
            'label': 1 if row['label_positive'] else 0,
            'repertoire_id': row['repertoire_id'],
            'dataset_id': dataset_id
        })

        # Memory cleanup
        if idx % 50 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    print(f"✓ Loaded {len(data)} repertoires from dataset {dataset_id}")
    return data

# ============================================================================
# Training Functions
# ============================================================================
def train_epoch(model, loader, optimizer, criterion, device, scaler=None):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []

    for batch in tqdm(loader, desc="Training", leave=False):
        embeddings = batch['embeddings'].to(device)
        masks = batch['masks'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        if scaler is not None:
            with autocast():
                logits, _ = model(embeddings, masks)
                loss = criterion(logits.squeeze(), labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits, _ = model(embeddings, masks)
            loss = criterion(logits.squeeze(), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item()
        probs = torch.sigmoid(logits.squeeze()).detach().cpu().numpy()
        all_preds.extend(probs)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    auc = roc_auc_score(all_labels, all_preds) if len(set(all_labels)) > 1 else 0.5
    return avg_loss, auc

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    all_repertoire_ids = []

    for batch in tqdm(loader, desc="Evaluating", leave=False):
        embeddings = batch['embeddings'].to(device)
        masks = batch['masks'].to(device)
        labels = batch['labels'].to(device)

        logits, _ = model(embeddings, masks)
        loss = criterion(logits.squeeze(), labels)

        total_loss += loss.item()
        probs = torch.sigmoid(logits.squeeze()).cpu().numpy()
        all_preds.extend(probs)
        all_labels.extend(labels.cpu().numpy())
        all_repertoire_ids.extend(batch['repertoire_ids'])

    avg_loss = total_loss / len(loader)
    auc = roc_auc_score(all_labels, all_preds) if len(set(all_labels)) > 1 else 0.5
    return avg_loss, auc, all_preds, all_repertoire_ids

# ============================================================================
# LODO Cross-Validation (Leave-One-Dataset-Out)
# ============================================================================
def train_lodo_fold(fold_id: int, train_data: List[Dict], val_data: List[Dict], device: str):
    """Train one LODO fold."""
    print(f"\n{'='*60}")
    print(f"🎯 LODO FOLD {fold_id}: Train on {len(train_data)}, Val on {len(val_data)}")
    print(f"{'='*60}")

    # Create data loaders
    train_loader = DataLoader(
        RepertoireDataset(train_data),
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True
    )
    val_loader = DataLoader(
        RepertoireDataset(val_data),
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True
    )

    # Initialize model
    model = BreakthroughClassifier(
        esm_dim=Config.ESM_DIM,
        hidden_dim=Config.HIDDEN_DIM,
        num_heads=Config.ATTENTION_HEADS,
        dropout=Config.DROPOUT
    ).to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY
    )

    # Loss with class weighting
    criterion = nn.BCEWithLogitsLoss()

    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=1e-6
    )

    # Mixed precision
    scaler = GradScaler() if Config.USE_AMP else None

    # Training loop
    best_auc = 0
    patience_counter = 0

    for epoch in range(Config.EPOCHS):
        # Train
        train_loss, train_auc = train_epoch(model, train_loader, optimizer, criterion, device, scaler)

        # Validate
        val_loss, val_auc, _, _ = evaluate(model, val_loader, criterion, device)

        scheduler.step()

        print(f"Epoch {epoch+1:2d} | Train AUC: {train_auc:.4f} | Val AUC: {val_auc:.4f}")

        # Save best model
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            Config.MODEL_DIR.mkdir(exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'fold_id': fold_id,
                'val_auc': val_auc,
                'epoch': epoch
            }, Config.MODEL_DIR / f'fold_{fold_id}_best.pt')
            print(f"  ✓ New best! AUC: {best_auc:.4f}")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= Config.PATIENCE:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    # Load best model
    checkpoint = torch.load(Config.MODEL_DIR / f'fold_{fold_id}_best.pt')
    model.load_state_dict(checkpoint['model_state_dict'])

    return model, best_auc

# ============================================================================
# Prediction and Submission
# ============================================================================
@torch.no_grad()
def predict_test(models: List[nn.Module], esm_extractor: ESM2Extractor, device: str) -> pd.DataFrame:
    """Generate predictions for test datasets."""
    print("\n📊 Generating test predictions...")

    predictions = []
    test_dir = Config.TEST_ROOT

    for test_dataset in sorted(test_dir.iterdir()):
        if not test_dataset.name.startswith('test_dataset_'):
            continue

        dataset_id = int(test_dataset.name.split('_')[-1])
        print(f"\n  Processing {test_dataset.name}...")

        metadata = pd.read_csv(test_dataset / 'metadata.csv')

        for idx, row in tqdm(metadata.iterrows(), total=len(metadata)):
            tsv_path = test_dataset / row['filename']

            # Load sequences
            sequences = load_repertoire_sequences(tsv_path, Config.MAX_SEQS_PER_REP)

            if len(sequences) < 5:
                prob = 0.5  # Default for empty
            else:
                # Extract embeddings
                embeddings = esm_extractor.extract_embeddings(sequences, Config.ESM_BATCH_SIZE)

                # Ensemble prediction
                probs = []
                for model in models:
                    model.eval()
                    x = torch.from_numpy(embeddings).unsqueeze(0).to(device)
                    mask = torch.ones(1, x.size(1), dtype=torch.bool, device=device)
                    logits, _ = model(x, mask)
                    probs.append(torch.sigmoid(logits).item())

                prob = np.mean(probs)

            predictions.append({
                'ID': row['repertoire_id'],
                'dataset': test_dataset.name,
                'label_positive_probability': prob
            })

    return pd.DataFrame(predictions)

def generate_task_b(all_data: List[Dict], models: List[nn.Module], device: str) -> pd.DataFrame:
    """Generate Task B predictions (important sequences)."""
    print("\n🧬 Generating Task B predictions...")

    task_b_rows = []

    for dataset_id in range(1, 9):
        dataset_data = [d for d in all_data if d['dataset_id'] == dataset_id]

        if len(dataset_data) == 0:
            continue

        # Collect sequences with attention weights from positive samples
        seq_scores = {}

        for rep in dataset_data:
            if rep['label'] != 1:  # Only from positive samples
                continue

            # Get attention weights from ensemble
            embeddings = torch.from_numpy(rep['embeddings']).unsqueeze(0).to(device)
            mask = torch.ones(1, embeddings.size(1), dtype=torch.bool, device=device)

            attn_weights_list = []
            for model in models:
                model.eval()
                with torch.no_grad():
                    _, attn = model(embeddings, mask)
                    attn_weights_list.append(attn.squeeze().cpu().numpy())

            avg_attn = np.mean(attn_weights_list, axis=0)

            # Note: We'd need to map attention weights back to sequences
            # For now, use a simpler approach based on frequency

        # Load sequences directly from files for Task B
        train_path = Config.TRAIN_ROOT / f'train_dataset_{dataset_id}'
        metadata = pd.read_csv(train_path / 'metadata.csv')

        all_sequences = []
        for _, row in metadata.iterrows():
            tsv_path = train_path / row['filename']
            if tsv_path.exists():
                df = pd.read_csv(tsv_path, sep='\t', usecols=['junction_aa', 'v_call', 'j_call'], nrows=10000)
                df['label'] = row['label_positive']
                all_sequences.append(df)

        if not all_sequences:
            continue

        combined = pd.concat(all_sequences, ignore_index=True)

        # Score by frequency difference
        pos_seqs = combined[combined['label'] == True]['junction_aa'].value_counts()
        neg_seqs = combined[combined['label'] == False]['junction_aa'].value_counts()

        all_seqs = set(pos_seqs.index) | set(neg_seqs.index)
        scores = {}
        for seq in all_seqs:
            pos_freq = pos_seqs.get(seq, 0) / max(pos_seqs.sum(), 1)
            neg_freq = neg_seqs.get(seq, 0) / max(neg_seqs.sum(), 1)
            scores[seq] = pos_freq - neg_freq

        # Sort by score
        top_seqs = sorted(scores.items(), key=lambda x: -x[1])[:50000]

        # Get V/J calls
        seq_info = combined.drop_duplicates('junction_aa').set_index('junction_aa')

        for rank, (seq, score) in enumerate(top_seqs, 1):
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
# Main Training Pipeline
# ============================================================================
def main():
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║  🏆 AIRR-ML-25 BREAKTHROUGH Model Training                  ║
    ║                                                              ║
    ║  Target: Beat SajayR (0.84590) → Achieve 0.85+              ║
    ║  Method: ESM-2 Layer 6 + Gated Attention MIL + LODO CV      ║
    ╚══════════════════════════════════════════════════════════════╝
    """)

    device = Config.DEVICE

    # Initialize ESM-2
    esm_extractor = ESM2Extractor(device=device)

    # Load all training data
    print("\n📥 Loading all training data with ESM-2 embeddings...")
    all_data = []

    for dataset_id in range(1, 9):
        dataset_path = Config.TRAIN_ROOT / f'train_dataset_{dataset_id}'
        if dataset_path.exists():
            data = load_dataset(dataset_path, dataset_id, esm_extractor)
            all_data.extend(data)
            gc.collect()
            torch.cuda.empty_cache()

    print(f"\n✓ Total loaded: {len(all_data)} repertoires")

    # LODO Cross-Validation
    print("\n" + "="*60)
    print("🎓 LEAVE-ONE-DATASET-OUT CROSS-VALIDATION")
    print("="*60)

    fold_results = []
    trained_models = []

    for val_dataset_id in range(1, 9):
        train_data = [d for d in all_data if d['dataset_id'] != val_dataset_id]
        val_data = [d for d in all_data if d['dataset_id'] == val_dataset_id]

        if len(val_data) == 0:
            continue

        model, val_auc = train_lodo_fold(val_dataset_id, train_data, val_data, device)
        fold_results.append({'fold': val_dataset_id, 'val_auc': val_auc})
        trained_models.append(model)

        gc.collect()
        torch.cuda.empty_cache()

    # Print results
    print("\n" + "="*60)
    print("📈 LODO CROSS-VALIDATION RESULTS")
    print("="*60)
    for r in fold_results:
        print(f"  Fold {r['fold']}: AUC = {r['val_auc']:.4f}")

    mean_auc = np.mean([r['val_auc'] for r in fold_results])
    std_auc = np.std([r['val_auc'] for r in fold_results])
    print(f"\n  🎯 Mean LODO AUC: {mean_auc:.4f} ± {std_auc:.4f}")

    # Generate predictions
    print("\n" + "="*60)
    print("📊 GENERATING PREDICTIONS")
    print("="*60)

    # Task A predictions
    task_a = predict_test(trained_models, esm_extractor, device)

    # Task B predictions
    task_b = generate_task_b(all_data, trained_models, device)

    # Combine and save
    submission = pd.concat([task_a, task_b], ignore_index=True)

    # Ensure correct column order
    submission = submission[['ID', 'dataset', 'label_positive_probability', 'junction_aa', 'v_call', 'j_call']]

    # Fill NaN for Task A rows
    submission['junction_aa'] = submission['junction_aa'].fillna('-999.0')
    submission['v_call'] = submission['v_call'].fillna('-999.0')
    submission['j_call'] = submission['j_call'].fillna('-999.0')

    # Save
    Config.OUTPUT_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = Config.OUTPUT_DIR / f'breakthrough_submission_{timestamp}.csv'
    submission.to_csv(output_path, index=False)

    print(f"\n✅ Submission saved: {output_path}")
    print(f"   Total rows: {len(submission)}")
    print(f"   Task A rows: {len(task_a)}")
    print(f"   Task B rows: {len(task_b)}")

    print("\n🏆 Training complete! Good luck!")

if __name__ == '__main__':
    main()
