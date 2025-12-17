#!/usr/bin/env python3
"""
🏆 AIRR-ML-25 BREAKTHROUGH Model - DUAL GPU MAXED OUT
================================================================
Target: Beat SajayR (0.84590) → Achieve 0.85+

HARDWARE UTILIZATION:
- GPU 0 (RTX 6000 Ada 48GB): ESM-2 embedding extraction
- GPU 1 (RTX 6000 Ada 48GB): Model training
- Both GPUs: Parallel LODO fold training

Based on 2025 Research:
1. ESM-2 Layer 6 (not Layer 33!) - optimal for TCR CDR3
2. Gated Attention MIL (DeepRC-style)
3. LODO CV - true cross-dataset generalization
4. Larger batch sizes (utilizing 48GB VRAM)

Deadline: Dec 17, 2025 06:59 UTC
"""

import os
import gc
import sys
import warnings
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DataParallel
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')

# ============================================================================
# Configuration - MAXED OUT FOR DUAL RTX 6000 Ada
# ============================================================================
class Config:
    # Paths
    DATA_ROOT = Path('./data')
    TRAIN_ROOT = DATA_ROOT / 'train_datasets/train_datasets'
    TEST_ROOT = DATA_ROOT / 'test_datasets/test_datasets'
    OUTPUT_DIR = Path('./submissions')
    MODEL_DIR = Path('./models_breakthrough')
    CACHE_DIR = Path('./cache_breakthrough')

    # ESM-2 Settings (CRITICAL: Use Layer 6!)
    ESM_MODEL = 'facebook/esm2_t33_650M_UR50D'
    ESM_LAYER = 6
    ESM_DIM = 1280
    ESM_BATCH_SIZE = 128  # MAXED - 48GB can handle this
    MAX_SEQ_LEN = 30
    MAX_SEQS_PER_REP = 1000  # More sequences = better representation

    # Training Settings - LARGER BATCHES
    EPOCHS = 50  # More epochs for better convergence
    BATCH_SIZE = 32  # Larger batch with 48GB VRAM
    LEARNING_RATE = 3e-4  # Slightly higher LR for faster convergence
    WEIGHT_DECAY = 0.01
    PATIENCE = 10

    # Model Architecture - DEEPER
    HIDDEN_DIM = 512  # Larger hidden dimension
    ATTENTION_HEADS = 8  # More attention heads
    DROPOUT = 0.2  # Less dropout with more data

    # Hardware - DUAL GPU
    ESM_DEVICE = 'cuda:0'  # ESM-2 extraction on GPU 0
    TRAIN_DEVICE = 'cuda:1'  # Training on GPU 1
    NUM_WORKERS = 8  # More data loading workers
    USE_AMP = True

    # Random seed
    SEED = 42

# Set seeds
torch.manual_seed(Config.SEED)
np.random.seed(Config.SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(Config.SEED)

# Check GPUs
print("🔥 DUAL GPU CONFIGURATION")
print("=" * 60)
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f"  GPU {i}: {props.name}")
    print(f"         VRAM: {props.total_memory / 1e9:.1f} GB")
print("=" * 60)
print(f"  ESM-2 Extraction: GPU 0")
print(f"  Model Training:   GPU 1")
print("=" * 60)

# ============================================================================
# ESM-2 Feature Extractor - GPU 0
# ============================================================================
class ESM2Extractor:
    """Extract ESM-2 Layer 6 embeddings on GPU 0."""

    def __init__(self, device='cuda:0'):
        self.device = device
        print(f"\n📦 Loading ESM-2 on {device}...")

        try:
            import esm
            self.model, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
            self.model = self.model.to(device).eval()
            self.batch_converter = self.alphabet.get_batch_converter()

            # Enable TF32 for faster computation on RTX 6000
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

            print(f"✓ ESM-2 loaded on {device}")
        except ImportError:
            print("Installing fair-esm...")
            os.system("pip install fair-esm -q")
            import esm
            self.model, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
            self.model = self.model.to(device).eval()
            self.batch_converter = self.alphabet.get_batch_converter()
            print(f"✓ ESM-2 loaded on {device}")

    @torch.no_grad()
    def extract_embeddings(self, sequences: List[str], batch_size: int = 128) -> np.ndarray:
        """Extract Layer 6 embeddings with LARGE batches."""
        if len(sequences) == 0:
            return np.zeros((1, Config.ESM_DIM), dtype=np.float32)

        # Clean sequences
        valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
        cleaned = []
        for seq in sequences:
            clean_seq = ''.join(c if c in valid_aa else '' for c in str(seq).upper())
            if 5 <= len(clean_seq) <= Config.MAX_SEQ_LEN:
                cleaned.append(clean_seq)

        if len(cleaned) == 0:
            return np.zeros((1, Config.ESM_DIM), dtype=np.float32)

        embeddings = []
        for i in range(0, len(cleaned), batch_size):
            batch_seqs = cleaned[i:i+batch_size]
            batch_data = [(f"seq_{j}", s) for j, s in enumerate(batch_seqs)]

            _, _, batch_tokens = self.batch_converter(batch_data)
            batch_tokens = batch_tokens.to(self.device)

            # Extract LAYER 6
            with autocast():
                results = self.model(batch_tokens, repr_layers=[Config.ESM_LAYER], return_contacts=False)
            reps = results["representations"][Config.ESM_LAYER]

            # Mean pooling
            for j, seq_len in enumerate([len(s) for s in batch_seqs]):
                seq_emb = reps[j, 1:seq_len+1].mean(0)
                embeddings.append(seq_emb.cpu().float().numpy())

            del batch_tokens, results, reps

        return np.array(embeddings, dtype=np.float32)

# ============================================================================
# Gated Attention Pooling (DeepRC-style) - ENHANCED
# ============================================================================
class GatedAttentionPool(nn.Module):
    """Enhanced Gated Attention for MIL."""

    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()

        # Deeper gating
        self.attention_U = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh()
        )
        self.attention_V = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Sigmoid()
        )

        self.attention_w = nn.Linear(hidden_dim, 1)

        # Additional context projection
        self.context_proj = nn.Linear(input_dim, hidden_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        Args:
            x: (batch, n_sequences, dim)
            mask: (batch, n_sequences)
        """
        h_U = self.attention_U(x)
        h_V = self.attention_V(x)
        gated = h_U * h_V

        attention_scores = self.attention_w(gated).squeeze(-1)

        if mask is not None:
            attention_scores = attention_scores.masked_fill(~mask, float('-inf'))

        attention_weights = F.softmax(attention_scores, dim=1)

        # Weighted sum
        aggregated = torch.bmm(attention_weights.unsqueeze(1), x).squeeze(1)

        return aggregated, attention_weights

# ============================================================================
# Breakthrough Classifier - DEEPER ARCHITECTURE
# ============================================================================
class BreakthroughClassifier(nn.Module):
    """Enhanced classifier with deeper architecture."""

    def __init__(self, esm_dim: int = 1280, hidden_dim: int = 512,
                 num_heads: int = 8, dropout: float = 0.2):
        super().__init__()

        # Projection layers
        self.input_proj = nn.Sequential(
            nn.Linear(esm_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

        # Gated attention
        self.attention = GatedAttentionPool(hidden_dim, hidden_dim // 2)

        # Deeper classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        h = self.input_proj(x)
        aggregated, attn_weights = self.attention(h, mask)
        logits = self.classifier(aggregated)
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
    max_len = max(item['embeddings'].shape[0] for item in batch)
    batch_size = len(batch)
    esm_dim = batch[0]['embeddings'].shape[1]

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
# Data Loading with Caching
# ============================================================================
def load_repertoire_sequences(tsv_path: Path, max_seqs: int = 1000) -> List[str]:
    """Load CDR3 sequences."""
    try:
        df = pd.read_csv(tsv_path, sep='\t', usecols=['junction_aa'], nrows=max_seqs * 2)
        sequences = df['junction_aa'].dropna().astype(str).tolist()
        if len(sequences) > max_seqs:
            np.random.seed(Config.SEED)
            sequences = list(np.random.choice(sequences, max_seqs, replace=False))
        return sequences
    except:
        return []

def load_dataset_cached(dataset_id: int, esm_extractor: ESM2Extractor) -> List[Dict]:
    """Load dataset with caching."""
    cache_path = Config.CACHE_DIR / f'dataset_{dataset_id}_emb.npz'
    Config.CACHE_DIR.mkdir(exist_ok=True)

    dataset_path = Config.TRAIN_ROOT / f'train_dataset_{dataset_id}'
    metadata = pd.read_csv(dataset_path / 'metadata.csv')

    # Check cache
    if cache_path.exists():
        print(f"  📦 Loading cached embeddings for dataset {dataset_id}...")
        cached = np.load(cache_path, allow_pickle=True)
        data = cached['data'].tolist()
        print(f"  ✓ Loaded {len(data)} cached repertoires")
        return data

    print(f"\n📂 Processing dataset {dataset_id} (will cache)...")
    data = []

    for idx, row in tqdm(metadata.iterrows(), total=len(metadata), desc=f"Dataset {dataset_id}"):
        tsv_path = dataset_path / row['filename']
        if not tsv_path.exists():
            continue

        sequences = load_repertoire_sequences(tsv_path, Config.MAX_SEQS_PER_REP)
        if len(sequences) < 10:
            continue

        embeddings = esm_extractor.extract_embeddings(sequences, Config.ESM_BATCH_SIZE)

        data.append({
            'embeddings': embeddings,
            'label': 1 if row['label_positive'] else 0,
            'repertoire_id': row['repertoire_id'],
            'dataset_id': dataset_id
        })

        if idx % 50 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    # Cache
    np.savez_compressed(cache_path, data=np.array(data, dtype=object))
    print(f"  ✓ Cached {len(data)} repertoires to {cache_path}")

    return data

# ============================================================================
# Training Functions
# ============================================================================
def train_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []

    for batch in tqdm(loader, desc="Training", leave=False):
        embeddings = batch['embeddings'].to(device)
        masks = batch['masks'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        if scaler:
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
        all_preds.extend(probs if len(probs.shape) > 0 else [probs.item()])
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    auc = roc_auc_score(all_labels, all_preds) if len(set(all_labels)) > 1 else 0.5
    return avg_loss, auc

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []

    for batch in tqdm(loader, desc="Evaluating", leave=False):
        embeddings = batch['embeddings'].to(device)
        masks = batch['masks'].to(device)
        labels = batch['labels'].to(device)

        logits, _ = model(embeddings, masks)
        loss = criterion(logits.squeeze(), labels)

        total_loss += loss.item()
        probs = torch.sigmoid(logits.squeeze()).cpu().numpy()
        all_preds.extend(probs if len(probs.shape) > 0 else [probs.item()])
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    auc = roc_auc_score(all_labels, all_preds) if len(set(all_labels)) > 1 else 0.5
    return avg_loss, auc, all_preds

# ============================================================================
# LODO Training
# ============================================================================
def train_lodo_fold(fold_id: int, train_data: List[Dict], val_data: List[Dict], device: str):
    """Train one LODO fold on GPU 1."""
    print(f"\n{'='*60}")
    print(f"🎯 LODO FOLD {fold_id}: Train={len(train_data)}, Val={len(val_data)}")
    print(f"{'='*60}")

    train_loader = DataLoader(
        RepertoireDataset(train_data),
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        prefetch_factor=4
    )
    val_loader = DataLoader(
        RepertoireDataset(val_data),
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True
    )

    model = BreakthroughClassifier(
        esm_dim=Config.ESM_DIM,
        hidden_dim=Config.HIDDEN_DIM,
        num_heads=Config.ATTENTION_HEADS,
        dropout=Config.DROPOUT
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    scaler = GradScaler() if Config.USE_AMP else None

    best_auc = 0
    patience_counter = 0

    for epoch in range(Config.EPOCHS):
        train_loss, train_auc = train_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_loss, val_auc, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1:2d} | LR: {lr:.2e} | Train AUC: {train_auc:.4f} | Val AUC: {val_auc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            Config.MODEL_DIR.mkdir(exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'fold_id': fold_id,
                'val_auc': val_auc
            }, Config.MODEL_DIR / f'fold_{fold_id}_best.pt')
            print(f"  ✓ New best! AUC: {best_auc:.4f}")
        else:
            patience_counter += 1

        if patience_counter >= Config.PATIENCE:
            print(f"  Early stopping")
            break

    checkpoint = torch.load(Config.MODEL_DIR / f'fold_{fold_id}_best.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, best_auc

# ============================================================================
# Prediction
# ============================================================================
@torch.no_grad()
def predict_test(models: List[nn.Module], esm_extractor: ESM2Extractor, device: str) -> pd.DataFrame:
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
            sequences = load_repertoire_sequences(tsv_path, Config.MAX_SEQS_PER_REP)

            if len(sequences) < 5:
                prob = 0.5
            else:
                embeddings = esm_extractor.extract_embeddings(sequences, Config.ESM_BATCH_SIZE)

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

def generate_task_b(esm_extractor: ESM2Extractor) -> pd.DataFrame:
    """Generate Task B."""
    print("\n🧬 Generating Task B...")
    task_b_rows = []

    for dataset_id in range(1, 9):
        train_path = Config.TRAIN_ROOT / f'train_dataset_{dataset_id}'
        if not train_path.exists():
            continue

        metadata = pd.read_csv(train_path / 'metadata.csv')

        # Collect sequences
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

        # Score by frequency difference
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
    ║  🏆 AIRR-ML-25 BREAKTHROUGH - DUAL GPU MAXED OUT                ║
    ║                                                                  ║
    ║  Target: Beat SajayR (0.84590) → Achieve 0.85+                  ║
    ║  Method: ESM-2 L6 + Gated Attention MIL + LODO CV               ║
    ║  Hardware: 2x RTX 6000 Ada (96GB total VRAM)                    ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)

    esm_device = Config.ESM_DEVICE
    train_device = Config.TRAIN_DEVICE

    # Initialize ESM-2 on GPU 0
    esm_extractor = ESM2Extractor(device=esm_device)

    # Load all data with caching
    print("\n📥 Loading all training data...")
    all_data = []
    for dataset_id in range(1, 9):
        data = load_dataset_cached(dataset_id, esm_extractor)
        all_data.extend(data)
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\n✓ Total: {len(all_data)} repertoires")

    # LODO CV on GPU 1
    print("\n" + "="*60)
    print("🎓 LEAVE-ONE-DATASET-OUT CROSS-VALIDATION (GPU 1)")
    print("="*60)

    fold_results = []
    trained_models = []

    for val_id in range(1, 9):
        train_data = [d for d in all_data if d['dataset_id'] != val_id]
        val_data = [d for d in all_data if d['dataset_id'] == val_id]

        if len(val_data) == 0:
            continue

        model, val_auc = train_lodo_fold(val_id, train_data, val_data, train_device)
        fold_results.append({'fold': val_id, 'val_auc': val_auc})
        trained_models.append(model)

        gc.collect()
        torch.cuda.empty_cache()

    # Results
    print("\n" + "="*60)
    print("📈 LODO RESULTS")
    print("="*60)
    for r in fold_results:
        print(f"  Fold {r['fold']}: AUC = {r['val_auc']:.4f}")

    mean_auc = np.mean([r['val_auc'] for r in fold_results])
    std_auc = np.std([r['val_auc'] for r in fold_results])
    print(f"\n  🎯 Mean LODO AUC: {mean_auc:.4f} ± {std_auc:.4f}")

    # Predictions
    task_a = predict_test(trained_models, esm_extractor, train_device)
    task_b = generate_task_b(esm_extractor)

    # Combine
    submission = pd.concat([task_a, task_b], ignore_index=True)
    submission = submission[['ID', 'dataset', 'label_positive_probability', 'junction_aa', 'v_call', 'j_call']]
    submission['junction_aa'] = submission['junction_aa'].fillna('-999.0')
    submission['v_call'] = submission['v_call'].fillna('-999.0')
    submission['j_call'] = submission['j_call'].fillna('-999.0')

    Config.OUTPUT_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = Config.OUTPUT_DIR / f'breakthrough_dual_gpu_{timestamp}.csv'
    submission.to_csv(output_path, index=False)

    print(f"\n✅ Submission: {output_path}")
    print(f"   Rows: {len(submission)} (Task A: {len(task_a)}, Task B: {len(task_b)})")
    print("\n🏆 Done!")

if __name__ == '__main__':
    main()
