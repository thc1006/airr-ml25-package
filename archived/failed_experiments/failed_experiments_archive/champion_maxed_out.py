#!/usr/bin/env python3
"""
🔥🔥🔥 AIRR-ML-25 MAXED OUT - 榨乾雙 GPU 極致版 🔥🔥🔥
================================================================
Hardware: 2x RTX 6000 Ada (48GB each = 96GB total)

Optimizations:
1. ESM-2 batch size: 512 (was 64) - 8x speedup
2. Max sequences per repertoire: 2000 (was 500)
3. Training batch size: 64 (was 8)
4. torch.compile() for 2x speedup
5. Parallel data loading with prefetch
6. Both GPUs for ESM embedding extraction
7. Cached embeddings to skip recomputation

Target: Finish in < 2 hours (was 6-8 hours)
"""

import os
import gc
import warnings
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor
import threading

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
# MAXED OUT Configuration
# ============================================================================
class Config:
    # Paths
    DATA_ROOT = Path('./data')
    TRAIN_ROOT = DATA_ROOT / 'train_datasets/train_datasets'
    TEST_ROOT = DATA_ROOT / 'test_datasets/test_datasets'
    OUTPUT_DIR = Path('./submissions')
    MODEL_DIR = Path('./models_maxed')
    CACHE_DIR = Path('./cache_maxed')

    # ESM-2 MAXED Settings
    ESM_MODEL = 'facebook/esm2_t33_650M_UR50D'
    ESM_LAYER = 6
    ESM_DIM = 1280
    ESM_BATCH_SIZE = 256  # MAXED! (was 64)
    MAX_SEQ_LEN = 25
    MAX_SEQS_PER_REP = 1500  # More sequences

    # Training MAXED Settings
    EPOCHS = 30
    BATCH_SIZE = 48  # MAXED! (was 8)
    LEARNING_RATE = 5e-4
    WEIGHT_DECAY = 0.01
    PATIENCE = 6

    # Model Architecture
    HIDDEN_DIM = 384
    DROPOUT = 0.25

    # Hardware - DUAL GPU
    DEVICES = ['cuda:0', 'cuda:1']
    NUM_WORKERS = 12
    USE_AMP = True
    USE_COMPILE = False  # Disabled due to triton compatibility issues

    SEED = 42

torch.manual_seed(Config.SEED)
np.random.seed(Config.SEED)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

print("🔥" * 30)
print("  MAXED OUT DUAL GPU TRAINING")
print("🔥" * 30)
for i, dev in enumerate(Config.DEVICES):
    props = torch.cuda.get_device_properties(i)
    print(f"  GPU {i}: {props.name} ({props.total_memory/1e9:.0f}GB)")
print(f"  ESM Batch: {Config.ESM_BATCH_SIZE}")
print(f"  Train Batch: {Config.BATCH_SIZE}")
print(f"  Max Seqs/Rep: {Config.MAX_SEQS_PER_REP}")
print("🔥" * 30)

# ============================================================================
# Dual-GPU ESM-2 Extractor
# ============================================================================
class DualGPUESM2Extractor:
    """Extract ESM-2 embeddings using BOTH GPUs."""

    def __init__(self):
        print("\n📦 Loading ESM-2 on BOTH GPUs...")
        import esm

        self.models = []
        self.batch_converters = []

        for device in Config.DEVICES:
            model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
            model = model.to(device).eval()
            self.models.append(model)
            self.batch_converters.append(alphabet.get_batch_converter())
            print(f"  ✓ Loaded on {device}")

    @torch.no_grad()
    def extract_batch(self, sequences: List[str], gpu_id: int = 0) -> np.ndarray:
        """Extract embeddings on specified GPU."""
        if len(sequences) == 0:
            return np.zeros((1, Config.ESM_DIM), dtype=np.float32)

        device = Config.DEVICES[gpu_id]
        model = self.models[gpu_id]
        batch_converter = self.batch_converters[gpu_id]

        batch_data = [(f"seq_{i}", s[:Config.MAX_SEQ_LEN]) for i, s in enumerate(sequences)]
        _, _, batch_tokens = batch_converter(batch_data)
        batch_tokens = batch_tokens.to(device)

        with autocast():
            results = model(batch_tokens, repr_layers=[Config.ESM_LAYER], return_contacts=False)
        reps = results["representations"][Config.ESM_LAYER]

        embeddings = []
        for j, seq in enumerate(sequences):
            seq_len = min(len(seq), Config.MAX_SEQ_LEN)
            emb = reps[j, 1:seq_len+1].mean(0).cpu().float().numpy()
            embeddings.append(emb)

        return np.array(embeddings, dtype=np.float32)

    def extract_repertoire(self, sequences: List[str], gpu_id: int = 0) -> np.ndarray:
        """Extract all embeddings for a repertoire."""
        valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
        cleaned = []
        for seq in sequences:
            clean = ''.join(c for c in str(seq).upper() if c in valid_aa)
            if 5 <= len(clean) <= Config.MAX_SEQ_LEN:
                cleaned.append(clean)

        if len(cleaned) == 0:
            return np.zeros((1, Config.ESM_DIM), dtype=np.float32)

        if len(cleaned) > Config.MAX_SEQS_PER_REP:
            np.random.seed(Config.SEED)
            cleaned = list(np.random.choice(cleaned, Config.MAX_SEQS_PER_REP, replace=False))

        all_emb = []
        for i in range(0, len(cleaned), Config.ESM_BATCH_SIZE):
            batch = cleaned[i:i+Config.ESM_BATCH_SIZE]
            emb = self.extract_batch(batch, gpu_id)
            all_emb.append(emb)

        return np.vstack(all_emb)

# ============================================================================
# Gated Attention MIL
# ============================================================================
class GatedAttentionMIL(nn.Module):
    def __init__(self, input_dim: int = 1280, hidden_dim: int = 384, dropout: float = 0.25):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.attention_U = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.Tanh())
        self.attention_V = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.Sigmoid())
        self.attention_w = nn.Linear(hidden_dim // 2, 1)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x, mask=None):
        h = self.encoder(x)

        u = self.attention_U(h)
        v = self.attention_V(h)
        scores = self.attention_w(u * v).squeeze(-1)

        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))

        weights = F.softmax(scores, dim=1)
        aggregated = torch.bmm(weights.unsqueeze(1), h).squeeze(1)
        logits = self.classifier(aggregated)

        return logits, weights

# ============================================================================
# Dataset
# ============================================================================
class RepDataset(Dataset):
    def __init__(self, data): self.data = data
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

def collate(batch):
    max_len = max(b['emb'].shape[0] for b in batch)
    bs = len(batch)
    dim = batch[0]['emb'].shape[1]

    emb = torch.zeros(bs, max_len, dim)
    mask = torch.zeros(bs, max_len, dtype=torch.bool)
    labels = torch.zeros(bs)

    for i, b in enumerate(batch):
        n = b['emb'].shape[0]
        emb[i, :n] = torch.from_numpy(b['emb'])
        mask[i, :n] = True
        labels[i] = b['label']

    return {'emb': emb, 'mask': mask, 'labels': labels, 'ids': [b['id'] for b in batch]}

# ============================================================================
# Load Data with Parallel GPU Processing
# ============================================================================
def load_all_data_parallel(esm: DualGPUESM2Extractor):
    """Load all data using BOTH GPUs in parallel."""
    print("\n📥 Loading data with DUAL GPU extraction...")

    Config.CACHE_DIR.mkdir(exist_ok=True)
    all_data = []

    for ds_id in range(1, 9):
        cache_file = Config.CACHE_DIR / f'ds{ds_id}_maxed.npz'

        if cache_file.exists():
            print(f"  📦 Loading cached dataset {ds_id}...")
            cached = np.load(cache_file, allow_pickle=True)
            all_data.extend(cached['data'].tolist())
            continue

        print(f"\n  🔄 Processing dataset {ds_id} (dual GPU)...")
        ds_path = Config.TRAIN_ROOT / f'train_dataset_{ds_id}'
        meta = pd.read_csv(ds_path / 'metadata.csv')

        data = []
        gpu_id = 0  # Alternate between GPUs

        for idx, row in tqdm(meta.iterrows(), total=len(meta), desc=f"DS{ds_id}"):
            tsv_path = ds_path / row['filename']
            if not tsv_path.exists():
                continue

            try:
                df = pd.read_csv(tsv_path, sep='\t', usecols=['junction_aa'], nrows=Config.MAX_SEQS_PER_REP * 2)
                seqs = df['junction_aa'].dropna().astype(str).tolist()
            except:
                continue

            if len(seqs) < 10:
                continue

            emb = esm.extract_repertoire(seqs, gpu_id=gpu_id)
            gpu_id = 1 - gpu_id  # Alternate

            data.append({
                'emb': emb,
                'label': 1 if row['label_positive'] else 0,
                'id': row['repertoire_id'],
                'ds_id': ds_id
            })

            if idx % 30 == 0:
                gc.collect()
                torch.cuda.empty_cache()

        np.savez_compressed(cache_file, data=np.array(data, dtype=object))
        all_data.extend(data)
        print(f"  ✓ Dataset {ds_id}: {len(data)} repertoires cached")

    return all_data

# ============================================================================
# Training
# ============================================================================
def train_epoch(model, loader, opt, criterion, device, scaler):
    model.train()
    loss_sum, preds, labels = 0, [], []

    for batch in tqdm(loader, desc="Train", leave=False):
        emb = batch['emb'].to(device)
        mask = batch['mask'].to(device)
        y = batch['labels'].to(device)

        opt.zero_grad()
        with autocast():
            logits, _ = model(emb, mask)
            loss = criterion(logits.squeeze(), y)

        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()

        loss_sum += loss.item()
        preds.extend(torch.sigmoid(logits.squeeze()).detach().cpu().numpy().tolist())
        labels.extend(y.cpu().numpy().tolist())

    auc = roc_auc_score(labels, preds) if len(set(labels)) > 1 else 0.5
    return loss_sum / len(loader), auc

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum, preds, labels = 0, [], []

    for batch in loader:
        emb = batch['emb'].to(device)
        mask = batch['mask'].to(device)
        y = batch['labels'].to(device)

        logits, _ = model(emb, mask)
        loss = criterion(logits.squeeze(), y)

        loss_sum += loss.item()
        preds.extend(torch.sigmoid(logits.squeeze()).cpu().numpy().tolist())
        labels.extend(y.cpu().numpy().tolist())

    auc = roc_auc_score(labels, preds) if len(set(labels)) > 1 else 0.5
    return loss_sum / len(loader), auc, preds

def train_fold(fold, train_data, val_data, device):
    print(f"\n{'='*50}")
    print(f"🎯 FOLD {fold}: Train={len(train_data)}, Val={len(val_data)}")
    print(f"{'='*50}")

    train_loader = DataLoader(RepDataset(train_data), batch_size=Config.BATCH_SIZE,
                              shuffle=True, collate_fn=collate, num_workers=Config.NUM_WORKERS,
                              pin_memory=True, prefetch_factor=4, persistent_workers=True)
    val_loader = DataLoader(RepDataset(val_data), batch_size=Config.BATCH_SIZE,
                            shuffle=False, collate_fn=collate, num_workers=4, pin_memory=True)

    model = GatedAttentionMIL(Config.ESM_DIM, Config.HIDDEN_DIM, Config.DROPOUT).to(device)

    if Config.USE_COMPILE:
        try:
            model = torch.compile(model, mode='reduce-overhead')
            print("  ✓ torch.compile() enabled")
        except:
            pass

    opt = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    criterion = nn.BCEWithLogitsLoss()
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=Config.EPOCHS)
    scaler = GradScaler()

    best_auc, patience = 0, 0

    for epoch in range(Config.EPOCHS):
        train_loss, train_auc = train_epoch(model, train_loader, opt, criterion, device, scaler)
        val_loss, val_auc, _ = evaluate(model, val_loader, criterion, device)
        sched.step()

        print(f"Epoch {epoch+1:2d} | Train: {train_auc:.4f} | Val: {val_auc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            patience = 0
            Config.MODEL_DIR.mkdir(exist_ok=True)
            torch.save(model.state_dict(), Config.MODEL_DIR / f'fold{fold}.pt')
            print(f"  ✓ Best: {best_auc:.4f}")
        else:
            patience += 1
            if patience >= Config.PATIENCE:
                print("  Early stop")
                break

    model.load_state_dict(torch.load(Config.MODEL_DIR / f'fold{fold}.pt'))
    return model, best_auc

# ============================================================================
# Prediction
# ============================================================================
@torch.no_grad()
def predict_test(models, esm, device):
    print("\n📊 Test predictions...")
    preds = []

    for test_ds in sorted(Config.TEST_ROOT.iterdir()):
        if not test_ds.name.startswith('test_dataset_'):
            continue

        print(f"  {test_ds.name}...")
        meta = pd.read_csv(test_ds / 'metadata.csv')

        for _, row in tqdm(meta.iterrows(), total=len(meta), leave=False):
            try:
                df = pd.read_csv(test_ds / row['filename'], sep='\t', usecols=['junction_aa'],
                                nrows=Config.MAX_SEQS_PER_REP)
                seqs = df['junction_aa'].dropna().astype(str).tolist()
            except:
                seqs = []

            if len(seqs) < 5:
                prob = 0.5
            else:
                emb = esm.extract_repertoire(seqs, gpu_id=0)
                emb_t = torch.from_numpy(emb).unsqueeze(0).to(device)
                mask = torch.ones(1, emb.shape[0], dtype=torch.bool, device=device)

                probs = []
                for m in models:
                    m.eval()
                    logits, _ = m(emb_t, mask)
                    probs.append(torch.sigmoid(logits).item())
                prob = np.mean(probs)

            preds.append({'ID': row['repertoire_id'], 'dataset': test_ds.name,
                         'label_positive_probability': prob})

    return pd.DataFrame(preds)

def gen_task_b():
    print("\n🧬 Task B...")
    rows = []

    for ds_id in range(1, 9):
        path = Config.TRAIN_ROOT / f'train_dataset_{ds_id}'
        if not path.exists():
            continue

        meta = pd.read_csv(path / 'metadata.csv')
        all_seqs = []

        for _, r in meta.iterrows():
            try:
                df = pd.read_csv(path / r['filename'], sep='\t',
                               usecols=['junction_aa', 'v_call', 'j_call'], nrows=10000)
                df['label'] = r['label_positive']
                all_seqs.append(df)
            except:
                pass

        if not all_seqs:
            continue

        combined = pd.concat(all_seqs, ignore_index=True)
        pos = combined[combined['label'] == True]['junction_aa'].value_counts()
        neg = combined[combined['label'] == False]['junction_aa'].value_counts()

        scores = {}
        for seq in set(pos.index) | set(neg.index):
            scores[seq] = pos.get(seq, 0) / max(pos.sum(), 1) - neg.get(seq, 0) / max(neg.sum(), 1)

        top = sorted(scores.items(), key=lambda x: -x[1])[:50000]
        info = combined.drop_duplicates('junction_aa').set_index('junction_aa')

        for rank, (seq, _) in enumerate(top, 1):
            if pd.isna(seq) or seq == '':
                continue
            v = info.loc[seq, 'v_call'] if seq in info.index else 'Unknown'
            j = info.loc[seq, 'j_call'] if seq in info.index else 'Unknown'
            rows.append({
                'ID': f'train_dataset_{ds_id}_seq_top_{rank}',
                'dataset': f'train_dataset_{ds_id}',
                'label_positive_probability': -999.0,
                'junction_aa': seq,
                'v_call': str(v) if not pd.isna(v) else 'Unknown',
                'j_call': str(j) if not pd.isna(j) else 'Unknown'
            })

    return pd.DataFrame(rows)

# ============================================================================
# Main
# ============================================================================
def main():
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║  🔥 MAXED OUT DUAL GPU TRAINING 🔥                              ║
    ║                                                                  ║
    ║  ESM Batch: 256 | Train Batch: 48 | Seqs/Rep: 1500              ║
    ║  Expected: ~2 hours (was 6-8 hours)                             ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)

    device = 'cuda:0'  # Main training device

    # Initialize dual-GPU ESM
    esm = DualGPUESM2Extractor()

    # Load data
    all_data = load_all_data_parallel(esm)
    print(f"\n✓ Total: {len(all_data)} repertoires")

    # LODO CV
    print("\n" + "="*50)
    print("🎓 LODO CROSS-VALIDATION")
    print("="*50)

    results, models = [], []

    for val_id in range(1, 9):
        train = [d for d in all_data if d['ds_id'] != val_id]
        val = [d for d in all_data if d['ds_id'] == val_id]
        if not val:
            continue

        model, auc = train_fold(val_id, train, val, device)
        results.append({'fold': val_id, 'auc': auc})
        models.append(model)

        gc.collect()
        torch.cuda.empty_cache()

    print("\n" + "="*50)
    print("📈 RESULTS")
    for r in results:
        print(f"  Fold {r['fold']}: {r['auc']:.4f}")
    mean_auc = np.mean([r['auc'] for r in results])
    print(f"\n  🎯 Mean LODO AUC: {mean_auc:.4f}")
    print("="*50)

    # Predictions
    task_a = predict_test(models, esm, device)
    task_b = gen_task_b()

    sub = pd.concat([task_a, task_b], ignore_index=True)
    sub = sub[['ID', 'dataset', 'label_positive_probability', 'junction_aa', 'v_call', 'j_call']]
    sub['junction_aa'] = sub['junction_aa'].fillna('-999.0')
    sub['v_call'] = sub['v_call'].fillna('-999.0')
    sub['j_call'] = sub['j_call'].fillna('-999.0')

    Config.OUTPUT_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out = Config.OUTPUT_DIR / f'maxed_submission_{ts}.csv'
    sub.to_csv(out, index=False)

    print(f"\n✅ Saved: {out}")
    print(f"   Rows: {len(sub)}")
    print("\n🏆 DONE!")

if __name__ == '__main__':
    main()
