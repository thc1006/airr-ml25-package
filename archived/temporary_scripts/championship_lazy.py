#!/usr/bin/env python3
"""
🏆 AIRR-ML-25 Championship Pipeline - Lazy Loading Version

Ultra memory-efficient: Loads data batch-by-batch from disk.
Never holds more than one batch in memory.

Target: Beat GROZD (0.81364) → Achieve 0.82+
"""

import os
import gc
import json
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, IterableDataset
from tqdm import tqdm
from typing import List, Dict, Tuple, Iterator
from collections import defaultdict
from scipy.stats import entropy
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# Configuration
# ============================================================================
CHECKPOINT_DIR = './checkpoints'
MODELS_DIR = './models'
SUBMISSION_DIR = './submissions'
TRAIN_ROOT = './data/train_datasets/train_datasets'
TEST_ROOT = './data/test_datasets/test_datasets'

# CRITICAL: Reduce sequences to save memory
MAX_SEQS = 200  # Reduced from 500

BATCH_SIZE = 8
NUM_EPOCHS = 15
EARLY_STOPPING = 3
LR = 2e-4
TOP_K = 50000

TEST_DATASETS = [
    'test_dataset_1', 'test_dataset_2', 'test_dataset_3',
    'test_dataset_4', 'test_dataset_5', 'test_dataset_6',
    'test_dataset_7_1', 'test_dataset_7_2',
    'test_dataset_8_1', 'test_dataset_8_2', 'test_dataset_8_3'
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Device: {device}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")


# ============================================================================
# Model (Simplified for memory efficiency)
# ============================================================================

class SimpleClassifier(nn.Module):
    """Simplified classifier: Mean pooling + MLP"""

    def __init__(self, esm_dim: int = 1280, trad_dim: int = 389):
        super().__init__()
        self.esm_proj = nn.Linear(esm_dim, 256)
        self.mlp = nn.Sequential(
            nn.Linear(256 + trad_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, esm_emb, trad_feat, mask=None):
        # Mean pooling with mask
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).float()
            esm_sum = (esm_emb * mask_expanded).sum(dim=1)
            esm_mean = esm_sum / mask_expanded.sum(dim=1).clamp(min=1)
        else:
            esm_mean = esm_emb.mean(dim=1)

        esm_proj = self.esm_proj(esm_mean)
        combined = torch.cat([esm_proj, trad_feat], dim=1)
        return self.mlp(combined)


# ============================================================================
# Lazy Dataset - Loads one item at a time from disk
# ============================================================================

class LazyRepertoireDataset(Dataset):
    """Loads repertoire data lazily from disk."""

    def __init__(self, dataset_ids: List[int], max_seqs: int = 200):
        self.max_seqs = max_seqs
        self.index = []  # (dataset_id, item_index)

        # Build index without loading data
        for ds_id in dataset_ids:
            path = os.path.join(CHECKPOINT_DIR, f'dataset_{ds_id}.npz')
            data = np.load(path, allow_pickle=True)

            if 'processed_data' in data.keys():
                n_items = len(data['processed_data'])
            else:
                n_items = len(data['labels'])

            for i in range(n_items):
                self.index.append((ds_id, i))

            data.close()

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        ds_id, item_idx = self.index[idx]
        path = os.path.join(CHECKPOINT_DIR, f'dataset_{ds_id}.npz')
        data = np.load(path, allow_pickle=True)

        if 'processed_data' in data.keys():
            item = data['processed_data'][item_idx]
            esm = item['esm_embeddings'][:self.max_seqs]
            trad = item['trad_features']
            label = int(item['label'])
        else:
            esm = data['esm_embeddings'][item_idx][:self.max_seqs]
            trad = data['trad_features'][item_idx]
            label = int(data['labels'][item_idx])

        data.close()

        return {
            'esm': esm.astype(np.float32),
            'trad': trad.astype(np.float32),
            'label': label
        }


def collate_fn(batch):
    max_len = max(item['esm'].shape[0] for item in batch)
    esm_dim = batch[0]['esm'].shape[1]
    trad_dim = batch[0]['trad'].shape[0]
    bs = len(batch)

    esm = torch.zeros(bs, max_len, esm_dim)
    masks = torch.zeros(bs, max_len, dtype=torch.bool)
    trad = torch.zeros(bs, trad_dim)
    labels = torch.zeros(bs)

    for i, item in enumerate(batch):
        seq_len = item['esm'].shape[0]
        esm[i, :seq_len] = torch.from_numpy(item['esm'])
        masks[i, :seq_len] = True
        trad[i] = torch.from_numpy(item['trad'])
        labels[i] = item['label']

    return {'esm': esm, 'trad': trad, 'masks': masks, 'labels': labels}


# ============================================================================
# Training with Lazy Loading
# ============================================================================

def train_fold_lazy(fold_id: int, train_ids: List[int], val_id: int, trad_dim: int):
    """Train one fold with lazy data loading."""
    print(f"\n{'='*60}")
    print(f"🎯 FOLD {fold_id}: Train on {train_ids}, Val on {val_id}")
    print(f"{'='*60}")

    # Create lazy datasets
    train_dataset = LazyRepertoireDataset(train_ids, max_seqs=MAX_SEQS)
    val_dataset = LazyRepertoireDataset([val_id], max_seqs=MAX_SEQS)

    print(f"   Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            collate_fn=collate_fn, num_workers=0)

    # Model
    model = SimpleClassifier(esm_dim=1280, trad_dim=trad_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    best_auc = 0
    patience = 0

    for epoch in range(NUM_EPOCHS):
        # Train
        model.train()
        train_preds, train_labels = [], []

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            esm = batch['esm'].to(device)
            trad = batch['trad'].to(device)
            masks = batch['masks'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                logits = model(esm, trad, masks).squeeze()
                loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            train_preds.extend(torch.sigmoid(logits).detach().cpu().numpy().flatten())
            train_labels.extend(labels.cpu().numpy().flatten())

            torch.cuda.empty_cache()

        scheduler.step()
        train_auc = roc_auc_score(train_labels, train_preds)

        # Validate
        model.eval()
        val_preds, val_labels = [], []

        with torch.no_grad():
            for batch in val_loader:
                esm = batch['esm'].to(device)
                trad = batch['trad'].to(device)
                masks = batch['masks'].to(device)
                labels = batch['labels'].to(device)

                logits = model(esm, trad, masks).squeeze()
                val_preds.extend(torch.sigmoid(logits).cpu().numpy().flatten())
                val_labels.extend(labels.cpu().numpy().flatten())

        val_auc = roc_auc_score(val_labels, val_preds)
        print(f"   Epoch {epoch+1:2d}: Train={train_auc:.4f}, Val={val_auc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            patience = 0
            os.makedirs(MODELS_DIR, exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'fold_id': fold_id,
                'val_auc': val_auc,
                'trad_dim': trad_dim
            }, f'{MODELS_DIR}/fold{fold_id}.pt')
        else:
            patience += 1
            if patience >= EARLY_STOPPING:
                print(f"   Early stopping")
                break

    print(f"   ✅ Best: {best_auc:.4f}")

    del model, optimizer, train_dataset, val_dataset
    gc.collect()
    torch.cuda.empty_cache()

    return best_auc


def train_all_folds():
    """Train all 8 folds with lazy loading."""
    print("\n" + "="*70)
    print("🎓 LEAVE-ONE-DATASET-OUT CV (Lazy Loading)")
    print("="*70)

    # Get trad_dim
    with open(os.path.join(CHECKPOINT_DIR, 'feature_names.json'), 'r') as f:
        trad_dim = len(json.load(f))
    print(f"Feature dim: {trad_dim}")

    results = []
    for val_id in range(1, 9):
        train_ids = [i for i in range(1, 9) if i != val_id]
        auc = train_fold_lazy(val_id, train_ids, val_id, trad_dim)
        results.append({'fold': val_id, 'auc': auc})
        gc.collect()

    print("\n" + "="*70)
    print("📈 RESULTS")
    print("="*70)
    for r in results:
        print(f"   Fold {r['fold']}: {r['auc']:.4f}")

    mean_auc = np.mean([r['auc'] for r in results])
    print(f"\n🎯 Mean: {mean_auc:.4f}")

    with open(os.path.join(CHECKPOINT_DIR, 'cv_results.json'), 'w') as f:
        json.dump({'results': results, 'mean': mean_auc}, f)

    return results


# ============================================================================
# ESM-2 (Lazy Load)
# ============================================================================

_esm = None

def get_esm():
    global _esm
    if _esm is None:
        print("🔧 Loading ESM-2...")
        import esm
        model, alphabet = esm.pretrained.load_model_and_alphabet("esm2_t33_650M_UR50D")
        _esm = (model.to(device).eval(), alphabet.get_batch_converter())
        print("   ✓ Done")
    return _esm


def extract_esm(sequences: List[str]) -> np.ndarray:
    if len(sequences) > MAX_SEQS:
        np.random.seed(42)
        indices = np.random.choice(len(sequences), MAX_SEQS, replace=False)
        sequences = [sequences[i] for i in sorted(indices)]

    valid = set("ACDEFGHIKLMNPQRSTVWYX")
    cleaned = [''.join(c if c in valid else 'X' for c in s.upper()) for s in sequences if s]
    cleaned = [s for s in cleaned if s]

    if not cleaned:
        return np.zeros((1, 1280), dtype=np.float32)

    model, converter = get_esm()
    embeddings = []

    with torch.no_grad():
        for i in range(0, len(cleaned), 16):
            batch = cleaned[i:i+16]
            _, _, tokens = converter([(f"s{j}", s) for j, s in enumerate(batch)])
            tokens = tokens.to(device)

            reps = model(tokens, repr_layers=[33], return_contacts=False)["representations"][33]
            for j, seq_len in enumerate([len(s) for s in batch]):
                embeddings.append(reps[j, 1:seq_len+1].mean(0).cpu().numpy())

            del tokens, reps
            torch.cuda.empty_cache()

    return np.array(embeddings, dtype=np.float32)


# ============================================================================
# Feature Extraction
# ============================================================================

def extract_features(df: pd.DataFrame) -> Dict[str, float]:
    features = {}
    total = len(df)
    if total == 0:
        return features

    if 'v_call' in df.columns:
        for g, c in df['v_call'].value_counts().head(50).items():
            if pd.notna(g):
                features[f"v_{g}"] = c / total

    if 'j_call' in df.columns:
        for g, c in df['j_call'].value_counts().head(50).items():
            if pd.notna(g):
                features[f"j_{g}"] = c / total

    if 'v_call' in df.columns and 'j_call' in df.columns:
        pairs = df.apply(lambda x: f"{x['v_call']}_{x['j_call']}", axis=1)
        for p, c in pairs.value_counts().head(50).items():
            if pd.notna(p):
                features[f"vj_{p}"] = c / total

    if 'junction_aa' in df.columns:
        seqs = df['junction_aa'].dropna()
        if len(seqs) > 0:
            counts = seqs.value_counts()
            freqs = counts.values / counts.sum()
            features['shannon_entropy'] = entropy(freqs)
            features['gini_simpson'] = 1 - np.sum(freqs ** 2)
            features['d50'] = np.sum(np.cumsum(np.sort(freqs)[::-1]) <= 0.5)
            max_ent = np.log(len(counts))
            features['clonality'] = 1 - (features['shannon_entropy'] / max_ent) if max_ent > 0 else 0
            lengths = seqs.str.len()
            features['mean_length'] = lengths.mean()
            features['std_length'] = lengths.std() if len(lengths) > 1 else 0
            features['top_clone_freq'] = freqs[0]

    return {k: (0.0 if pd.isna(v) or np.isinf(v) else float(v)) for k, v in features.items()}


def standardize(feat_dict: Dict, feat_names: List[str]) -> np.ndarray:
    arr = np.zeros(len(feat_names), dtype=np.float32)
    for i, name in enumerate(feat_names):
        if name in feat_dict:
            v = feat_dict[name]
            arr[i] = 0.0 if pd.isna(v) or np.isinf(v) else float(v)
    return arr


# ============================================================================
# Task A
# ============================================================================

def predict_test():
    print("\n" + "="*70)
    print("🔮 TASK A")
    print("="*70)

    with open(os.path.join(CHECKPOINT_DIR, 'feature_names.json'), 'r') as f:
        feat_names = json.load(f)
    trad_dim = len(feat_names)

    predictions = []

    for test_ds in TEST_DATASETS:
        print(f"\n📂 {test_ds}...")
        test_path = os.path.join(TEST_ROOT, test_ds)

        # Model selection
        if test_ds.startswith('test_dataset_7'):
            fold = 7
        elif test_ds.startswith('test_dataset_8'):
            fold = 8
        else:
            fold = int(test_ds.split('_')[-1])

        checkpoint = torch.load(f'{MODELS_DIR}/fold{fold}.pt')
        model = SimpleClassifier(esm_dim=1280, trad_dim=trad_dim).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        tsv_files = list(Path(test_path).glob('*.tsv'))
        print(f"   {len(tsv_files)} reps, fold {fold}")

        for tsv in tqdm(tsv_files, desc=f"   {test_ds}", leave=False):
            rep_id = tsv.stem

            try:
                df = pd.read_csv(tsv, sep='\t')
                feat = extract_features(df)
                trad = standardize(feat, feat_names)

                seqs = df['junction_aa'].dropna().astype(str).tolist()
                if seqs:
                    esm = extract_esm(seqs)
                    esm_t = torch.from_numpy(esm).unsqueeze(0).to(device)
                    trad_t = torch.from_numpy(trad).unsqueeze(0).to(device)
                    mask = torch.ones(1, esm.shape[0], dtype=torch.bool).to(device)

                    with torch.no_grad():
                        prob = torch.sigmoid(model(esm_t, trad_t, mask)).item()
                else:
                    prob = 0.5

                predictions.append({
                    'ID': rep_id,
                    'dataset': test_ds,
                    'label_positive_probability': prob,
                    'junction_aa': -999.0,
                    'v_call': -999.0,
                    'j_call': -999.0
                })
            except Exception as e:
                print(f"      ⚠️ {rep_id}: {e}")
                predictions.append({
                    'ID': rep_id,
                    'dataset': test_ds,
                    'label_positive_probability': 0.5,
                    'junction_aa': -999.0,
                    'v_call': -999.0,
                    'j_call': -999.0
                })

        del model
        gc.collect()
        torch.cuda.empty_cache()

    df = pd.DataFrame(predictions)
    print(f"\n✅ {len(df)} predictions")
    return df


# ============================================================================
# Task B
# ============================================================================

def identify_seqs():
    print("\n" + "="*70)
    print("🧬 TASK B")
    print("="*70)

    all_seqs = []

    for ds_id in range(1, 9):
        print(f"\n📂 train_dataset_{ds_id}...")
        train_path = os.path.join(TRAIN_ROOT, f'train_dataset_{ds_id}')
        metadata = pd.read_csv(os.path.join(train_path, 'metadata.csv'))

        scores = defaultdict(lambda: {'s': 0, 'c': 0, 'v': None, 'j': None})

        for _, row in tqdm(metadata.iterrows(), total=len(metadata), desc=f"   DS{ds_id}", leave=False):
            tsv = os.path.join(train_path, row['filename'])
            weight = 2.0 if row['label_positive'] else 0.5

            try:
                df = pd.read_csv(tsv, sep='\t')
                for _, sr in df.iterrows():
                    junc = sr.get('junction_aa')
                    if pd.isna(junc):
                        continue
                    junc = str(junc)
                    v = str(sr.get('v_call', '')) if pd.notna(sr.get('v_call')) else ''
                    j = str(sr.get('j_call', '')) if pd.notna(sr.get('j_call')) else ''

                    scores[junc]['s'] += weight
                    scores[junc]['c'] += 1
                    if v and not scores[junc]['v']:
                        scores[junc]['v'] = v
                    if j and not scores[junc]['j']:
                        scores[junc]['j'] = j
            except:
                continue

        sorted_seqs = sorted(scores.items(), key=lambda x: (x[1]['s'], x[1]['c']), reverse=True)[:TOP_K]

        for rank, (junc, info) in enumerate(sorted_seqs, 1):
            all_seqs.append({
                'ID': f'train_dataset_{ds_id}_seq_top_{rank}',
                'dataset': f'train_dataset_{ds_id}',
                'label_positive_probability': -999.0,
                'junction_aa': junc,
                'v_call': info['v'] if info['v'] else 'TRBV20-1',
                'j_call': info['j'] if info['j'] else 'TRBJ2-7'
            })

        print(f"   ✓ {len(sorted_seqs)}")
        gc.collect()

    df = pd.DataFrame(all_seqs)
    print(f"\n✅ {len(df)} seqs")
    return df


# ============================================================================
# Main
# ============================================================================

def main():
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║  🏆 AIRR-ML-25 Championship (Lazy Loading) 🏆                  ║
    ║  Target: Beat GROZD (0.81364) → 0.82+                           ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)

    # Train if needed
    if not all(os.path.exists(f'{MODELS_DIR}/fold{i}.pt') for i in range(1, 9)):
        train_all_folds()
    else:
        print("\n✅ Models exist!")

    # Task A
    task_a = predict_test()

    # Task B
    task_b = identify_seqs()

    # Combine
    print("\n" + "="*70)
    print("📝 SUBMISSION")
    print("="*70)

    sub = pd.concat([task_a, task_b], ignore_index=True)
    sub = sub[['ID', 'dataset', 'label_positive_probability', 'junction_aa', 'v_call', 'j_call']]

    expected = 4213 + 8 * 50000
    print(f"   Rows: {len(sub)} (expected: {expected}) {'✅' if len(sub) == expected else '❌'}")

    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(SUBMISSION_DIR, f'submission_{ts}.csv')
    sub.to_csv(path, index=False)
    sub.to_csv(os.path.join(SUBMISSION_DIR, 'submission_latest.csv'), index=False)

    print(f"\n💾 Saved: {path}")
    print(f"   Size: {os.path.getsize(path) / 1e6:.2f} MB")

    probs = task_a['label_positive_probability']
    print(f"\n📊 Stats: min={probs.min():.3f}, max={probs.max():.3f}, mean={probs.mean():.3f}")

    print("\n🏆 DONE!")


if __name__ == '__main__':
    main()
