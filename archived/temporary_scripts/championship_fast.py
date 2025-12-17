#!/usr/bin/env python3
"""
🏆 AIRR-ML-25 Championship Pipeline - Fast Version

Strategy: Pre-aggregate ESM embeddings to repertoire-level vectors.
Each repertoire: 1280 (ESM mean) + 389 (traditional) = 1669 features
Total memory: ~25 MB for all 3,602 repertoires

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
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from typing import List, Dict
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

BATCH_SIZE = 32
NUM_EPOCHS = 30
EARLY_STOPPING = 5
LR = 1e-3
TOP_K = 50000
MAX_SEQS = 200

TEST_DATASETS = [
    'test_dataset_1', 'test_dataset_2', 'test_dataset_3',
    'test_dataset_4', 'test_dataset_5', 'test_dataset_6',
    'test_dataset_7_1', 'test_dataset_7_2',
    'test_dataset_8_1', 'test_dataset_8_2', 'test_dataset_8_3'
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Device: {device}")


# ============================================================================
# Pre-aggregate ESM embeddings
# ============================================================================

def load_aggregated_data() -> Dict:
    """Load all data with pre-aggregated ESM embeddings."""
    agg_path = os.path.join(CHECKPOINT_DIR, 'aggregated_features.npz')

    if os.path.exists(agg_path):
        print("📂 Loading pre-aggregated features...")
        data = np.load(agg_path, allow_pickle=True)
        return {
            'esm_agg': data['esm_agg'],
            'trad': data['trad'],
            'labels': data['labels'],
            'dataset_ids': data['dataset_ids'],
            'repertoire_ids': data['repertoire_ids']
        }

    print("🔧 Pre-aggregating ESM embeddings (one-time)...")

    all_esm_agg = []
    all_trad = []
    all_labels = []
    all_ds_ids = []
    all_rep_ids = []

    for ds_id in range(1, 9):
        print(f"   Dataset {ds_id}...")
        path = os.path.join(CHECKPOINT_DIR, f'dataset_{ds_id}.npz')
        data = np.load(path, allow_pickle=True)

        if 'processed_data' in data.keys():
            for item in tqdm(data['processed_data'], desc=f"      DS{ds_id}", leave=False):
                esm = item['esm_embeddings']
                esm_mean = esm.mean(axis=0)  # Mean pool: (500, 1280) -> (1280,)
                all_esm_agg.append(esm_mean)
                all_trad.append(item['trad_features'])
                all_labels.append(int(item['label']))
                all_ds_ids.append(int(item['dataset_id']))
                all_rep_ids.append(str(item['repertoire_id']))
        else:
            esm_all = data['esm_embeddings']
            for i in tqdm(range(len(data['labels'])), desc=f"      DS{ds_id}", leave=False):
                esm = esm_all[i]
                esm_mean = esm.mean(axis=0)
                all_esm_agg.append(esm_mean)
                all_trad.append(data['trad_features'][i])
                all_labels.append(int(data['labels'][i]))
                all_ds_ids.append(int(data['dataset_ids'][i]))
                all_rep_ids.append(str(data['repertoire_ids'][i]))

        data.close()
        gc.collect()

    # Convert to arrays
    esm_agg = np.array(all_esm_agg, dtype=np.float32)
    trad = np.array(all_trad, dtype=np.float32)
    labels = np.array(all_labels, dtype=np.int32)
    ds_ids = np.array(all_ds_ids, dtype=np.int32)
    rep_ids = np.array(all_rep_ids)

    # Save for future use
    np.savez_compressed(agg_path, esm_agg=esm_agg, trad=trad, labels=labels,
                        dataset_ids=ds_ids, repertoire_ids=rep_ids)

    print(f"   ✓ Saved aggregated features: {esm_agg.shape[0]} repertoires")
    print(f"   ✓ ESM dim: {esm_agg.shape[1]}, Trad dim: {trad.shape[1]}")

    return {
        'esm_agg': esm_agg,
        'trad': trad,
        'labels': labels,
        'dataset_ids': ds_ids,
        'repertoire_ids': rep_ids
    }


# ============================================================================
# Model
# ============================================================================

class FastClassifier(nn.Module):
    """Simple MLP classifier for aggregated features."""

    def __init__(self, esm_dim: int = 1280, trad_dim: int = 389):
        super().__init__()
        total_dim = esm_dim + trad_dim

        self.net = nn.Sequential(
            nn.Linear(total_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

    def forward(self, esm, trad):
        x = torch.cat([esm, trad], dim=1)
        return self.net(x)


# ============================================================================
# Dataset
# ============================================================================

class AggregatedDataset(Dataset):
    def __init__(self, esm, trad, labels):
        self.esm = torch.from_numpy(esm).float()
        self.trad = torch.from_numpy(trad).float()
        self.labels = torch.from_numpy(labels).float()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.esm[idx], self.trad[idx], self.labels[idx]


# ============================================================================
# Training
# ============================================================================

def train_fold(fold_id: int, data: Dict, train_mask: np.ndarray, val_mask: np.ndarray):
    """Train one fold with pre-aggregated features."""
    print(f"\n{'='*60}")
    print(f"🎯 FOLD {fold_id}")
    print(f"{'='*60}")

    # Split data
    train_esm = data['esm_agg'][train_mask]
    train_trad = data['trad'][train_mask]
    train_labels = data['labels'][train_mask]

    val_esm = data['esm_agg'][val_mask]
    val_trad = data['trad'][val_mask]
    val_labels = data['labels'][val_mask]

    print(f"   Train: {len(train_labels)} | Val: {len(val_labels)}")

    train_loader = DataLoader(
        AggregatedDataset(train_esm, train_trad, train_labels),
        batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        AggregatedDataset(val_esm, val_trad, val_labels),
        batch_size=BATCH_SIZE, shuffle=False
    )

    model = FastClassifier(esm_dim=1280, trad_dim=train_trad.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    best_auc = 0
    patience = 0

    for epoch in range(NUM_EPOCHS):
        # Train
        model.train()
        train_preds, train_labels_all = [], []

        for esm, trad, labels in train_loader:
            esm = esm.to(device)
            trad = trad.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(esm, trad).squeeze()
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_preds.extend(torch.sigmoid(logits).detach().cpu().numpy())
            train_labels_all.extend(labels.cpu().numpy())

        scheduler.step()
        train_auc = roc_auc_score(train_labels_all, train_preds)

        # Validate
        model.eval()
        val_preds, val_labels_all = [], []

        with torch.no_grad():
            for esm, trad, labels in val_loader:
                esm = esm.to(device)
                trad = trad.to(device)
                labels = labels.to(device)

                logits = model(esm, trad).squeeze()
                val_preds.extend(torch.sigmoid(logits).cpu().numpy())
                val_labels_all.extend(labels.cpu().numpy())

        val_auc = roc_auc_score(val_labels_all, val_preds)

        if (epoch + 1) % 5 == 0 or val_auc > best_auc:
            print(f"   Epoch {epoch+1:2d}: Train={train_auc:.4f}, Val={val_auc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            patience = 0
            os.makedirs(MODELS_DIR, exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'fold_id': fold_id,
                'val_auc': val_auc,
                'trad_dim': train_trad.shape[1]
            }, f'{MODELS_DIR}/fold{fold_id}.pt')
        else:
            patience += 1
            if patience >= EARLY_STOPPING:
                print(f"   Early stopping at epoch {epoch+1}")
                break

    print(f"   ✅ Best: {best_auc:.4f}")
    return best_auc


def train_all_folds(data: Dict):
    """Train all 8 folds."""
    print("\n" + "="*70)
    print("🎓 LEAVE-ONE-DATASET-OUT CV (Fast)")
    print("="*70)

    results = []
    ds_ids = data['dataset_ids']

    for val_id in range(1, 9):
        train_mask = ds_ids != val_id
        val_mask = ds_ids == val_id

        auc = train_fold(val_id, data, train_mask, val_mask)
        results.append({'fold': val_id, 'auc': auc})

    print("\n" + "="*70)
    print("📈 RESULTS")
    print("="*70)
    for r in results:
        print(f"   Fold {r['fold']}: {r['auc']:.4f}")

    mean_auc = np.mean([r['auc'] for r in results])
    std_auc = np.std([r['auc'] for r in results])
    print(f"\n🎯 Mean: {mean_auc:.4f} ± {std_auc:.4f}")

    with open(os.path.join(CHECKPOINT_DIR, 'cv_results.json'), 'w') as f:
        json.dump({'results': results, 'mean': mean_auc, 'std': std_auc}, f)

    return results


# ============================================================================
# ESM-2 for Test Data
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


def extract_esm_mean(sequences: List[str]) -> np.ndarray:
    """Extract mean-pooled ESM embedding for a repertoire."""
    if len(sequences) > MAX_SEQS:
        np.random.seed(42)
        indices = np.random.choice(len(sequences), MAX_SEQS, replace=False)
        sequences = [sequences[i] for i in sorted(indices)]

    valid = set("ACDEFGHIKLMNPQRSTVWYX")
    cleaned = [''.join(c if c in valid else 'X' for c in s.upper()) for s in sequences if s]
    cleaned = [s for s in cleaned if s]

    if not cleaned:
        return np.zeros(1280, dtype=np.float32)

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

    return np.mean(embeddings, axis=0).astype(np.float32)


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
        model = FastClassifier(esm_dim=1280, trad_dim=trad_dim).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        tsv_files = list(Path(test_path).glob('*.tsv'))
        print(f"   {len(tsv_files)} reps, using fold {fold}")

        for tsv in tqdm(tsv_files, desc=f"   {test_ds}", leave=False):
            rep_id = tsv.stem

            try:
                df = pd.read_csv(tsv, sep='\t')
                feat = extract_features(df)
                trad = standardize(feat, feat_names)

                seqs = df['junction_aa'].dropna().astype(str).tolist()
                if seqs:
                    esm_mean = extract_esm_mean(seqs)

                    esm_t = torch.from_numpy(esm_mean).unsqueeze(0).to(device)
                    trad_t = torch.from_numpy(trad).unsqueeze(0).to(device)

                    with torch.no_grad():
                        prob = torch.sigmoid(model(esm_t, trad_t)).item()
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
    ║  🏆 AIRR-ML-25 Championship (Fast Version) 🏆                  ║
    ║  Strategy: Pre-aggregated ESM + MLP                             ║
    ║  Target: Beat GROZD (0.81364) → 0.82+                           ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)

    # Load aggregated data
    data = load_aggregated_data()
    print(f"\n📊 Data loaded: {len(data['labels'])} repertoires")
    print(f"   ESM dim: {data['esm_agg'].shape[1]}")
    print(f"   Trad dim: {data['trad'].shape[1]}")

    # Train if needed
    if not all(os.path.exists(f'{MODELS_DIR}/fold{i}.pt') for i in range(1, 9)):
        train_all_folds(data)
    else:
        print("\n✅ Models exist!")

    # Free training data memory
    del data
    gc.collect()

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
