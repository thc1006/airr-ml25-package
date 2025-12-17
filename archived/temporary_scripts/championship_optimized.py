#!/usr/bin/env python3
"""
🏆 AIRR-ML-25 Championship Pipeline - Memory Optimized Version

Key optimizations:
1. Per-fold loading: Only load datasets needed for each fold
2. Immediate memory release after each fold
3. Streaming test predictions
4. Per-dataset Task B processing

Target: Beat GROZD (0.81364) → Achieve 0.82+
"""

import os
import sys
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
from typing import List, Dict, Tuple
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
SAMPLE_SUBMISSION = './data/sample_submissions.csv'

MAX_SEQS_PER_REPERTOIRE = 500
BATCH_SIZE = 4
NUM_EPOCHS = 20
EARLY_STOPPING_PATIENCE = 4
LEARNING_RATE = 1e-4
TOP_K_SEQUENCES = 50000

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
# Model Architecture
# ============================================================================

class AttentionAggregator(nn.Module):
    def __init__(self, input_dim: int = 1280, num_heads: int = 4):
        super().__init__()
        self.attention = nn.MultiheadAttention(input_dim, num_heads, dropout=0.1, batch_first=True)
        self.query = nn.Parameter(torch.randn(1, 1, input_dim))
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, x, mask=None):
        batch_size = x.size(0)
        query = self.query.expand(batch_size, -1, -1)
        out, weights = self.attention(query, x, x, key_padding_mask=~mask if mask is not None else None)
        return self.norm(out.squeeze(1)), weights


class ChampionshipClassifier(nn.Module):
    def __init__(self, esm_dim: int = 1280, trad_dim: int = 389):
        super().__init__()
        self.attention = AttentionAggregator(esm_dim, num_heads=4)

        self.mlp = nn.Sequential(
            nn.Linear(esm_dim + trad_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, esm_emb, trad_feat, mask=None):
        agg_esm, attn_weights = self.attention(esm_emb, mask)
        combined = torch.cat([agg_esm, trad_feat], dim=1)
        return self.mlp(combined), attn_weights


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


def collate_fn(batch):
    max_len = max(item['esm_embeddings'].shape[0] for item in batch)
    esm_dim = batch[0]['esm_embeddings'].shape[1]
    trad_dim = batch[0]['trad_features'].shape[0]
    bs = len(batch)

    esm = torch.zeros(bs, max_len, esm_dim)
    masks = torch.zeros(bs, max_len, dtype=torch.bool)
    trad = torch.zeros(bs, trad_dim)
    labels = torch.zeros(bs)

    for i, item in enumerate(batch):
        seq_len = item['esm_embeddings'].shape[0]
        esm[i, :seq_len] = torch.from_numpy(item['esm_embeddings'])
        masks[i, :seq_len] = True
        trad[i] = torch.from_numpy(item['trad_features'])
        labels[i] = item.get('label', 0)

    return {'esm': esm.float(), 'trad': trad.float(), 'masks': masks, 'labels': labels}


# ============================================================================
# Data Loading (Memory Optimized)
# ============================================================================

def load_single_dataset(dataset_id: int) -> List[Dict]:
    """Load a single dataset from checkpoint."""
    path = os.path.join(CHECKPOINT_DIR, f'dataset_{dataset_id}.npz')
    if not os.path.exists(path):
        return []

    data = np.load(path, allow_pickle=True)
    result = []

    if 'processed_data' in data.keys():
        for item in data['processed_data']:
            result.append({
                'esm_embeddings': item['esm_embeddings'],
                'trad_features': item['trad_features'],
                'label': int(item['label']),
                'repertoire_id': str(item['repertoire_id']),
                'dataset_id': int(item['dataset_id'])
            })
    else:
        for i in range(len(data['labels'])):
            result.append({
                'esm_embeddings': data['esm_embeddings'][i],
                'trad_features': data['trad_features'][i],
                'label': int(data['labels'][i]),
                'repertoire_id': str(data['repertoire_ids'][i]),
                'dataset_id': int(data['dataset_ids'][i])
            })

    data.close()
    return result


def load_feature_names() -> List[str]:
    path = os.path.join(CHECKPOINT_DIR, 'feature_names.json')
    with open(path, 'r') as f:
        return json.load(f)


# ============================================================================
# Training Functions
# ============================================================================

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    preds, labels_all = [], []

    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

    for batch in loader:
        esm = batch['esm'].to(device)
        trad = batch['trad'].to(device)
        masks = batch['masks'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        if scaler:
            with torch.cuda.amp.autocast():
                logits, _ = model(esm, trad, masks)
                loss = criterion(logits.squeeze(), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits, _ = model(esm, trad, masks)
            loss = criterion(logits.squeeze(), labels)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        preds.extend(torch.sigmoid(logits.squeeze()).detach().cpu().numpy().flatten())
        labels_all.extend(labels.cpu().numpy().flatten())

        torch.cuda.empty_cache()

    return total_loss / len(loader), roc_auc_score(labels_all, preds)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    preds, labels_all = [], []

    with torch.no_grad():
        for batch in loader:
            esm = batch['esm'].to(device)
            trad = batch['trad'].to(device)
            masks = batch['masks'].to(device)
            labels = batch['labels'].to(device)

            logits, _ = model(esm, trad, masks)
            loss = criterion(logits.squeeze(), labels)

            total_loss += loss.item()
            preds.extend(torch.sigmoid(logits.squeeze()).cpu().numpy().flatten())
            labels_all.extend(labels.cpu().numpy().flatten())

    return total_loss / len(loader), roc_auc_score(labels_all, preds)


def train_fold(fold_id: int, train_ids: List[int], val_id: int, trad_dim: int):
    """Train one fold with memory-optimized loading."""
    print(f"\n{'='*60}")
    print(f"🎯 FOLD {fold_id}: Train on datasets {train_ids}, Val on dataset {val_id}")
    print(f"{'='*60}")

    # Load training data incrementally
    print("   Loading training data...")
    train_data = []
    for ds_id in train_ids:
        ds_data = load_single_dataset(ds_id)
        train_data.extend(ds_data)
        print(f"      Dataset {ds_id}: {len(ds_data)} repertoires")
        gc.collect()

    # Load validation data
    print("   Loading validation data...")
    val_data = load_single_dataset(val_id)
    print(f"      Dataset {val_id}: {len(val_data)} repertoires")

    print(f"   Total: Train={len(train_data)}, Val={len(val_data)}")

    # Create dataloaders
    train_loader = DataLoader(
        RepertoireDataset(train_data), batch_size=BATCH_SIZE,
        shuffle=True, collate_fn=collate_fn, num_workers=0
    )
    val_loader = DataLoader(
        RepertoireDataset(val_data), batch_size=BATCH_SIZE,
        shuffle=False, collate_fn=collate_fn, num_workers=0
    )

    # Initialize model
    model = ChampionshipClassifier(esm_dim=1280, trad_dim=trad_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    best_auc = 0
    patience_counter = 0

    for epoch in range(NUM_EPOCHS):
        train_loss, train_auc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_auc = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_auc)

        print(f"   Epoch {epoch+1:2d}: Train AUC={train_auc:.4f}, Val AUC={val_auc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            os.makedirs(MODELS_DIR, exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'fold_id': fold_id,
                'val_auc': val_auc,
                'trad_dim': trad_dim
            }, f'{MODELS_DIR}/fold{fold_id}.pt')
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"   Early stopping at epoch {epoch+1}")
                break

    print(f"   ✅ Best Val AUC: {best_auc:.4f}")

    # Cleanup
    del train_data, val_data, train_loader, val_loader, model, optimizer
    gc.collect()
    torch.cuda.empty_cache()

    return best_auc


def train_all_folds():
    """Train all 8 folds with per-fold memory management."""
    print("\n" + "="*70)
    print("🎓 LEAVE-ONE-DATASET-OUT CROSS-VALIDATION (Memory Optimized)")
    print("="*70)

    feature_names = load_feature_names()
    trad_dim = len(feature_names)
    print(f"Feature dimension: {trad_dim}")

    results = []

    for val_id in range(1, 9):
        train_ids = [i for i in range(1, 9) if i != val_id]
        val_auc = train_fold(val_id, train_ids, val_id, trad_dim)
        results.append({'fold': val_id, 'val_auc': val_auc})

        gc.collect()
        torch.cuda.empty_cache()

    # Summary
    print("\n" + "="*70)
    print("📈 CROSS-VALIDATION RESULTS")
    print("="*70)
    for r in results:
        print(f"   Fold {r['fold']}: Val AUC = {r['val_auc']:.4f}")

    mean_auc = np.mean([r['val_auc'] for r in results])
    std_auc = np.std([r['val_auc'] for r in results])
    print(f"\n🎯 Mean AUC: {mean_auc:.4f} ± {std_auc:.4f}")

    # Save results
    with open(os.path.join(CHECKPOINT_DIR, 'cv_results.json'), 'w') as f:
        json.dump({'results': results, 'mean_auc': mean_auc, 'std_auc': std_auc}, f, indent=2)

    return results


# ============================================================================
# ESM-2 Feature Extractor (Lazy Load)
# ============================================================================

_esm_extractor = None

def get_esm_extractor():
    global _esm_extractor
    if _esm_extractor is None:
        print("🔧 Loading ESM-2 model...")
        import esm
        model, alphabet = esm.pretrained.load_model_and_alphabet("esm2_t33_650M_UR50D")
        model = model.to(device).eval()
        batch_converter = alphabet.get_batch_converter()
        _esm_extractor = (model, batch_converter)
        print("   ✓ ESM-2 loaded")
    return _esm_extractor


def extract_esm_embeddings(sequences: List[str], max_seqs: int = 500) -> np.ndarray:
    """Extract ESM-2 embeddings for sequences."""
    if len(sequences) > max_seqs:
        np.random.seed(42)
        indices = np.random.choice(len(sequences), max_seqs, replace=False)
        sequences = [sequences[i] for i in sorted(indices)]

    valid_aa = set("ACDEFGHIKLMNPQRSTVWYX")
    cleaned = [''.join(c if c in valid_aa else 'X' for c in s.upper()) for s in sequences if s]
    cleaned = [s for s in cleaned if s]

    if not cleaned:
        return np.zeros((1, 1280))

    model, batch_converter = get_esm_extractor()
    embeddings = []

    with torch.no_grad():
        for i in range(0, len(cleaned), 16):
            batch = cleaned[i:i+16]
            batch_labels = [(f"s{j}", s) for j, s in enumerate(batch)]
            _, _, tokens = batch_converter(batch_labels)
            tokens = tokens.to(device)

            results = model(tokens, repr_layers=[33], return_contacts=False)
            reps = results["representations"][33]

            for j, seq_len in enumerate([len(s) for s in batch]):
                embeddings.append(reps[j, 1:seq_len+1].mean(0).cpu().numpy())

            del tokens, results, reps
            torch.cuda.empty_cache()

    return np.array(embeddings)


# ============================================================================
# Traditional Feature Extraction
# ============================================================================

def extract_features(df: pd.DataFrame) -> Dict[str, float]:
    """Extract traditional features from repertoire."""
    features = {}
    total = len(df)
    if total == 0:
        return features

    # V/J gene usage
    if 'v_call' in df.columns:
        for gene, count in df['v_call'].value_counts().head(50).items():
            if pd.notna(gene):
                features[f"v_{gene}"] = count / total

    if 'j_call' in df.columns:
        for gene, count in df['j_call'].value_counts().head(50).items():
            if pd.notna(gene):
                features[f"j_{gene}"] = count / total

    # VJ pairs
    if 'v_call' in df.columns and 'j_call' in df.columns:
        pairs = df.apply(lambda x: f"{x['v_call']}_{x['j_call']}", axis=1)
        for pair, count in pairs.value_counts().head(50).items():
            if pd.notna(pair):
                features[f"vj_{pair}"] = count / total

    # Clonality
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


def standardize_features(feat_dict: Dict, feat_names: List[str]) -> np.ndarray:
    """Convert feature dict to array."""
    arr = np.zeros(len(feat_names))
    for i, name in enumerate(feat_names):
        if name in feat_dict:
            v = feat_dict[name]
            arr[i] = 0.0 if pd.isna(v) or np.isinf(v) else float(v)
    return arr


# ============================================================================
# Task A: Test Predictions
# ============================================================================

def predict_test_data():
    """Generate Task A predictions with per-repertoire processing."""
    print("\n" + "="*70)
    print("🔮 TASK A: TEST PREDICTIONS")
    print("="*70)

    feature_names = load_feature_names()
    trad_dim = len(feature_names)
    predictions = []

    for test_dataset in TEST_DATASETS:
        print(f"\n📂 {test_dataset}...")
        test_path = os.path.join(TEST_ROOT, test_dataset)

        # Determine model to use
        if test_dataset.startswith('test_dataset_7'):
            model_fold = 7
        elif test_dataset.startswith('test_dataset_8'):
            model_fold = 8
        else:
            model_fold = int(test_dataset.split('_')[-1])

        # Load model
        checkpoint = torch.load(f'{MODELS_DIR}/fold{model_fold}.pt')
        model = ChampionshipClassifier(esm_dim=1280, trad_dim=trad_dim).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        # Process each repertoire
        tsv_files = list(Path(test_path).glob('*.tsv'))
        print(f"   {len(tsv_files)} repertoires, using fold {model_fold} model")

        for tsv_file in tqdm(tsv_files, desc=f"   {test_dataset}", leave=False):
            rep_id = tsv_file.stem

            try:
                df = pd.read_csv(tsv_file, sep='\t')

                # Traditional features
                feat_dict = extract_features(df)
                trad_feat = standardize_features(feat_dict, feature_names)

                # ESM embeddings
                seqs = df['junction_aa'].dropna().astype(str).tolist()
                if seqs:
                    esm_emb = extract_esm_embeddings(seqs, MAX_SEQS_PER_REPERTOIRE)

                    # Predict
                    esm_t = torch.from_numpy(esm_emb).unsqueeze(0).float().to(device)
                    trad_t = torch.from_numpy(trad_feat).unsqueeze(0).float().to(device)
                    mask = torch.ones(1, esm_emb.shape[0], dtype=torch.bool).to(device)

                    with torch.no_grad():
                        logits, _ = model(esm_t, trad_t, mask)
                        prob = torch.sigmoid(logits).item()
                else:
                    prob = 0.5

                predictions.append({
                    'ID': rep_id,
                    'dataset': test_dataset,
                    'label_positive_probability': prob,
                    'junction_aa': -999.0,
                    'v_call': -999.0,
                    'j_call': -999.0
                })

            except Exception as e:
                print(f"      ⚠️ {rep_id}: {e}")
                predictions.append({
                    'ID': rep_id,
                    'dataset': test_dataset,
                    'label_positive_probability': 0.5,
                    'junction_aa': -999.0,
                    'v_call': -999.0,
                    'j_call': -999.0
                })

        del model
        gc.collect()
        torch.cuda.empty_cache()

    df = pd.DataFrame(predictions)
    print(f"\n✅ Task A: {len(df)} predictions")
    return df


# ============================================================================
# Task B: Sequence Identification
# ============================================================================

def identify_sequences():
    """Identify top 50,000 sequences per training dataset."""
    print("\n" + "="*70)
    print("🧬 TASK B: SEQUENCE IDENTIFICATION")
    print("="*70)

    all_seqs = []

    for dataset_id in range(1, 9):
        print(f"\n📂 train_dataset_{dataset_id}...")
        train_path = os.path.join(TRAIN_ROOT, f'train_dataset_{dataset_id}')
        metadata = pd.read_csv(os.path.join(train_path, 'metadata.csv'))

        # Score sequences by frequency in positive vs negative samples
        seq_scores = defaultdict(lambda: {'score': 0, 'count': 0, 'v': None, 'j': None})

        for _, row in tqdm(metadata.iterrows(), total=len(metadata), desc=f"   Dataset {dataset_id}", leave=False):
            tsv_path = os.path.join(train_path, row['filename'])
            label = 1 if row['label_positive'] else 0
            weight = 2.0 if label == 1 else 0.5

            try:
                df = pd.read_csv(tsv_path, sep='\t')
                for _, seq_row in df.iterrows():
                    junction = seq_row.get('junction_aa')
                    if pd.isna(junction):
                        continue
                    junction = str(junction)
                    v = str(seq_row.get('v_call', '')) if pd.notna(seq_row.get('v_call')) else ''
                    j = str(seq_row.get('j_call', '')) if pd.notna(seq_row.get('j_call')) else ''

                    seq_scores[junction]['score'] += weight
                    seq_scores[junction]['count'] += 1
                    if v and not seq_scores[junction]['v']:
                        seq_scores[junction]['v'] = v
                    if j and not seq_scores[junction]['j']:
                        seq_scores[junction]['j'] = j
            except:
                continue

        # Top 50,000
        sorted_seqs = sorted(seq_scores.items(), key=lambda x: (x[1]['score'], x[1]['count']), reverse=True)[:TOP_K_SEQUENCES]

        for rank, (junction, info) in enumerate(sorted_seqs, 1):
            all_seqs.append({
                'ID': f'train_dataset_{dataset_id}_seq_top_{rank}',
                'dataset': f'train_dataset_{dataset_id}',
                'label_positive_probability': -999.0,
                'junction_aa': junction,
                'v_call': info['v'] if info['v'] else 'TRBV20-1',
                'j_call': info['j'] if info['j'] else 'TRBJ2-7'
            })

        print(f"   ✓ {len(sorted_seqs)} sequences selected")
        gc.collect()

    df = pd.DataFrame(all_seqs)
    print(f"\n✅ Task B: {len(df)} sequences")
    return df


# ============================================================================
# Main Pipeline
# ============================================================================

def main():
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║  🏆 AIRR-ML-25 Championship Pipeline (Memory Optimized) 🏆     ║
    ║  Target: Beat GROZD (0.81364) → Achieve 0.82+                   ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)

    # Check if models exist
    models_exist = all(os.path.exists(f'{MODELS_DIR}/fold{i}.pt') for i in range(1, 9))

    if not models_exist:
        print("\n🎓 Training models...")
        train_all_folds()
    else:
        print("\n✅ All 8 fold models found!")

    # Task A
    task_a_df = predict_test_data()

    # Task B
    task_b_df = identify_sequences()

    # Combine
    print("\n" + "="*70)
    print("📝 GENERATING SUBMISSION")
    print("="*70)

    submission = pd.concat([task_a_df, task_b_df], ignore_index=True)
    submission = submission[['ID', 'dataset', 'label_positive_probability', 'junction_aa', 'v_call', 'j_call']]

    expected = 4213 + 8 * 50000
    print(f"   Rows: {len(submission)} (expected: {expected})")
    print(f"   Match: {'✅' if len(submission) == expected else '❌'}")

    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(SUBMISSION_DIR, f'submission_{timestamp}.csv')
    submission.to_csv(path, index=False)
    submission.to_csv(os.path.join(SUBMISSION_DIR, 'submission_latest.csv'), index=False)

    print(f"\n💾 Saved: {path}")
    print(f"   Size: {os.path.getsize(path) / 1e6:.2f} MB")

    # Validate
    probs = task_a_df['label_positive_probability']
    print(f"\n📊 Task A Stats:")
    print(f"   Min: {probs.min():.4f}, Max: {probs.max():.4f}, Mean: {probs.mean():.4f}")
    print(f"   NaN: {submission.isna().sum().sum()}")

    print("\n" + "="*70)
    print("🏆 PIPELINE COMPLETE!")
    print("="*70)


if __name__ == '__main__':
    main()
