#!/usr/bin/env python3
"""
Complete Submission Generator
=============================
Completes Task A + Task B and generates submission file.
Uses pre-trained models from ./models/fold*.pt
"""

import os
import gc
import json
import sys
import subprocess
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from typing import List, Dict
from collections import defaultdict
from scipy.stats import entropy
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

TOP_K = 50000
MAX_SEQS = 200

TEST_DATASETS = [
    'test_dataset_1', 'test_dataset_2', 'test_dataset_3',
    'test_dataset_4', 'test_dataset_5', 'test_dataset_6',
    'test_dataset_7_1', 'test_dataset_7_2',
    'test_dataset_8_1', 'test_dataset_8_2', 'test_dataset_8_3'
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ============================================================================
# Model
# ============================================================================

class FastClassifier(nn.Module):
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
# ESM-2
# ============================================================================

_esm = None

def get_esm():
    global _esm
    if _esm is None:
        print("Loading ESM-2 (650M)...")
        import esm
        model, alphabet = esm.pretrained.load_model_and_alphabet("esm2_t33_650M_UR50D")
        _esm = (model.to(device).eval(), alphabet.get_batch_converter())
        print("ESM-2 loaded!")
    return _esm


def extract_esm_mean(sequences: List[str]) -> np.ndarray:
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
# Task A: Predict test repertoires
# ============================================================================

def predict_test():
    print("\n" + "="*70)
    print("TASK A: Predicting 4,213 test repertoires")
    print("="*70)

    with open(os.path.join(CHECKPOINT_DIR, 'feature_names.json'), 'r') as f:
        feat_names = json.load(f)
    trad_dim = len(feat_names)
    print(f"Feature dim: {trad_dim}")

    predictions = []
    total_count = 0

    for test_ds in TEST_DATASETS:
        print(f"\n{test_ds}...")
        test_path = os.path.join(TEST_ROOT, test_ds)

        # Model selection
        if test_ds.startswith('test_dataset_7'):
            fold = 7
        elif test_ds.startswith('test_dataset_8'):
            fold = 8
        else:
            fold = int(test_ds.split('_')[-1])

        checkpoint = torch.load(f'{MODELS_DIR}/fold{fold}.pt', weights_only=False)
        model = FastClassifier(esm_dim=1280, trad_dim=trad_dim).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        tsv_files = list(Path(test_path).glob('*.tsv'))
        print(f"  {len(tsv_files)} repertoires, using fold {fold}")

        for tsv in tqdm(tsv_files, desc=f"  {test_ds}"):
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
                total_count += 1
            except Exception as e:
                print(f"    Error {rep_id}: {e}")
                predictions.append({
                    'ID': rep_id,
                    'dataset': test_ds,
                    'label_positive_probability': 0.5,
                    'junction_aa': -999.0,
                    'v_call': -999.0,
                    'j_call': -999.0
                })
                total_count += 1

        del model
        gc.collect()
        torch.cuda.empty_cache()

    df = pd.DataFrame(predictions)
    print(f"\nTask A complete: {len(df)} predictions")
    return df


# ============================================================================
# Task B: Identify top sequences
# ============================================================================

def identify_seqs():
    print("\n" + "="*70)
    print("TASK B: Identifying 8 x 50,000 sequences")
    print("="*70)

    all_seqs = []

    for ds_id in range(1, 9):
        print(f"\ntrain_dataset_{ds_id}...")
        train_path = os.path.join(TRAIN_ROOT, f'train_dataset_{ds_id}')
        metadata = pd.read_csv(os.path.join(train_path, 'metadata.csv'))

        scores = defaultdict(lambda: {'s': 0, 'c': 0, 'v': None, 'j': None})

        for _, row in tqdm(metadata.iterrows(), total=len(metadata), desc=f"  DS{ds_id}"):
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

        print(f"  {len(sorted_seqs)} sequences identified")
        gc.collect()

    df = pd.DataFrame(all_seqs)
    print(f"\nTask B complete: {len(df)} sequences")
    return df


# ============================================================================
# Submit to Kaggle
# ============================================================================

def submit_to_kaggle(path: str, message: str):
    print("\n" + "="*70)
    print("Submitting to Kaggle...")
    print("="*70)

    cmd = f'kaggle competitions submit -c adaptive-immune-profiling-challenge-2025 -f "{path}" -m "{message}"'
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
        print(f"  Result: {result.stdout}")
        if result.stderr:
            print(f"  Stderr: {result.stderr}")
        return result.returncode == 0
    except Exception as e:
        print(f"  Error: {e}")
        return False


# ============================================================================
# Main
# ============================================================================

def main():
    start_time = datetime.now()
    print(f"""
================================================================================
  AIRR-ML-25 Submission Generator
  Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}
  Target: 404,213 rows (4,213 predictions + 400,000 sequences)
================================================================================
    """)

    # Verify models exist
    for fold in range(1, 9):
        if not os.path.exists(f'{MODELS_DIR}/fold{fold}.pt'):
            print(f"ERROR: {MODELS_DIR}/fold{fold}.pt not found!")
            return 1

    print("All 8 fold models found!")

    # Task A
    task_a = predict_test()

    # Task B
    task_b = identify_seqs()

    # Combine
    print("\n" + "="*70)
    print("Generating submission file...")
    print("="*70)

    sub = pd.concat([task_a, task_b], ignore_index=True)
    sub = sub[['ID', 'dataset', 'label_positive_probability', 'junction_aa', 'v_call', 'j_call']]

    expected = 4213 + 8 * 50000
    print(f"  Rows: {len(sub)} (expected: {expected})")

    if len(sub) != expected:
        print(f"  WARNING: Row count mismatch!")

    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(SUBMISSION_DIR, f'submission_{ts}.csv')
    sub.to_csv(path, index=False)

    # Also save as latest
    latest_path = os.path.join(SUBMISSION_DIR, 'submission_latest.csv')
    sub.to_csv(latest_path, index=False)

    print(f"  Saved: {path}")
    print(f"  Size: {os.path.getsize(path) / 1e6:.2f} MB")

    probs = task_a['label_positive_probability']
    print(f"\n  Probability stats:")
    print(f"    Min: {probs.min():.4f}")
    print(f"    Max: {probs.max():.4f}")
    print(f"    Mean: {probs.mean():.4f}")
    print(f"    Std: {probs.std():.4f}")

    # Submit to Kaggle
    message = f"ESM2-MLP-Fast CV={0.5307:.4f} ({ts})"
    submit_to_kaggle(path, message)

    # Check submission status
    print("\n" + "="*70)
    print("Checking submission status...")
    print("="*70)
    try:
        result = subprocess.run(
            "kaggle competitions submissions -c adaptive-immune-profiling-challenge-2025 | head -10",
            shell=True, capture_output=True, text=True, timeout=60
        )
        print(result.stdout)
    except:
        print("  Could not check status")

    end_time = datetime.now()
    duration = end_time - start_time
    print(f"""
================================================================================
  COMPLETE!
  Duration: {duration}
  Submission: {path}
================================================================================
    """)

    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
