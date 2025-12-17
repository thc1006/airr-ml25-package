#!/usr/bin/env python3
"""
Fast Submission Generator - Memory Efficient
=============================================
Generates submission.csv in a streaming manner to avoid memory issues.
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
from collections import defaultdict
from scipy.stats import entropy
import warnings
warnings.filterwarnings('ignore')

# Configuration
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

def extract_esm_mean(sequences):
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

def extract_features(df):
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

def standardize(feat_dict, feat_names):
    arr = np.zeros(len(feat_names), dtype=np.float32)
    for i, name in enumerate(feat_names):
        if name in feat_dict:
            v = feat_dict[name]
            arr[i] = 0.0 if pd.isna(v) or np.isinf(v) else float(v)
    return arr

def main():
    start_time = datetime.now()
    print(f"\n{'='*70}")
    print(f"  AIRR-ML-25 Submission Generator (Memory Efficient)")
    print(f"  Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")

    # Load feature names
    with open(os.path.join(CHECKPOINT_DIR, 'feature_names.json'), 'r') as f:
        feat_names = json.load(f)
    trad_dim = len(feat_names)
    print(f"Feature dim: {trad_dim}")

    # Create output directory and file
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(SUBMISSION_DIR, f'submission_{ts}.csv')

    # Write header
    with open(output_path, 'w') as f:
        f.write("ID,dataset,label_positive_probability,junction_aa,v_call,j_call\n")

    prediction_count = 0
    sequence_count = 0

    # ========== TASK A: Predictions ==========
    print("\n" + "="*70)
    print("TASK A: Predicting test repertoires (streaming to file)")
    print("="*70)

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

        batch_rows = []
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

                batch_rows.append(f"{rep_id},{test_ds},{prob},-999.0,-999.0,-999.0\n")
                prediction_count += 1

            except Exception as e:
                print(f"    Error {rep_id}: {e}")
                batch_rows.append(f"{rep_id},{test_ds},0.5,-999.0,-999.0,-999.0\n")
                prediction_count += 1

            # Write batch every 100 items
            if len(batch_rows) >= 100:
                with open(output_path, 'a') as f:
                    f.writelines(batch_rows)
                batch_rows = []

        # Write remaining
        if batch_rows:
            with open(output_path, 'a') as f:
                f.writelines(batch_rows)
            batch_rows = []

        del model
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\nTask A complete: {prediction_count} predictions written")

    # ========== TASK B: Sequences ==========
    print("\n" + "="*70)
    print("TASK B: Identifying sequences (streaming to file)")
    print("="*70)

    for ds_id in range(1, 9):
        print(f"\ntrain_dataset_{ds_id}...")
        train_path = os.path.join(TRAIN_ROOT, f'train_dataset_{ds_id}')
        metadata = pd.read_csv(os.path.join(train_path, 'metadata.csv'))

        # Use simple dict for scores
        scores = {}

        for _, row in tqdm(metadata.iterrows(), total=len(metadata), desc=f"  DS{ds_id}"):
            tsv = os.path.join(train_path, row['filename'])
            weight = 2.0 if row['label_positive'] else 0.5

            try:
                df = pd.read_csv(tsv, sep='\t', usecols=['junction_aa', 'v_call', 'j_call'])
                for _, sr in df.iterrows():
                    junc = sr.get('junction_aa')
                    if pd.isna(junc):
                        continue
                    junc = str(junc)

                    if junc not in scores:
                        v = str(sr.get('v_call', '')) if pd.notna(sr.get('v_call')) else ''
                        j = str(sr.get('j_call', '')) if pd.notna(sr.get('j_call')) else ''
                        scores[junc] = {'s': 0, 'c': 0, 'v': v if v else None, 'j': j if j else None}

                    scores[junc]['s'] += weight
                    scores[junc]['c'] += 1
            except:
                continue

        # Sort and get top K
        sorted_seqs = sorted(scores.items(), key=lambda x: (x[1]['s'], x[1]['c']), reverse=True)[:TOP_K]

        # Write to file
        batch_rows = []
        for rank, (junc, info) in enumerate(sorted_seqs, 1):
            v = info['v'] if info['v'] else 'TRBV20-1'
            j = info['j'] if info['j'] else 'TRBJ2-7'
            batch_rows.append(f"train_dataset_{ds_id}_seq_top_{rank},train_dataset_{ds_id},-999.0,{junc},{v},{j}\n")
            sequence_count += 1

        with open(output_path, 'a') as f:
            f.writelines(batch_rows)

        print(f"  {len(sorted_seqs)} sequences written")

        # Clear memory
        del scores, sorted_seqs, batch_rows
        gc.collect()

    print(f"\nTask B complete: {sequence_count} sequences written")

    # ========== Verify and Submit ==========
    print("\n" + "="*70)
    print("Verifying submission...")
    print("="*70)

    # Count lines (subtract 1 for header)
    with open(output_path, 'r') as f:
        total_lines = sum(1 for _ in f) - 1

    expected = 4213 + 8 * 50000
    print(f"  Total rows: {total_lines} (expected: {expected})")
    print(f"  Predictions: {prediction_count}")
    print(f"  Sequences: {sequence_count}")
    print(f"  File: {output_path}")
    print(f"  Size: {os.path.getsize(output_path) / 1e6:.2f} MB")

    # Also save as latest
    latest_path = os.path.join(SUBMISSION_DIR, 'submission_latest.csv')
    import shutil
    shutil.copy(output_path, latest_path)
    print(f"  Also saved as: {latest_path}")

    # Submit to Kaggle
    print("\n" + "="*70)
    print("Submitting to Kaggle...")
    print("="*70)

    message = f"ESM2-MLP-Stream CV=0.5307 ({ts})"
    cmd = f'kaggle competitions submit -c adaptive-immune-profiling-challenge-2025 -f "{output_path}" -m "{message}"'
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
        print(f"  Result: {result.stdout}")
        if result.stderr:
            print(f"  Stderr: {result.stderr}")
    except Exception as e:
        print(f"  Submit error: {e}")

    end_time = datetime.now()
    duration = end_time - start_time
    print(f"\n{'='*70}")
    print(f"  COMPLETE!")
    print(f"  Duration: {duration}")
    print(f"  Submission: {output_path}")
    print(f"{'='*70}\n")

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
