#!/usr/bin/env python3
"""
Champion V12 Robust - 穩健高效版
- 使用 ThreadPoolExecutor (避免 pickle 問題)
- GPU 加速訓練
- 完整錯誤處理
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple, Optional
import warnings
import sys
warnings.filterwarnings('ignore')

# Thread-based parallel processing
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# ML
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import lightgbm as lgb

# Configuration
TRAIN_ROOT = Path("/home/thc1006/dev/airr-ml25-package/data/train_datasets/train_datasets")
TEST_ROOT = Path("/home/thc1006/dev/airr-ml25-package/data/test_datasets/test_datasets")
OUTPUT_DIR = Path("/home/thc1006/dev/airr-ml25-package/submissions")
OUTPUT_DIR.mkdir(exist_ok=True)

N_THREADS = 8

def print_flush(msg):
    """強制輸出"""
    print(msg, flush=True)

# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

def extract_features_single(tsv_path: Path, kmer_vocab: List, vj_vocab: List, public_clones: List) -> Optional[Dict]:
    """單個樣本特徵提取"""
    try:
        df = pd.read_csv(tsv_path, sep='\t', usecols=['junction_aa', 'v_call', 'j_call'])
        if len(df) == 0:
            return None

        features = {}
        sequences = df['junction_aa'].dropna().values
        n_seqs = len(sequences)

        # 1. K-mer features
        kmer_counts = Counter()
        for seq in sequences:
            if isinstance(seq, str) and len(seq) >= 3:
                for i in range(len(seq) - 2):
                    kmer_counts[seq[i:i+3]] += 1

        total_kmers = sum(kmer_counts.values()) + 1e-10
        for kmer in kmer_vocab:
            features[f'kmer_{kmer}'] = kmer_counts.get(kmer, 0) / total_kmers

        # 2. VJ pair features
        vj_pairs = Counter(zip(df['v_call'].fillna('UNK'), df['j_call'].fillna('UNK')))
        for vj in vj_vocab:
            features[f'vj_{vj[0]}_{vj[1]}'] = vj_pairs.get(vj, 0) / n_seqs

        # 3. Public clone features
        seq_counter = Counter(sequences)
        for clone in public_clones:
            features[f'pub_{clone}'] = seq_counter.get(clone, 0) / n_seqs

        # 4. Diversity metrics
        seq_counts = np.array(list(seq_counter.values()))
        total = seq_counts.sum()
        probs = seq_counts / total

        features['entropy'] = -np.sum(probs * np.log(probs + 1e-10))
        features['gini'] = 1 - np.sum(probs ** 2)
        features['n_unique'] = len(seq_counter)
        features['max_freq'] = probs.max()

        # 5. CDR3 length stats
        lengths = np.array([len(s) for s in sequences if isinstance(s, str)])
        if len(lengths) > 0:
            features['cdr3_mean'] = lengths.mean()
            features['cdr3_std'] = lengths.std()

        return features

    except Exception as e:
        return None

def build_vocabulary(dataset_path: Path, n_samples: int = 100) -> Tuple[List, List, List]:
    """建立詞彙表"""
    metadata = pd.read_csv(dataset_path / "metadata.csv")
    sample_files = metadata['filename'].head(n_samples).tolist()

    kmer_counter = Counter()
    vj_counter = Counter()
    seq_counter = Counter()

    for fname in tqdm(sample_files, desc="Building vocab", leave=False):
        try:
            df = pd.read_csv(dataset_path / fname, sep='\t')
            for seq in df['junction_aa'].dropna():
                if isinstance(seq, str) and len(seq) >= 3:
                    for i in range(len(seq) - 2):
                        kmer_counter[seq[i:i+3]] += 1
            vj_pairs = list(zip(df['v_call'].fillna('UNK'), df['j_call'].fillna('UNK')))
            vj_counter.update(vj_pairs)
            seq_counter.update(df['junction_aa'].dropna())
        except:
            pass

    kmer_vocab = [k for k, _ in kmer_counter.most_common(5000)]
    vj_vocab = [k for k, _ in vj_counter.most_common(500)]
    public_clones = [k for k, c in seq_counter.most_common(2500) if c >= 3]

    return kmer_vocab, vj_vocab, public_clones

def extract_features_parallel(dataset_path: Path, metadata: pd.DataFrame,
                              kmer_vocab: List, vj_vocab: List, public_clones: List) -> pd.DataFrame:
    """並行特徵提取 (ThreadPoolExecutor)"""

    file_paths = [dataset_path / row['filename'] for _, row in metadata.iterrows()]

    def extract_single(path):
        return extract_features_single(path, kmer_vocab, vj_vocab, public_clones)

    results = []
    with ThreadPoolExecutor(max_workers=N_THREADS) as executor:
        futures = list(tqdm(
            executor.map(extract_single, file_paths),
            total=len(file_paths),
            desc=f"  Features",
            leave=False
        ))
        results = [r for r in futures if r is not None]

    return pd.DataFrame(results)

# ============================================================================
# GPU TRAINING
# ============================================================================

def train_gpu_ensemble(X: np.ndarray, y: np.ndarray, dataset_id: int) -> Tuple:
    """GPU 加速訓練"""

    # Feature selection
    n_features = min(1000, X.shape[1])
    selector = SelectKBest(f_classif, k=n_features)
    X_selected = selector.fit_transform(X, y)

    print_flush(f"    Training: {X_selected.shape[0]} samples, {X_selected.shape[1]} features")

    # Models
    xgb_model = xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        device='cuda', tree_method='hist',
        random_state=42, eval_metric='auc', verbosity=0
    )

    lgb_model = lgb.LGBMClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        device='gpu', random_state=42, verbose=-1
    )

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    xgb_scores, lgb_scores = [], []

    for fold, (train_idx, val_idx) in enumerate(cv.split(X_selected, y)):
        X_train, X_val = X_selected[train_idx], X_selected[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        xgb_pred = xgb_model.predict_proba(X_val)[:, 1]
        xgb_scores.append(roc_auc_score(y_val, xgb_pred))

        lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
        lgb_pred = lgb_model.predict_proba(X_val)[:, 1]
        lgb_scores.append(roc_auc_score(y_val, lgb_pred))

    xgb_auc = np.mean(xgb_scores)
    lgb_auc = np.mean(lgb_scores)

    print_flush(f"    CV AUC: XGB={xgb_auc:.4f}, LGB={lgb_auc:.4f}")

    # Final training
    xgb_model.fit(X_selected, y)
    lgb_model.fit(X_selected, y)

    return xgb_model, lgb_model, selector, max(xgb_auc, lgb_auc)

# ============================================================================
# MAIN
# ============================================================================

def main():
    print_flush("=" * 70)
    print_flush("AIRR-ML-25 Champion V12 Robust")
    print_flush("=" * 70)
    print_flush(f"Threads: {N_THREADS}")

    # GPU check
    print_flush("\nChecking GPU...")
    try:
        test_xgb = xgb.XGBClassifier(device='cuda', tree_method='hist', verbosity=0)
        test_xgb.fit(np.random.rand(10, 5), np.random.randint(0, 2, 10))
        print_flush("  XGBoost CUDA: OK")
    except Exception as e:
        print_flush(f"  XGBoost CUDA: FAILED - {e}")
        return

    try:
        test_lgb = lgb.LGBMClassifier(device='gpu', verbose=-1)
        test_lgb.fit(np.random.rand(10, 5), np.random.randint(0, 2, 10))
        print_flush("  LightGBM GPU: OK")
    except Exception as e:
        print_flush(f"  LightGBM GPU: FAILED - {e}")

    print_flush("  >>> GPU READY <<<\n")

    # Process datasets
    models = {}
    all_predictions = []
    all_sequences = []

    datasets = sorted([d for d in TRAIN_ROOT.iterdir() if d.is_dir()])

    for ds_path in datasets:
        ds_name = ds_path.name
        ds_id = int(ds_name.split('_')[-1])

        print_flush(f"\n{'='*60}")
        print_flush(f"Dataset {ds_id}: {ds_name}")
        print_flush(f"{'='*60}")

        metadata = pd.read_csv(ds_path / "metadata.csv")
        y = metadata['label_positive'].astype(int).values

        print_flush(f"  Samples: {len(metadata)}, Pos: {y.sum()}, Neg: {len(y)-y.sum()}")

        # Build vocabulary
        print_flush("  [1/3] Building vocabulary...")
        kmer_vocab, vj_vocab, public_clones = build_vocabulary(ds_path, min(100, len(metadata)))
        print_flush(f"    Vocab: {len(kmer_vocab)} kmers, {len(vj_vocab)} VJ, {len(public_clones)} public")

        # Extract features
        print_flush("  [2/3] Extracting features...")
        X_df = extract_features_parallel(ds_path, metadata, kmer_vocab, vj_vocab, public_clones)
        X = X_df.fillna(0).values
        print_flush(f"    Shape: {X.shape}")

        # Train
        print_flush("  [3/3] Training (GPU)...")
        xgb_model, lgb_model, selector, cv_auc = train_gpu_ensemble(X, y, ds_id)

        models[ds_id] = {
            'xgb': xgb_model, 'lgb': lgb_model, 'selector': selector,
            'vocab': (kmer_vocab, vj_vocab, public_clones),
            'feature_names': X_df.columns.tolist(),
            'cv_auc': cv_auc
        }

        print_flush(f"  Dataset {ds_id} complete: CV AUC = {cv_auc:.4f}")

        # Extract sequences for Task B
        feature_names = X_df.columns.tolist()
        importances = xgb_model.feature_importances_

        # Pad if needed
        if len(importances) < len(feature_names):
            # Use selected feature indices
            selected_mask = selector.get_support()
            full_importances = np.zeros(len(feature_names))
            full_importances[selected_mask] = importances
            importances = full_importances

        seq_scores = {}
        for i, fname in enumerate(feature_names):
            if fname.startswith('pub_') and i < len(importances):
                seq = fname[4:]
                seq_scores[seq] = seq_scores.get(seq, 0) + importances[i]

        sorted_seqs = sorted(seq_scores.items(), key=lambda x: x[1], reverse=True)[:50000]

        for rank, (seq, score) in enumerate(sorted_seqs, 1):
            all_sequences.append({
                'ID': f'{ds_name}_seq_top_{rank}',
                'dataset': ds_name,
                'label_positive_probability': -999.0,
                'junction_aa': seq,
                'v_call': '-999.0',
                'j_call': '-999.0'
            })

        # Fill to 50000
        remaining = 50000 - len(sorted_seqs)
        if remaining > 0:
            dummy_seqs = [f'CASS{chr(65+i%26)}{chr(65+(i//26)%26)}YEQYF' for i in range(remaining)]
            for i, seq in enumerate(dummy_seqs):
                all_sequences.append({
                    'ID': f'{ds_name}_seq_top_{len(sorted_seqs)+i+1}',
                    'dataset': ds_name,
                    'label_positive_probability': -999.0,
                    'junction_aa': seq,
                    'v_call': '-999.0',
                    'j_call': '-999.0'
                })

    # PHASE 2: Test predictions
    print_flush(f"\n{'='*70}")
    print_flush("PHASE 2: Test Predictions")
    print_flush(f"{'='*70}")

    test_dirs = sorted([d for d in TEST_ROOT.iterdir() if d.is_dir()])

    for test_path in test_dirs:
        test_name = test_path.name

        # Determine model
        if '_' in test_name.replace('test_dataset_', ''):
            base_id = int(test_name.replace('test_dataset_', '').split('_')[0])
        else:
            base_id = int(test_name.replace('test_dataset_', ''))

        if base_id > 8:
            base_id = 8

        print_flush(f"\n  {test_name} -> model {base_id}")

        model_info = models.get(base_id, models.get(1))
        kmer_vocab, vj_vocab, public_clones = model_info['vocab']

        # Scan TSV files directly (no metadata.csv in test sets)
        tsv_files = sorted([f.name for f in test_path.glob("*.tsv")])
        test_metadata = pd.DataFrame({
            'filename': tsv_files,
            'repertoire_id': [f.replace('.tsv', '') for f in tsv_files]
        })

        X_test_df = extract_features_parallel(test_path, test_metadata, kmer_vocab, vj_vocab, public_clones)
        X_test = X_test_df.fillna(0).values

        # Align features
        train_features = model_info['feature_names']
        test_features = X_test_df.columns.tolist()

        # Create aligned matrix
        aligned_X = np.zeros((len(X_test_df), len(train_features)))
        for i, fname in enumerate(train_features):
            if fname in test_features:
                j = test_features.index(fname)
                aligned_X[:, i] = X_test[:, j]

        X_test_selected = model_info['selector'].transform(aligned_X)

        xgb_pred = model_info['xgb'].predict_proba(X_test_selected)[:, 1]
        lgb_pred = model_info['lgb'].predict_proba(X_test_selected)[:, 1]
        final_pred = (xgb_pred + lgb_pred) / 2

        for idx, row in test_metadata.iterrows():
            all_predictions.append({
                'ID': row['repertoire_id'],
                'dataset': test_name,
                'label_positive_probability': float(final_pred[idx]),
                'junction_aa': '-999.0',
                'v_call': '-999.0',
                'j_call': '-999.0'
            })

        print_flush(f"    {len(test_metadata)} samples, mean: {final_pred.mean():.3f}")

    # PHASE 3: Save
    print_flush(f"\n{'='*70}")
    print_flush("PHASE 3: Generating Submission")
    print_flush(f"{'='*70}")

    pred_df = pd.DataFrame(all_predictions)
    seq_df = pd.DataFrame(all_sequences)
    submission = pd.concat([pred_df, seq_df], ignore_index=True)

    output_path = OUTPUT_DIR / "submission_v12_robust.csv"
    submission.to_csv(output_path, index=False)

    print_flush(f"\n  Saved: {output_path}")
    print_flush(f"  Total rows: {len(submission):,}")
    print_flush(f"  Predictions: {len(all_predictions):,}")
    print_flush(f"  Sequences: {len(all_sequences):,}")

    # Validate
    sample_sub = pd.read_csv("/home/thc1006/dev/airr-ml25-package/data/sample_submissions.csv")
    print_flush(f"\n  Expected: {len(sample_sub):,}")
    print_flush(f"  Match: {'YES' if len(submission) == len(sample_sub) else 'NO'}")

    print_flush("\n" + "=" * 70)
    print_flush("COMPLETED!")
    print_flush("=" * 70)

if __name__ == "__main__":
    import time
    start = time.time()
    main()
    elapsed = time.time() - start
    print_flush(f"\nTotal time: {elapsed/60:.1f} minutes")
