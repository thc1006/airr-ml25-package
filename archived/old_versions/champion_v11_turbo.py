#!/usr/bin/env python3
"""
Champion V11 Turbo - 全面硬體加速版
- 多線程特徵提取 (8核心 Ryzen 7800X3D)
- GPU加速訓練 (RTX 5080)
- 向量化操作取代迴圈
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Parallel processing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from joblib import Parallel, delayed
import multiprocessing as mp

# ML
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SelectKBest, f_classif
import xgboost as xgb
import lightgbm as lgb

# Configuration
TRAIN_ROOT = Path("/home/thc1006/dev/airr-ml25-package/data/train_datasets/train_datasets")
TEST_ROOT = Path("/home/thc1006/dev/airr-ml25-package/data/test_datasets/test_datasets")
OUTPUT_DIR = Path("/home/thc1006/dev/airr-ml25-package/submissions")
OUTPUT_DIR.mkdir(exist_ok=True)

# Use all available cores
N_JOBS = mp.cpu_count()  # 8 cores
print(f"Using {N_JOBS} CPU cores for parallel processing")

# ============================================================================
# OPTIMIZED FEATURE EXTRACTION (Vectorized + Parallel)
# ============================================================================

def extract_kmers_vectorized(sequences: np.ndarray, k: int = 3) -> Counter:
    """向量化 k-mer 提取"""
    kmer_counts = Counter()
    for seq in sequences:
        if isinstance(seq, str) and len(seq) >= k:
            # Use list comprehension (faster than nested loops)
            kmers = [seq[i:i+k] for i in range(len(seq) - k + 1)]
            kmer_counts.update(kmers)
    return kmer_counts

def extract_features_single(args: Tuple) -> Optional[Dict]:
    """單個樣本的特徵提取 (for parallel execution)"""
    tsv_path, kmer_vocab, vj_vocab, public_clones = args

    try:
        # Read with optimal settings
        df = pd.read_csv(tsv_path, sep='\t', usecols=['junction_aa', 'v_call', 'j_call'])

        if len(df) == 0:
            return None

        features = {}
        sequences = df['junction_aa'].dropna().values
        n_seqs = len(sequences)

        # 1. K-mer features (vectorized)
        kmer_counts = Counter()
        for seq in sequences:
            if isinstance(seq, str) and len(seq) >= 3:
                for i in range(len(seq) - 2):
                    kmer_counts[seq[i:i+3]] += 1

        # Normalize by total
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

        # 4. Diversity metrics (vectorized)
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
            features['cdr3_median'] = np.median(lengths)

        return features

    except Exception as e:
        return None

def build_vocabulary(dataset_path: Path, n_samples: int = 50) -> Tuple[List, List, List]:
    """快速建立詞彙表 (只用部分樣本)"""
    metadata = pd.read_csv(dataset_path / "metadata.csv")
    sample_files = metadata['filename'].head(n_samples).tolist()

    kmer_counter = Counter()
    vj_counter = Counter()
    seq_counter = Counter()

    for fname in sample_files:
        try:
            df = pd.read_csv(dataset_path / fname, sep='\t')

            # K-mers
            for seq in df['junction_aa'].dropna():
                if isinstance(seq, str) and len(seq) >= 3:
                    for i in range(len(seq) - 2):
                        kmer_counter[seq[i:i+3]] += 1

            # VJ pairs
            vj_pairs = list(zip(df['v_call'].fillna('UNK'), df['j_call'].fillna('UNK')))
            vj_counter.update(vj_pairs)

            # Sequences
            seq_counter.update(df['junction_aa'].dropna())

        except:
            pass

    # Top features
    kmer_vocab = [k for k, _ in kmer_counter.most_common(5000)]
    vj_vocab = [k for k, _ in vj_counter.most_common(500)]
    public_clones = [k for k, c in seq_counter.most_common(2500) if c >= 3]

    return kmer_vocab, vj_vocab, public_clones

def extract_features_parallel(dataset_path: Path, metadata: pd.DataFrame,
                             kmer_vocab: List, vj_vocab: List, public_clones: List) -> pd.DataFrame:
    """並行特徵提取 - 使用所有 CPU 核心"""

    # Prepare arguments
    args_list = [
        (dataset_path / row['filename'], kmer_vocab, vj_vocab, public_clones)
        for _, row in metadata.iterrows()
    ]

    # Parallel execution with joblib
    print(f"    Extracting features from {len(args_list)} samples using {N_JOBS} workers...")
    results = Parallel(n_jobs=N_JOBS, verbose=1)(
        delayed(extract_features_single)(args) for args in args_list
    )

    # Filter None results
    valid_results = [r for r in results if r is not None]

    return pd.DataFrame(valid_results)

# ============================================================================
# GPU-ACCELERATED TRAINING
# ============================================================================

def train_gpu_ensemble(X: np.ndarray, y: np.ndarray, dataset_id: int) -> Tuple[object, object, float]:
    """GPU 加速的 ensemble 訓練"""

    # Feature selection (reduce dimensionality for faster training)
    n_features = min(1000, X.shape[1])
    selector = SelectKBest(f_classif, k=n_features)
    X_selected = selector.fit_transform(X, y)

    print(f"    Training on {X_selected.shape[0]} samples, {X_selected.shape[1]} features")

    # XGBoost with CUDA
    xgb_model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        device='cuda',
        tree_method='hist',
        random_state=42,
        eval_metric='auc'
    )

    # LightGBM with GPU
    lgb_model = lgb.LGBMClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        device='gpu',
        random_state=42,
        verbose=-1
    )

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    xgb_scores = []
    lgb_scores = []

    for fold, (train_idx, val_idx) in enumerate(cv.split(X_selected, y)):
        X_train, X_val = X_selected[train_idx], X_selected[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # XGBoost
        xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        xgb_pred = xgb_model.predict_proba(X_val)[:, 1]

        # LightGBM
        lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
        lgb_pred = lgb_model.predict_proba(X_val)[:, 1]

        from sklearn.metrics import roc_auc_score
        xgb_scores.append(roc_auc_score(y_val, xgb_pred))
        lgb_scores.append(roc_auc_score(y_val, lgb_pred))

    xgb_auc = np.mean(xgb_scores)
    lgb_auc = np.mean(lgb_scores)

    print(f"    CV AUC: XGB={xgb_auc:.4f}, LGB={lgb_auc:.4f}")

    # Final training on all data
    xgb_model.fit(X_selected, y)
    lgb_model.fit(X_selected, y)

    return xgb_model, lgb_model, selector, max(xgb_auc, lgb_auc)

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    print("=" * 70)
    print("AIRR-ML-25 Champion V11 TURBO - Full Hardware Acceleration")
    print("=" * 70)
    print(f"CPU Cores: {N_JOBS}")
    print(f"GPU: RTX 5080 (CUDA)")
    print()

    # Check GPU
    print("Checking GPU availability...")
    try:
        import xgboost as xgb
        test_xgb = xgb.XGBClassifier(device='cuda', tree_method='hist')
        test_xgb.fit(np.random.rand(10, 5), np.random.randint(0, 2, 10))
        print("  XGBoost CUDA: OK")
    except Exception as e:
        print(f"  XGBoost CUDA: FAILED - {e}")
        return

    try:
        import lightgbm as lgb
        test_lgb = lgb.LGBMClassifier(device='gpu', verbose=-1)
        test_lgb.fit(np.random.rand(10, 5), np.random.randint(0, 2, 10))
        print("  LightGBM GPU: OK")
    except Exception as e:
        print(f"  LightGBM GPU: FAILED - {e}")

    print()

    # Process all datasets
    models = {}
    all_predictions = []
    all_sequences = []

    datasets = sorted([d for d in TRAIN_ROOT.iterdir() if d.is_dir()])

    for ds_path in datasets:
        ds_name = ds_path.name
        ds_id = int(ds_name.split('_')[-1])

        print(f"\n{'='*60}")
        print(f"Processing {ds_name} (ID={ds_id})")
        print(f"{'='*60}")

        # Load metadata
        metadata = pd.read_csv(ds_path / "metadata.csv")
        y = metadata['label_positive'].astype(int).values

        print(f"  Samples: {len(metadata)}, Positive: {y.sum()}, Negative: {len(y) - y.sum()}")

        # Step 1: Build vocabulary (fast, using subset)
        print("  [1/3] Building vocabulary...")
        kmer_vocab, vj_vocab, public_clones = build_vocabulary(ds_path, n_samples=min(100, len(metadata)))
        print(f"    Vocab: {len(kmer_vocab)} kmers, {len(vj_vocab)} VJ pairs, {len(public_clones)} public clones")

        # Step 2: Parallel feature extraction
        print("  [2/3] Extracting features (parallel)...")
        X_df = extract_features_parallel(ds_path, metadata, kmer_vocab, vj_vocab, public_clones)
        X = X_df.fillna(0).values
        print(f"    Features: {X.shape}")

        # Step 3: GPU training
        print("  [3/3] Training ensemble (GPU)...")
        xgb_model, lgb_model, selector, cv_auc = train_gpu_ensemble(X, y, ds_id)

        models[ds_id] = {
            'xgb': xgb_model,
            'lgb': lgb_model,
            'selector': selector,
            'vocab': (kmer_vocab, vj_vocab, public_clones),
            'cv_auc': cv_auc
        }

        print(f"  Dataset {ds_id} CV AUC: {cv_auc:.4f}")

        # Extract top sequences for Task B
        print("  Extracting top sequences for Task B...")
        feature_names = X_df.columns.tolist()
        importances = xgb_model.feature_importances_

        # Map to sequences
        seq_scores = {}
        for fname, imp in zip(feature_names, importances):
            if fname.startswith('pub_'):
                seq = fname[4:]
                seq_scores[seq] = seq_scores.get(seq, 0) + imp

        # Get top 50000 sequences
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

        # Fill remaining if needed
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

    # ========================================================================
    # PHASE 2: Predictions on test sets
    # ========================================================================
    print(f"\n{'='*70}")
    print("PHASE 2: Generating predictions for test datasets")
    print(f"{'='*70}")

    test_dirs = sorted([d for d in TEST_ROOT.iterdir() if d.is_dir()])

    for test_path in test_dirs:
        test_name = test_path.name

        # Determine which model to use
        if '_' in test_name.replace('test_dataset_', ''):
            base_id = int(test_name.replace('test_dataset_', '').split('_')[0])
        else:
            base_id = int(test_name.replace('test_dataset_', ''))

        if base_id > 8:
            base_id = 8  # Use dataset 8 model for unknown

        print(f"\n  Predicting {test_name} (using model from dataset {base_id})...")

        model_info = models.get(base_id)
        if model_info is None:
            print(f"    WARNING: No model for dataset {base_id}, using dataset 1")
            model_info = models.get(1)

        kmer_vocab, vj_vocab, public_clones = model_info['vocab']

        # Load test metadata
        test_metadata = pd.read_csv(test_path / "metadata.csv")

        # Extract features
        X_test_df = extract_features_parallel(test_path, test_metadata, kmer_vocab, vj_vocab, public_clones)
        X_test = X_test_df.fillna(0).values

        # Ensure same features
        X_test_selected = model_info['selector'].transform(X_test)

        # Predict
        xgb_pred = model_info['xgb'].predict_proba(X_test_selected)[:, 1]
        lgb_pred = model_info['lgb'].predict_proba(X_test_selected)[:, 1]

        # Ensemble average
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

        print(f"    Predicted {len(test_metadata)} samples, mean prob: {final_pred.mean():.3f}")

    # ========================================================================
    # PHASE 3: Generate submission
    # ========================================================================
    print(f"\n{'='*70}")
    print("PHASE 3: Generating submission file")
    print(f"{'='*70}")

    # Combine predictions and sequences
    pred_df = pd.DataFrame(all_predictions)
    seq_df = pd.DataFrame(all_sequences)

    submission = pd.concat([pred_df, seq_df], ignore_index=True)

    # Save
    output_path = OUTPUT_DIR / "submission_v11_turbo.csv"
    submission.to_csv(output_path, index=False)

    print(f"\n  Submission saved: {output_path}")
    print(f"  Total rows: {len(submission):,}")
    print(f"  Predictions: {len(all_predictions):,}")
    print(f"  Sequences: {len(all_sequences):,}")

    # Validate
    sample_sub = pd.read_csv("/home/thc1006/dev/airr-ml25-package/data/sample_submissions.csv")
    print(f"\n  Expected rows: {len(sample_sub):,}")
    print(f"  Match: {len(submission) == len(sample_sub)}")

    print("\n" + "=" * 70)
    print("COMPLETED!")
    print("=" * 70)

if __name__ == "__main__":
    import time
    start = time.time()
    main()
    elapsed = time.time() - start
    print(f"\nTotal time: {elapsed/60:.1f} minutes")
