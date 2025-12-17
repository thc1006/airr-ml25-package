#!/usr/bin/env python3
"""
找出所有 8 個 dataset 的最佳模型配置（GPU 加速版本）
使用多核心並行處理 k-mer 特徵提取 + GPU 訓練
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import warnings
import sys
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial

warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))
from src.features.kmer_features import extract_kmer_features, generate_all_kmers

# ============================================================================
# Configuration
# ============================================================================
class Config:
    PROJECT_ROOT = Path(__file__).parent
    TRAIN_ROOT = PROJECT_ROOT / 'data/train_datasets/train_datasets'
    OUTPUT_DIR = PROJECT_ROOT / 'model_comparison_results'

    # K-mer configuration (k=3 only to prevent overfitting)
    K_RANGE = [3]
    TOP_K_PER_SCALE = 5000  # Limit features to prevent explosion

    # Use ALL CPU cores for parallel processing
    N_JOBS = cpu_count()  # 榨乾所有核心！

    # Model configurations with GPU acceleration
    MODELS = {
        'XGBoost': {
            'n_estimators': 500,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1,
            'eval_metric': 'auc',
            'tree_method': 'hist',  # Use hist for CPU/GPU auto-select
            'device': 'cuda',  # XGBoost 3.1+ uses 'device' instead of 'gpu_id'
        },
        'LightGBM': {
            'n_estimators': 500,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1,
            'device': 'gpu',
        },
        'CatBoost': {
            'iterations': 500,
            'depth': 6,
            'learning_rate': 0.1,
            'random_state': 42,
            'verbose': False,
            'task_type': 'GPU',
            'devices': '0',
        }
    }

    CV_FOLDS = 5
    RANDOM_STATE = 42

Config.OUTPUT_DIR.mkdir(exist_ok=True)

print(f"🚀 Using {Config.N_JOBS} CPU cores for parallel k-mer extraction")
print(f"🎮 Using GPU for XGBoost, LightGBM, CatBoost training")

# ============================================================================
# Parallel K-mer Extraction
# ============================================================================

def extract_kmer_single_repertoire(args):
    """
    單個 repertoire 的 k-mer 特徵提取（用於並行處理）
    """
    repertoire, k_range, kmer_vocabs, top_k_vocabs = args

    features_list = []
    for k in k_range:
        feat = extract_kmer_features(
            repertoire, k,
            sequence_col='junction_aa',
            binary=False,  # Use counts, not binary
            normalize=True,  # Normalize to frequencies
            kmer_vocab=top_k_vocabs[k] if top_k_vocabs else kmer_vocabs[k]
        )
        features_list.append(feat)

    return np.concatenate(features_list)


def extract_features_parallel(repertoires, k_range):
    """
    並行提取所有 repertoires 的 k-mer 特徵
    """
    from collections import Counter

    # Generate k-mer vocabularies
    kmer_vocabs = {k: generate_all_kmers(k) for k in k_range}

    # Determine top k-mers if TOP_K_PER_SCALE is set
    top_k_vocabs = {}
    if Config.TOP_K_PER_SCALE:
        print(f"  Finding top {Config.TOP_K_PER_SCALE} k-mers per scale...")
        for k in k_range:
            kmer_counts = Counter()
            for rep in tqdm(repertoires, desc=f"  Scanning k={k}", leave=False):
                if 'junction_aa' in rep.columns:
                    for seq in rep['junction_aa'].dropna():
                        if isinstance(seq, str) and len(seq) >= k:
                            for i in range(len(seq) - k + 1):
                                kmer = seq[i:i+k]
                                if all(aa in 'ACDEFGHIKLMNPQRSTVWY' for aa in kmer):
                                    kmer_counts[kmer] += 1
            # Get top K
            top_kmers = [kmer for kmer, _ in kmer_counts.most_common(Config.TOP_K_PER_SCALE)]
            top_k_vocabs[k] = top_kmers
            print(f"    k={k}: selected {len(top_kmers)} k-mers")
    else:
        top_k_vocabs = None

    # Prepare arguments for parallel processing
    args_list = [(rep, k_range, kmer_vocabs, top_k_vocabs) for rep in repertoires]

    # Parallel extraction
    print(f"  Extracting k-mer features (k={k_range}) using {Config.N_JOBS} cores...")
    with Pool(Config.N_JOBS) as pool:
        features = list(tqdm(
            pool.imap(extract_kmer_single_repertoire, args_list),
            total=len(repertoires),
            desc="  Parallel k-mer extraction"
        ))

    return np.vstack(features)

# ============================================================================
# Data Loading
# ============================================================================

def load_dataset(dataset_id):
    """
    載入單個 dataset 的 metadata 和所有 repertoires
    """
    dataset_dir = Config.TRAIN_ROOT / f'train_dataset_{dataset_id}'
    metadata_path = dataset_dir / 'metadata.csv'

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")

    # Load metadata
    metadata = pd.read_csv(metadata_path)

    # Load repertoires
    repertoires = []
    labels = []

    print(f"  Loading {len(metadata)} repertoires...")
    for _, row in tqdm(metadata.iterrows(), total=len(metadata),
                       desc=f"  Loading dataset {dataset_id}", leave=False):
        rep_file = dataset_dir / row['filename']
        if rep_file.exists():
            rep_df = pd.read_csv(rep_file, sep='\t')
            repertoires.append(rep_df)
            labels.append(int(row['label_positive']))

    return repertoires, np.array(labels)


def extract_features_for_dataset(dataset_id):
    """
    提取單個 dataset 的 k-mer 特徵（並行加速版）
    """
    print(f"\n{'='*60}")
    print(f"Dataset {dataset_id}")
    print(f"{'='*60}")

    # Load data
    repertoires, labels = load_dataset(dataset_id)

    print(f"  Loaded {len(repertoires)} repertoires")
    print(f"  Positive ratio: {labels.mean():.2%}")

    # Extract k-mer features in parallel
    X = extract_features_parallel(repertoires, Config.K_RANGE)

    print(f"  Features shape: {X.shape}")

    return X, labels

# ============================================================================
# Model Training and Evaluation
# ============================================================================

def train_and_evaluate_model(model_name, model_params, X, y):
    """
    訓練並評估單個模型 (5-fold CV, GPU 加速)
    """
    print(f"    {model_name:12s}...", end=' ', flush=True)

    # Create model
    if model_name == 'XGBoost':
        model = xgb.XGBClassifier(**model_params)
    elif model_name == 'LightGBM':
        model = lgb.LGBMClassifier(**model_params)
    elif model_name == 'CatBoost':
        model = CatBoostClassifier(**model_params)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # 5-Fold CV
    skf = StratifiedKFold(n_splits=Config.CV_FOLDS, shuffle=True,
                          random_state=Config.RANDOM_STATE)

    fold_aucs = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Train on GPU
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict_proba(X_val)[:, 1]

        # Evaluate
        auc = roc_auc_score(y_val, y_pred)
        fold_aucs.append(auc)

    mean_auc = np.mean(fold_aucs)
    std_auc = np.std(fold_aucs)

    print(f"AUC = {mean_auc:.4f} ± {std_auc:.4f}")

    return mean_auc, std_auc, fold_aucs

# ============================================================================
# Main Comparison
# ============================================================================

def compare_all_models():
    """
    比較所有 dataset 上的所有模型
    """
    print("="*80)
    print("🏆 MODEL COMPARISON: XGBoost vs LightGBM vs CatBoost")
    print(f"📊 K-mer range: k={Config.K_RANGE}")
    print(f"⚡ Parallel cores: {Config.N_JOBS}")
    print("="*80)

    all_results = {}

    for dataset_id in range(1, 9):
        try:
            # Extract features (parallel + fast)
            X, y = extract_features_for_dataset(dataset_id)

            # Test all models (GPU accelerated)
            dataset_results = {}

            print(f"\n  Training models on GPU:")
            for model_name, model_params in Config.MODELS.items():
                mean_auc, std_auc, fold_aucs = train_and_evaluate_model(
                    model_name, model_params, X, y
                )

                dataset_results[model_name] = {
                    'mean_auc': float(mean_auc),
                    'std_auc': float(std_auc),
                    'fold_aucs': [float(x) for x in fold_aucs]
                }

            # Find best model
            best_model = max(dataset_results.items(),
                           key=lambda x: x[1]['mean_auc'])

            print(f"\n  ✅ Best: {best_model[0]} (AUC = {best_model[1]['mean_auc']:.4f})")

            all_results[f'dataset_{dataset_id}'] = {
                'best_model': best_model[0],
                'best_auc': best_model[1]['mean_auc'],
                'all_models': dataset_results,
                'n_samples': int(len(y)),
                'n_features': int(X.shape[1]),
                'positive_ratio': float(y.mean())
            }

        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            all_results[f'dataset_{dataset_id}'] = {'error': str(e)}

    # Save results
    output_file = Config.OUTPUT_DIR / 'model_comparison_results.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    # Summary
    print(f"\n{'='*80}")
    print("📊 FINAL SUMMARY")
    print(f"{'='*80}\n")
    print(f"{'Dataset':<10} {'Best Model':<12} {'CV AUC':<10} {'Samples':<10} {'Features':<10}")
    print("-"*80)

    for dataset_id in range(1, 9):
        key = f'dataset_{dataset_id}'
        if key in all_results and 'best_model' in all_results[key]:
            result = all_results[key]
            print(f"Dataset {dataset_id:<3} {result['best_model']:<12} "
                  f"{result['best_auc']:<10.4f} "
                  f"{result['n_samples']:<10} "
                  f"{result['n_features']:<10}")

    # Predictions
    aucs = [all_results[f'dataset_{i}']['best_auc']
            for i in range(1, 9)
            if f'dataset_{i}' in all_results and 'best_auc' in all_results[f'dataset_{i}']]

    if aucs:
        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)
        min_auc = np.min(aucs)
        max_auc = np.max(aucs)

        print(f"\n{'='*80}")
        print(f"📈 Average CV AUC: {mean_auc:.4f} ± {std_auc:.4f}")
        print(f"📊 Range: {min_auc:.4f} - {max_auc:.4f}")
        print(f"🎯 Estimated LB Score: {mean_auc:.4f}")
        print(f"🏆 Current LB: 0.67887 → Target: 0.81364")

        if mean_auc > 0.67887:
            improvement = (mean_auc - 0.67887) / 0.67887 * 100
            print(f"✅ Expected improvement: +{improvement:.1f}%")

        print(f"{'='*80}")

    print(f"\n💾 Results saved to: {output_file}")

    return all_results

# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    results = compare_all_models()
    print("\n✅ 完成！所有 8 個 dataset 的最佳模型配置已找出！")
