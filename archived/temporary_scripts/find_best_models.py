#!/usr/bin/env python3
"""
找出所有 8 個 dataset 的最佳模型配置
快速比較 XGBoost, LightGBM, CatBoost 在所有 8 個 dataset 上的表現
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

warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))
from src.features.kmer_features import MultiScaleKmerFeaturizer

# ============================================================================
# Configuration
# ============================================================================
class Config:
    PROJECT_ROOT = Path(__file__).parent
    TRAIN_ROOT = PROJECT_ROOT / 'data/train_datasets/train_datasets'
    OUTPUT_DIR = PROJECT_ROOT / 'model_comparison_results'

    # K-mer configuration (proven to work: k=3-5)
    K_MIN = 3
    K_MAX = 5
    TOP_K_PER_SCALE = None  # Use all k-mers for now (can limit later if needed)

    # Model configurations (proven hyperparameters)
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
            'tree_method': 'gpu_hist',  # GPU acceleration
            'gpu_id': 0,
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
            'device': 'gpu',  # GPU acceleration
            'gpu_platform_id': 0,
            'gpu_device_id': 0,
        },
        'CatBoost': {
            'iterations': 500,
            'depth': 6,
            'learning_rate': 0.1,
            'random_state': 42,
            'verbose': False,
            'task_type': 'GPU',  # GPU acceleration
            'devices': '0',
        }
    }

    CV_FOLDS = 5
    RANDOM_STATE = 42

Config.OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================================================
# Data Loading
# ============================================================================

def load_dataset(dataset_id):
    """
    載入單個 dataset 的 metadata 和所有 repertoires

    Returns:
        repertoires: List[pd.DataFrame] - 所有 repertoire 的序列
        labels: np.ndarray - 標籤 (0/1)
        repertoire_ids: List[str] - repertoire IDs
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
    repertoire_ids = []

    for _, row in tqdm(metadata.iterrows(), total=len(metadata),
                       desc=f"  Loading dataset {dataset_id}", leave=False):
        rep_file = dataset_dir / row['filename']
        if rep_file.exists():
            rep_df = pd.read_csv(rep_file, sep='\t')
            repertoires.append(rep_df)
            labels.append(int(row['label_positive']))
            repertoire_ids.append(row['repertoire_id'])

    return repertoires, np.array(labels), repertoire_ids


def extract_features_for_dataset(dataset_id):
    """
    提取單個 dataset 的 k-mer 特徵

    Returns:
        X: features (n_samples, n_features)
        y: labels (n_samples,)
    """
    print(f"  Loading and extracting features for dataset {dataset_id}...")

    # Load data
    repertoires, labels, _ = load_dataset(dataset_id)

    print(f"    Loaded {len(repertoires)} repertoires")
    print(f"    Positive ratio: {labels.mean():.2%}")

    # Extract k-mer features
    featurizer = MultiScaleKmerFeaturizer(
        k_range=(Config.K_MIN, Config.K_MAX),
        top_k_per_scale=Config.TOP_K_PER_SCALE
    )

    featurizer.fit(repertoires)
    X = featurizer.transform_many(repertoires, show_progress=True)

    print(f"    Features shape: {X.shape}")

    return X, labels

# ============================================================================
# Model Training and Evaluation
# ============================================================================

def train_and_evaluate_model(model_name, model_params, X, y):
    """
    訓練並評估單個模型 (5-fold CV)

    Returns:
        mean_auc: 平均 CV AUC
        std_auc: 標準差
        fold_aucs: 每個 fold 的 AUC
    """
    print(f"    Training {model_name:10s}...", end=' ', flush=True)

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

        # Train
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
    print("MODEL COMPARISON: XGBoost vs LightGBM vs CatBoost")
    print(f"K-mer range: k={Config.K_MIN}-{Config.K_MAX}")
    print("="*80)

    all_results = {}

    for dataset_id in range(1, 9):
        print(f"\n{'='*60}")
        print(f"Dataset {dataset_id}")
        print(f"{'='*60}")

        try:
            # Extract features
            X, y = extract_features_for_dataset(dataset_id)

            # Test all models
            dataset_results = {}

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

            print(f"\n  ✅ Best Model: {best_model[0]} "
                  f"(AUC = {best_model[1]['mean_auc']:.4f})")

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
            all_results[f'dataset_{dataset_id}'] = {
                'error': str(e)
            }

    # Save results
    output_file = Config.OUTPUT_DIR / 'model_comparison_results.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")

    # Summary table
    print(f"{'Dataset':<10} {'Best Model':<12} {'AUC':<10} {'n_samples':<12} {'n_features':<12}")
    print("-"*80)

    for dataset_id in range(1, 9):
        key = f'dataset_{dataset_id}'
        if key in all_results and 'best_model' in all_results[key]:
            result = all_results[key]
            print(f"Dataset {dataset_id:<3} {result['best_model']:<12} "
                  f"{result['best_auc']:<10.4f} "
                  f"{result['n_samples']:<12} "
                  f"{result['n_features']:<12}")

    print(f"\nResults saved to: {output_file}")

    # Calculate average improvement prediction
    aucs = [all_results[f'dataset_{i}']['best_auc']
            for i in range(1, 9)
            if f'dataset_{i}' in all_results and 'best_auc' in all_results[f'dataset_{i}']]

    if aucs:
        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)
        print(f"\n📊 Average Best AUC: {mean_auc:.4f} ± {std_auc:.4f}")
        print(f"📈 Expected LB Score: {mean_auc:.4f} (if CV matches LB)")

    return all_results

# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--k-min', type=int, default=3,
                       help='Minimum k-mer size')
    parser.add_argument('--k-max', type=int, default=5,
                       help='Maximum k-mer size')
    parser.add_argument('--top-k', type=int, default=None,
                       help='Top k-mers per scale (None = all)')
    parser.add_argument('--datasets', type=str, default='1,2,3,4,5,6,7,8',
                       help='Comma-separated dataset IDs to test')
    args = parser.parse_args()

    # Update config
    Config.K_MIN = args.k_min
    Config.K_MAX = args.k_max
    Config.TOP_K_PER_SCALE = args.top_k

    print(f"Configuration:")
    print(f"  K-mer range: k={Config.K_MIN}-{Config.K_MAX}")
    print(f"  Top k-mers per scale: {Config.TOP_K_PER_SCALE}")
    print(f"  GPU acceleration: ENABLED")
    print()

    results = compare_all_models()

    print("\n✅ 完成！所有 8 個 dataset 的最佳模型配置已找出！")
