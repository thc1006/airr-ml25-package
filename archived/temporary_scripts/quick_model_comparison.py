#!/usr/bin/env python3
"""
Quick Model Comparison Script
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
warnings.filterwarnings('ignore')

# ============================================================================
# Configuration
# ============================================================================
class Config:
    PROJECT_ROOT = Path(__file__).parent
    TRAIN_ROOT = PROJECT_ROOT / 'data/train_datasets/train_datasets'
    OUTPUT_DIR = PROJECT_ROOT / 'model_comparison_results'

    # Model configurations
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
        },
        'CatBoost': {
            'iterations': 500,
            'depth': 6,
            'learning_rate': 0.1,
            'random_state': 42,
            'verbose': False,
        }
    }

    CV_FOLDS = 5
    RANDOM_STATE = 42

Config.OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================================================
# Helper Functions (假設你已經有特徵提取函數)
# ============================================================================

def load_dataset_features(dataset_id):
    """
    載入 dataset 的特徵和標籤
    這裡需要調用你現有的特徵提取代碼

    Returns:
        X: features (n_samples, n_features)
        y: labels (n_samples,)
    """
    # TODO: 替換為你的實際特徵提取代碼
    # 例如：from your_feature_extractor import extract_features

    print(f"  Loading features for dataset {dataset_id}...")

    # 假設你已經有預先提取的特徵文件
    feature_file = Config.OUTPUT_DIR / f'dataset_{dataset_id}_features.npz'

    if feature_file.exists():
        data = np.load(feature_file)
        return data['X'], data['y']
    else:
        raise FileNotFoundError(
            f"Features not found for dataset {dataset_id}. "
            f"Please extract features first using your existing pipeline."
        )

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
    print(f"    Training {model_name}...", end=' ')

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
    print("="*80)

    all_results = {}

    for dataset_id in range(1, 9):
        print(f"\n{'='*60}")
        print(f"Dataset {dataset_id}")
        print(f"{'='*60}")

        try:
            # Load features
            X, y = load_dataset_features(dataset_id)
            print(f"  Features shape: {X.shape}")
            print(f"  Positive ratio: {y.mean():.2%}")

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
                'all_models': dataset_results
            }

        except Exception as e:
            print(f"  ❌ Error: {e}")
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

    for dataset_id in range(1, 9):
        key = f'dataset_{dataset_id}'
        if key in all_results and 'best_model' in all_results[key]:
            result = all_results[key]
            print(f"Dataset {dataset_id}: {result['best_model']:10s} "
                  f"AUC = {result['best_auc']:.4f}")

    print(f"\nResults saved to: {output_file}")

    return all_results

# ============================================================================
# Generate Submission with Best Models
# ============================================================================

def generate_best_submission(results):
    """
    根據比較結果，為每個 dataset 使用最佳模型生成提交檔案
    """
    print(f"\n{'='*80}")
    print("GENERATING SUBMISSION WITH PER-DATASET BEST MODELS")
    print(f"{'='*80}")

    # TODO: 這裡需要你的完整 submission generation pipeline
    # 例如：
    # for dataset_id in range(1, 9):
    #     best_model_name = results[f'dataset_{dataset_id}']['best_model']
    #     train_and_predict(dataset_id, best_model_name)

    print("\n⚠️  TODO: Implement submission generation with your pipeline")
    print("Suggested approach:")
    print("1. For each dataset, load best model from results")
    print("2. Train on full training data")
    print("3. Predict on test data")
    print("4. Combine all predictions into submission.csv")

# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--compare', action='store_true',
                       help='Run model comparison')
    parser.add_argument('--generate', action='store_true',
                       help='Generate submission with best models')
    args = parser.parse_args()

    if args.compare or (not args.compare and not args.generate):
        results = compare_all_models()

    if args.generate:
        # Load results if not already available
        if 'results' not in locals():
            results_file = Config.OUTPUT_DIR / 'model_comparison_results.json'
            with open(results_file) as f:
                results = json.load(f)

        generate_best_submission(results)

    print("\n✅ Done!")
