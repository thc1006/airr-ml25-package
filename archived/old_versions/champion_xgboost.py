#!/usr/bin/env python3
"""
Champion XGBoost Pipeline for AIRR-ML-25 Competition
=====================================================

Simple but effective approach using:
- Traditional features (V/J gene usage, diversity metrics)
- XGBoost with GPU acceleration
- 5-fold stratified cross-validation
- Feature importance for Task B (sequence identification)

Author: Claude Code
Date: 2025-12-15
"""

import os
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


@dataclass
class Config:
    """Configuration for XGBoost pipeline"""
    # Paths
    checkpoint_dir: str = "checkpoints"
    output_dir: str = "checkpoints_xgb"
    cache_dir: str = "cache_xgb"

    # Training
    n_folds: int = 5
    seed: int = 42

    # XGBoost parameters - optimized for GPU
    xgb_params: dict = None

    def __post_init__(self):
        if self.xgb_params is None:
            self.xgb_params = {
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'device': 'cuda',  # GPU acceleration
                'tree_method': 'hist',
                'max_depth': 6,
                'learning_rate': 0.05,
                'n_estimators': 500,
                'min_child_weight': 5,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'random_state': self.seed,
                'verbosity': 0,
                'early_stopping_rounds': 50,
            }


def load_dataset(dataset_id: int, checkpoint_dir: str = "checkpoints") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load dataset from checkpoint"""
    npz_path = Path(checkpoint_dir) / f"dataset_{dataset_id}.npz"
    print(f"Loading dataset {dataset_id} from {npz_path}")

    data = np.load(npz_path, allow_pickle=True)

    # Get traditional features
    trad_raw = data['trad_features']
    if trad_raw.dtype == object:
        trad_features = np.stack([x.astype(np.float32) for x in trad_raw])
    else:
        trad_features = trad_raw.astype(np.float32)

    labels = data['labels']
    rep_ids = data['repertoire_ids']
    ds_ids = data['dataset_ids']

    return trad_features, labels, rep_ids, ds_ids


def train_xgboost_cv(X: np.ndarray, y: np.ndarray, config: Config) -> Tuple[List[xgb.XGBClassifier], float, np.ndarray]:
    """Train XGBoost with cross-validation"""
    skf = StratifiedKFold(n_splits=config.n_folds, shuffle=True, random_state=config.seed)

    models = []
    cv_aucs = []
    oof_preds = np.zeros(len(y))
    feature_importance = np.zeros(X.shape[1])

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n--- Fold {fold}/{config.n_folds} ---")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Create model
        model = xgb.XGBClassifier(**config.xgb_params)

        # Train with early stopping
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        # Predict
        val_preds = model.predict_proba(X_val)[:, 1]
        oof_preds[val_idx] = val_preds

        # Calculate AUC
        fold_auc = roc_auc_score(y_val, val_preds)
        cv_aucs.append(fold_auc)
        print(f"  Fold {fold} AUC: {fold_auc:.4f} (best iter: {model.best_iteration})")

        models.append(model)
        feature_importance += model.feature_importances_

    # Average AUC
    mean_auc = np.mean(cv_aucs)
    std_auc = np.std(cv_aucs)
    print(f"\nCV AUC: {mean_auc:.4f} (+/- {std_auc:.4f})")

    # Normalize feature importance
    feature_importance /= config.n_folds

    return models, mean_auc, feature_importance


def train_final_model(X: np.ndarray, y: np.ndarray, config: Config) -> xgb.XGBClassifier:
    """Train final model on all data"""
    # Remove early stopping for final model
    params = config.xgb_params.copy()
    params.pop('early_stopping_rounds', None)
    params['n_estimators'] = 200  # Fixed number for final model

    model = xgb.XGBClassifier(**params)
    model.fit(X, y, verbose=False)

    return model


class ChampionXGBoost:
    """Main pipeline for XGBoost training"""

    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.models = {}
        self.results = {}
        self.feature_names = None

        # Create directories
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.cache_dir).mkdir(parents=True, exist_ok=True)

        # Load feature names
        feature_path = Path(self.config.checkpoint_dir) / "feature_names.json"
        if feature_path.exists():
            with open(feature_path) as f:
                self.feature_names = json.load(f)

    def train_single_dataset(self, dataset_id: int) -> Dict:
        """Train on a single dataset"""
        print(f"\n{'#'*60}")
        print(f"# Dataset {dataset_id}")
        print(f"{'#'*60}")

        # Load data
        X, y, rep_ids, ds_ids = load_dataset(dataset_id, self.config.checkpoint_dir)
        print(f"Data shape: X={X.shape}, y={y.shape}")
        print(f"Label distribution: {np.bincount(y.astype(int))}")

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Cross-validation
        models, cv_auc, feat_imp = train_xgboost_cv(X_scaled, y, self.config)

        print(f"\n{'='*40}")
        print(f"Dataset {dataset_id} CV AUC: {cv_auc:.4f}")
        print(f"{'='*40}")

        # Train final model
        print("\nTraining final model on all data...")
        final_model = train_final_model(X_scaled, y, self.config)

        # Store results
        result = {
            'dataset_id': dataset_id,
            'cv_auc': cv_auc,
            'cv_models': models,
            'final_model': final_model,
            'scaler': scaler,
            'feature_importance': feat_imp
        }

        self.models[dataset_id] = result
        self.results[dataset_id] = result

        # Save model
        model_path = Path(self.config.output_dir) / f"xgb_ds{dataset_id}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(result, f)
        print(f"Saved model to {model_path}")

        return result

    def train_all_datasets(self) -> Dict:
        """Train on all 8 datasets"""
        all_results = {}

        for ds_id in range(1, 9):
            try:
                result = self.train_single_dataset(ds_id)
                all_results[ds_id] = {
                    'cv_auc': result['cv_auc'],
                }
            except Exception as e:
                print(f"Error training dataset {ds_id}: {e}")
                all_results[ds_id] = {'error': str(e)}

        # Summary
        print("\n" + "="*60)
        print("Training Summary")
        print("="*60)
        aucs = []
        for ds_id, res in all_results.items():
            if 'cv_auc' in res:
                print(f"  Dataset {ds_id}: CV AUC = {res['cv_auc']:.4f}")
                aucs.append(res['cv_auc'])
            else:
                print(f"  Dataset {ds_id}: ERROR - {res.get('error', 'unknown')}")

        if aucs:
            print(f"\nMean CV AUC: {np.mean(aucs):.4f}")

        # Save summary
        summary_path = Path(self.config.output_dir) / "training_results.json"
        with open(summary_path, 'w') as f:
            json.dump([
                {'dataset': f'train_dataset_{ds_id}', 'val_auc': res.get('cv_auc', 0)}
                for ds_id, res in all_results.items()
            ], f, indent=2)

        return all_results

    def get_top_features(self, dataset_id: int, top_k: int = 100) -> pd.DataFrame:
        """Get top important features for a dataset"""
        if dataset_id not in self.results:
            return pd.DataFrame()

        result = self.results[dataset_id]
        feat_imp = result['feature_importance']

        # Get top k features
        top_indices = np.argsort(feat_imp)[::-1][:top_k]

        if self.feature_names:
            df = pd.DataFrame({
                'feature': [self.feature_names[i] for i in top_indices],
                'importance': feat_imp[top_indices]
            })
        else:
            df = pd.DataFrame({
                'feature_idx': top_indices,
                'importance': feat_imp[top_indices]
            })

        return df


def main():
    """Main entry point"""
    print("="*60)
    print("Champion XGBoost Pipeline")
    print("="*60)

    # Check GPU
    try:
        import xgboost as xgb
        # Test GPU availability
        test_data = xgb.DMatrix(np.random.randn(10, 5))
        params = {'device': 'cuda', 'tree_method': 'hist'}
        bst = xgb.train(params, test_data, num_boost_round=1, verbose_eval=False)
        print("GPU: Available (CUDA)")
        del test_data, bst
    except Exception as e:
        print(f"GPU: Not available ({e}), using CPU")

    print("="*60)

    # Initialize
    config = Config()
    pipeline = ChampionXGBoost(config)

    # Train all datasets
    results = pipeline.train_all_datasets()

    return pipeline


if __name__ == "__main__":
    main()
