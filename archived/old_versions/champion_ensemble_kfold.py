#!/usr/bin/env python3
"""
AIRR-ML-25 Championship Ensemble with K-Fold Cross-Validation
=============================================================
Ultimate solution combining:
- Multi-model stacking ensemble (XGBoost + LightGBM + CatBoost)
- Stratified 5-Fold cross-validation
- GPU acceleration for all models
- Per-dataset optimization
- Proper Task B sequence identification

Target: Score > 0.81364 (beat GROZD)
"""

import os
import gc
import json
import pickle
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import numpy as np
import pandas as pd
from scipy.stats import entropy
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, LogisticRegression
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

# GPU-accelerated gradient boosting
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

warnings.filterwarnings('ignore')

# ============================================================================
# Configuration
# ============================================================================
TRAIN_ROOT = Path('./data/train_datasets/train_datasets')
TEST_ROOT = Path('./data/test_datasets/test_datasets')
SUBMISSION_DIR = Path('./submissions')
CHECKPOINT_DIR = Path('./checkpoints_ensemble')

# Feature configuration
K_MER_SIZES = [3, 4, 5]  # Multi-scale k-mers
MAX_KMERS_PER_K = {3: 2000, 4: 3000, 5: 2000}  # Limit per k-mer size
TOP_K = 50000

# Cross-validation
N_FOLDS = 5
RANDOM_STATE = 42

# Dataset mapping
TEST_TO_TRAIN = {
    'test_dataset_1': 1, 'test_dataset_2': 2, 'test_dataset_3': 3,
    'test_dataset_4': 4, 'test_dataset_5': 5, 'test_dataset_6': 6,
    'test_dataset_7_1': 7, 'test_dataset_7_2': 7,
    'test_dataset_8_1': 8, 'test_dataset_8_2': 8, 'test_dataset_8_3': 8,
}

# ============================================================================
# Feature Extraction (Enhanced)
# ============================================================================
class FeatureExtractor:
    """Enhanced feature extraction with multi-scale k-mers and diversity metrics."""

    def __init__(self, k_mer_sizes: List[int] = [3, 4, 5]):
        self.k_mer_sizes = k_mer_sizes
        self.feature_names: List[str] = []
        self.kmer_vocab: Dict[str, Set[str]] = {}
        self.v_genes: Set[str] = set()
        self.j_genes: Set[str] = set()
        self.is_fitted = False

    def fit(self, train_paths: List[Path], metadata_list: List[pd.DataFrame]) -> 'FeatureExtractor':
        """Learn feature vocabulary from training data."""
        print("\n[FeatureExtractor] Building vocabulary...")

        kmer_counters = {k: Counter() for k in self.k_mer_sizes}
        v_counter = Counter()
        j_counter = Counter()

        # Sample files for vocabulary building
        for train_path, metadata in zip(train_paths, metadata_list):
            sample_files = metadata['filename'].tolist()[:50]  # Sample for speed

            for filename in tqdm(sample_files, desc=f"  Scanning {train_path.name}", leave=False):
                tsv_path = train_path / filename
                if not tsv_path.exists():
                    continue

                try:
                    df = pd.read_csv(tsv_path, sep='\t', usecols=['junction_aa', 'v_call', 'j_call'])

                    # K-mers
                    for seq in df['junction_aa'].dropna().astype(str):
                        if len(seq) < 8 or not seq.isalpha():
                            continue
                        for k in self.k_mer_sizes:
                            for i in range(len(seq) - k + 1):
                                kmer = seq[i:i+k]
                                if kmer.isalpha():
                                    kmer_counters[k][kmer] += 1

                    # V genes
                    for v in df['v_call'].dropna().astype(str):
                        v_base = v.split('*')[0].split(',')[0].strip()
                        if v_base:
                            v_counter[v_base] += 1

                    # J genes
                    for j in df['j_call'].dropna().astype(str):
                        j_base = j.split('*')[0].split(',')[0].strip()
                        if j_base:
                            j_counter[j_base] += 1

                except Exception:
                    continue

        # Build vocabulary
        self.feature_names = []

        # Multi-scale k-mers (limited per k)
        for k in self.k_mer_sizes:
            top_kmers = [kmer for kmer, _ in kmer_counters[k].most_common(MAX_KMERS_PER_K[k])]
            self.kmer_vocab[k] = set(top_kmers)
            for kmer in top_kmers:
                self.feature_names.append(f'kmer_{k}_{kmer}')

        # V genes (top 200)
        top_v = [v for v, _ in v_counter.most_common(200)]
        self.v_genes = set(top_v)
        for v in top_v:
            self.feature_names.append(f'v_{v}')

        # J genes (top 50)
        top_j = [j for j, _ in j_counter.most_common(50)]
        self.j_genes = set(top_j)
        for j in top_j:
            self.feature_names.append(f'j_{j}')

        # Diversity features
        diversity_features = [
            'shannon_entropy', 'gini_simpson', 'clonality', 'd50',
            'richness', 'top_clone_freq', 'top10_clone_freq',
            'cdr3_len_mean', 'cdr3_len_std', 'cdr3_len_median',
            'cdr3_len_skew', 'unique_ratio'
        ]
        self.feature_names.extend(diversity_features)

        self.is_fitted = True
        print(f"  Vocabulary: {len(self.feature_names)} features")
        print(f"    K-mers: {sum(len(self.kmer_vocab[k]) for k in self.k_mer_sizes)}")
        print(f"    V genes: {len(self.v_genes)}, J genes: {len(self.j_genes)}")

        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Extract features from a repertoire DataFrame."""
        if not self.is_fitted:
            raise RuntimeError("FeatureExtractor must be fitted first")

        features = {}
        total = len(df)
        if total == 0:
            return np.zeros(len(self.feature_names), dtype=np.float32)

        # K-mer frequencies (multi-scale)
        if 'junction_aa' in df.columns:
            for k in self.k_mer_sizes:
                kmer_counts = Counter()
                for seq in df['junction_aa'].dropna().astype(str):
                    if len(seq) < k or not seq.isalpha():
                        continue
                    for i in range(len(seq) - k + 1):
                        kmer = seq[i:i+k]
                        if kmer in self.kmer_vocab[k]:
                            kmer_counts[kmer] += 1

                total_kmers = sum(kmer_counts.values()) or 1
                for kmer, count in kmer_counts.items():
                    features[f'kmer_{k}_{kmer}'] = count / total_kmers

        # V gene usage
        if 'v_call' in df.columns:
            v_counts = df['v_call'].dropna().astype(str).apply(lambda x: x.split('*')[0].split(',')[0].strip()).value_counts()
            v_total = v_counts.sum() or 1
            for gene, count in v_counts.items():
                if gene in self.v_genes:
                    features[f'v_{gene}'] = count / v_total

        # J gene usage
        if 'j_call' in df.columns:
            j_counts = df['j_call'].dropna().astype(str).apply(lambda x: x.split('*')[0].split(',')[0].strip()).value_counts()
            j_total = j_counts.sum() or 1
            for gene, count in j_counts.items():
                if gene in self.j_genes:
                    features[f'j_{gene}'] = count / j_total

        # Diversity metrics
        if 'junction_aa' in df.columns:
            seqs = df['junction_aa'].dropna()
            if len(seqs) > 0:
                counts = seqs.value_counts()
                freqs = counts.values / counts.sum()

                # Shannon entropy
                shan = entropy(freqs)
                features['shannon_entropy'] = shan

                # Gini-Simpson
                features['gini_simpson'] = 1 - np.sum(freqs ** 2)

                # Clonality
                max_ent = np.log(len(counts)) if len(counts) > 1 else 1
                features['clonality'] = 1 - (shan / max_ent) if max_ent > 0 else 0

                # D50
                cumsum = np.cumsum(np.sort(freqs)[::-1])
                d50_idx = np.searchsorted(cumsum, 0.5) + 1
                features['d50'] = d50_idx / len(freqs)

                # Richness and top clones
                features['richness'] = len(counts) / total
                features['top_clone_freq'] = freqs[0] if len(freqs) > 0 else 0
                features['top10_clone_freq'] = freqs[:10].sum() if len(freqs) >= 10 else freqs.sum()
                features['unique_ratio'] = len(counts) / len(seqs)

                # CDR3 length stats
                lengths = seqs.str.len()
                features['cdr3_len_mean'] = lengths.mean()
                features['cdr3_len_std'] = lengths.std() if len(lengths) > 1 else 0
                features['cdr3_len_median'] = lengths.median()
                features['cdr3_len_skew'] = lengths.skew() if len(lengths) > 2 else 0

        # Build feature vector
        result = np.zeros(len(self.feature_names), dtype=np.float32)
        for i, name in enumerate(self.feature_names):
            if name in features:
                v = features[name]
                if pd.notna(v) and not np.isinf(v):
                    result[i] = float(v)

        return result


# ============================================================================
# Stacking Ensemble with K-Fold CV
# ============================================================================
class StackingEnsemble:
    """
    GPU-accelerated stacking ensemble with:
    - XGBoost (GPU)
    - LightGBM (GPU)
    - CatBoost (GPU)
    - Ridge meta-learner
    """

    def __init__(self, n_folds: int = 5, random_state: int = 42):
        self.n_folds = n_folds
        self.random_state = random_state
        self.base_models = []
        self.meta_model = None
        self.scaler = StandardScaler()
        self.cv_scores = []
        self.is_fitted = False

    def _create_base_models(self) -> List:
        """Create GPU-accelerated base models."""
        return [
            # XGBoost (GPU)
            xgb.XGBClassifier(
                objective='binary:logistic',
                eval_metric='auc',
                tree_method='hist',
                device='cuda',
                max_depth=6,
                learning_rate=0.05,
                n_estimators=300,
                min_child_weight=3,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=self.random_state,
                verbosity=0,
            ),
            # LightGBM (GPU)
            lgb.LGBMClassifier(
                objective='binary',
                metric='auc',
                device='gpu',
                gpu_platform_id=0,
                gpu_device_id=0,
                num_leaves=63,
                learning_rate=0.05,
                n_estimators=300,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=self.random_state,
                verbose=-1,
            ),
            # CatBoost (GPU)
            cb.CatBoostClassifier(
                loss_function='Logloss',
                eval_metric='AUC',
                task_type='GPU',
                devices='0',
                depth=6,
                learning_rate=0.05,
                iterations=300,
                l2_leaf_reg=3,
                random_state=self.random_state,
                verbose=False,
            ),
        ]

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'StackingEnsemble':
        """Train ensemble with k-fold CV stacking."""
        print(f"\n[StackingEnsemble] Training with {self.n_folds}-fold CV...")

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Initialize
        n_samples = len(X)
        n_models = 3  # XGB, LGB, CatBoost
        meta_features = np.zeros((n_samples, n_models))

        kfold = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)

        fold_aucs = []

        for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X_scaled, y)):
            print(f"  Fold {fold_idx + 1}/{self.n_folds}...")

            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            fold_models = self._create_base_models()
            fold_preds = []

            for model_idx, model in enumerate(fold_models):
                model.fit(X_train, y_train)
                val_pred = model.predict_proba(X_val)[:, 1]
                meta_features[val_idx, model_idx] = val_pred
                fold_preds.append(val_pred)

            # Ensemble prediction for this fold (simple average)
            ensemble_pred = np.mean(fold_preds, axis=0)
            fold_auc = roc_auc_score(y_val, ensemble_pred)
            fold_aucs.append(fold_auc)
            print(f"    Fold {fold_idx + 1} AUC: {fold_auc:.4f}")

        # Train meta-learner
        self.meta_model = RidgeCV(alphas=[0.001, 0.01, 0.1, 1.0, 10.0])
        self.meta_model.fit(meta_features, y)

        # Retrain base models on full data
        print("  Retraining on full data...")
        self.base_models = self._create_base_models()
        for model in self.base_models:
            model.fit(X_scaled, y)

        self.cv_scores = fold_aucs
        self.is_fitted = True

        mean_auc = np.mean(fold_aucs)
        std_auc = np.std(fold_aucs)
        print(f"  CV AUC: {mean_auc:.4f} ± {std_auc:.4f}")

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities using stacked ensemble."""
        if not self.is_fitted:
            raise RuntimeError("Ensemble must be fitted first")

        X_scaled = self.scaler.transform(X)

        # Get base model predictions
        meta_features = np.column_stack([
            model.predict_proba(X_scaled)[:, 1] for model in self.base_models
        ])

        # Meta-learner prediction
        predictions = self.meta_model.predict(meta_features)

        # Clip to valid probability range
        return np.clip(predictions, 0.001, 0.999)

    def get_feature_importance(self) -> np.ndarray:
        """Get averaged feature importance from XGBoost."""
        if not self.is_fitted:
            return None
        # Use XGBoost importance (first model)
        return self.base_models[0].feature_importances_


# ============================================================================
# Task B: Sequence Identification
# ============================================================================
def identify_sequences_for_dataset(train_path: Path, metadata: pd.DataFrame,
                                   feature_extractor: FeatureExtractor,
                                   ensemble: StackingEnsemble,
                                   top_k: int = 50000) -> pd.DataFrame:
    """Identify top disease-associated sequences using multiple signals."""

    print(f"\n  [Task B] {train_path.name}...")

    # Get feature importance from ensemble
    importance = ensemble.get_feature_importance()

    # Map k-mer features to importance scores
    kmer_importance = {}
    for i, name in enumerate(feature_extractor.feature_names):
        if name.startswith('kmer_'):
            kmer_importance[name] = importance[i] if importance is not None else 1.0

    # Score sequences
    seq_scores = defaultdict(lambda: {'score': 0.0, 'count': 0, 'pos_count': 0, 'v': '', 'j': ''})

    for _, row in tqdm(metadata.iterrows(), total=len(metadata), desc="    Scanning", leave=False):
        tsv_path = train_path / row['filename']
        if not tsv_path.exists():
            continue

        label = int(row['label_positive'])
        weight = 2.0 if label == 1 else 0.3

        try:
            df = pd.read_csv(tsv_path, sep='\t', usecols=['junction_aa', 'v_call', 'j_call'])

            for _, seq_row in df.iterrows():
                junction = seq_row.get('junction_aa')
                if pd.isna(junction):
                    continue
                junction = str(junction)
                if len(junction) < 8 or not junction.isalpha():
                    continue

                # Calculate k-mer importance score
                kmer_score = 0.0
                seen_kmers = set()
                for k in feature_extractor.k_mer_sizes:
                    for i in range(len(junction) - k + 1):
                        kmer = junction[i:i+k]
                        kmer_key = f'kmer_{k}_{kmer}'
                        if kmer_key in kmer_importance and kmer not in seen_kmers:
                            kmer_score += kmer_importance[kmer_key]
                            seen_kmers.add(kmer)

                # Combine with label weight
                final_score = weight * (1 + kmer_score)

                seq_scores[junction]['score'] += final_score
                seq_scores[junction]['count'] += 1
                if label == 1:
                    seq_scores[junction]['pos_count'] += 1

                # Store V/J info
                v_call = seq_row.get('v_call')
                j_call = seq_row.get('j_call')
                if pd.notna(v_call) and not seq_scores[junction]['v']:
                    seq_scores[junction]['v'] = str(v_call).split('*')[0]
                if pd.notna(j_call) and not seq_scores[junction]['j']:
                    seq_scores[junction]['j'] = str(j_call).split('*')[0]

        except Exception:
            continue

    # Sort by score and differential enrichment
    sorted_seqs = sorted(
        seq_scores.items(),
        key=lambda x: (x[1]['score'] * (x[1]['pos_count'] / max(x[1]['count'], 1) + 0.1), x[1]['count']),
        reverse=True
    )[:top_k]

    # Build result DataFrame
    results = []
    ds_name = train_path.name

    for rank, (junction, info) in enumerate(sorted_seqs, 1):
        results.append({
            'ID': f'{ds_name}_seq_top_{rank}',
            'dataset': ds_name,
            'label_positive_probability': -999.0,
            'junction_aa': junction,
            'v_call': info['v'] if info['v'] else 'TRBV20-1',
            'j_call': info['j'] if info['j'] else 'TRBJ2-7'
        })

    # Pad if needed
    while len(results) < top_k:
        rank = len(results) + 1
        results.append({
            'ID': f'{ds_name}_seq_top_{rank}',
            'dataset': ds_name,
            'label_positive_probability': -999.0,
            'junction_aa': 'CASSXXX',
            'v_call': 'TRBV20-1',
            'j_call': 'TRBJ2-7'
        })

    print(f"    {len(sorted_seqs)} unique sequences identified")

    return pd.DataFrame(results)


# ============================================================================
# Main Pipeline
# ============================================================================
def main():
    start_time = datetime.now()
    print("=" * 80)
    print("  AIRR-ML-25 Championship Ensemble with K-Fold CV")
    print(f"  Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Target: Beat GROZD (0.81364)")
    print("=" * 80)

    CHECKPOINT_DIR.mkdir(exist_ok=True)

    # ========================================================================
    # Phase 1: Load all training data and build vocabulary
    # ========================================================================
    print("\n" + "=" * 60)
    print("Phase 1: Building Feature Vocabulary")
    print("=" * 60)

    train_paths = []
    metadata_list = []

    for ds in range(1, 9):
        train_path = TRAIN_ROOT / f'train_dataset_{ds}'
        if train_path.exists():
            metadata = pd.read_csv(train_path / 'metadata.csv')
            train_paths.append(train_path)
            metadata_list.append(metadata)
            print(f"  DS{ds}: {len(metadata)} samples")

    # Build feature extractor
    feature_extractor = FeatureExtractor(k_mer_sizes=K_MER_SIZES)
    feature_extractor.fit(train_paths, metadata_list)

    # Save feature extractor
    with open(CHECKPOINT_DIR / 'feature_extractor.pkl', 'wb') as f:
        pickle.dump(feature_extractor, f)

    # ========================================================================
    # Phase 2: Train per-dataset ensembles with K-Fold CV
    # ========================================================================
    print("\n" + "=" * 60)
    print("Phase 2: Training Stacking Ensembles (K-Fold CV)")
    print("=" * 60)

    ensembles = {}
    all_cv_results = []

    for ds_idx, (train_path, metadata) in enumerate(zip(train_paths, metadata_list)):
        ds_num = ds_idx + 1
        print(f"\n--- Dataset {ds_num} ({train_path.name}) ---")

        # Extract features for all repertoires
        X_list = []
        y_list = []

        for _, row in tqdm(metadata.iterrows(), total=len(metadata), desc="  Extracting features", leave=False):
            tsv_path = train_path / row['filename']
            try:
                df = pd.read_csv(tsv_path, sep='\t')
                features = feature_extractor.transform(df)
                X_list.append(features)
                y_list.append(int(row['label_positive']))
            except Exception:
                X_list.append(np.zeros(len(feature_extractor.feature_names), dtype=np.float32))
                y_list.append(int(row['label_positive']))

        X = np.array(X_list)
        y = np.array(y_list)

        print(f"  Data shape: {X.shape}, Class distribution: {np.bincount(y)}")

        # Train stacking ensemble
        ensemble = StackingEnsemble(n_folds=N_FOLDS, random_state=RANDOM_STATE)
        ensemble.fit(X, y)

        # Save
        ensembles[ds_num] = ensemble
        with open(CHECKPOINT_DIR / f'ensemble_ds{ds_num}.pkl', 'wb') as f:
            pickle.dump({'ensemble': ensemble, 'cv_scores': ensemble.cv_scores}, f)

        all_cv_results.append({
            'dataset': train_path.name,
            'cv_mean': np.mean(ensemble.cv_scores),
            'cv_std': np.std(ensemble.cv_scores),
            'cv_scores': ensemble.cv_scores
        })

        gc.collect()

    # Save CV results
    with open(CHECKPOINT_DIR / 'cv_results.json', 'w') as f:
        json.dump(all_cv_results, f, indent=2)

    print("\n" + "-" * 40)
    print("Cross-Validation Summary:")
    for result in all_cv_results:
        print(f"  {result['dataset']}: {result['cv_mean']:.4f} ± {result['cv_std']:.4f}")
    overall_cv = np.mean([r['cv_mean'] for r in all_cv_results])
    print(f"  Overall CV AUC: {overall_cv:.4f}")

    # ========================================================================
    # Phase 3: Task A - Test Predictions
    # ========================================================================
    print("\n" + "=" * 60)
    print("Phase 3: Task A - Test Predictions")
    print("=" * 60)

    predictions = []

    for test_name, train_ds in TEST_TO_TRAIN.items():
        test_path = TEST_ROOT / test_name
        if not test_path.exists():
            continue

        if train_ds not in ensembles:
            print(f"  Skipping {test_name}: no model")
            continue

        ensemble = ensembles[train_ds]
        tsv_files = list(test_path.glob('*.tsv'))

        print(f"\n  {test_name}: {len(tsv_files)} files")

        for tsv_file in tqdm(tsv_files, desc="    Predicting", leave=False):
            try:
                df = pd.read_csv(tsv_file, sep='\t')
                features = feature_extractor.transform(df)
                prob = ensemble.predict_proba(features.reshape(1, -1))[0]
            except Exception:
                prob = 0.5

            predictions.append({
                'ID': tsv_file.stem,
                'dataset': test_name,
                'label_positive_probability': float(prob),
                'junction_aa': -999.0,
                'v_call': -999.0,
                'j_call': -999.0
            })

        probs = [p['label_positive_probability'] for p in predictions if p['dataset'] == test_name]
        print(f"    Range: [{min(probs):.4f}, {max(probs):.4f}], Mean: {np.mean(probs):.4f}")

    print(f"\n  Total Task A predictions: {len(predictions)}")

    # ========================================================================
    # Phase 4: Task B - Sequence Identification
    # ========================================================================
    print("\n" + "=" * 60)
    print("Phase 4: Task B - Sequence Identification")
    print("=" * 60)

    all_sequences = []

    for ds_idx, (train_path, metadata) in enumerate(zip(train_paths, metadata_list)):
        ds_num = ds_idx + 1
        if ds_num not in ensembles:
            continue

        sequences_df = identify_sequences_for_dataset(
            train_path, metadata, feature_extractor, ensembles[ds_num], top_k=TOP_K
        )
        all_sequences.append(sequences_df)
        gc.collect()

    task_b_df = pd.concat(all_sequences, ignore_index=True)

    # ========================================================================
    # Phase 5: Generate Submission
    # ========================================================================
    print("\n" + "=" * 60)
    print("Phase 5: Generating Submission")
    print("=" * 60)

    task_a_df = pd.DataFrame(predictions)
    submission_df = pd.concat([task_a_df, task_b_df], ignore_index=True)
    submission_df = submission_df[['ID', 'dataset', 'label_positive_probability', 'junction_aa', 'v_call', 'j_call']]

    # Save
    SUBMISSION_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = SUBMISSION_DIR / f'ensemble_kfold_{timestamp}.csv'
    submission_df.to_csv(out_path, index=False)

    # Validation
    expected = 4213 + 8 * TOP_K
    print(f"\n  Total rows: {len(submission_df)} (expected: {expected})")
    print(f"  Match: {len(submission_df) == expected}")
    print(f"  File: {out_path}")
    print(f"  Size: {out_path.stat().st_size / 1e6:.2f} MB")

    # Stats
    task_a_probs = task_a_df['label_positive_probability']
    print(f"\n  Task A Stats:")
    print(f"    Count: {len(task_a_probs)}")
    print(f"    Range: [{task_a_probs.min():.4f}, {task_a_probs.max():.4f}]")
    print(f"    Mean: {task_a_probs.mean():.4f}")
    print(f"    Std: {task_a_probs.std():.4f}")

    # Per-dataset stats
    print(f"\n  Per-Dataset Stats:")
    for test_name in TEST_TO_TRAIN.keys():
        ds_probs = task_a_df[task_a_df['dataset'] == test_name]['label_positive_probability']
        if len(ds_probs) > 0:
            print(f"    {test_name}: [{ds_probs.min():.3f}, {ds_probs.max():.3f}], mean={ds_probs.mean():.3f}")

    duration = datetime.now() - start_time
    print(f"\n" + "=" * 60)
    print(f"  COMPLETE in {duration}")
    print(f"  CV AUC: {overall_cv:.4f}")
    print(f"  Submission: {out_path}")
    print("=" * 60)

    return str(out_path)


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
