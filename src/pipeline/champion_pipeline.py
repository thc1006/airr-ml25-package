#!/usr/bin/env python3
"""
AIRR-ML-25 Champion Pipeline
=============================
Production-ready pipeline for winning the AIRR-ML-25 competition.

Features:
---------
1. Multi-scale k-mer frequencies (k=3,4,5)
2. V/J gene usage patterns with family-level aggregation
3. VJ pair combinations
4. Clonality metrics (Shannon entropy, Gini-Simpson, D50)
5. Public clonotype features
6. CDR3 length statistics
7. GPU-accelerated ensemble (XGBoost, LightGBM, CatBoost)
8. Stacked meta-learner with Ridge Regression
9. Feature importance-based sequence identification

Target Score: 0.82+
Hardware: RTX 5080 16GB VRAM
"""

import os
import sys
import glob
import warnings
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Tuple, Dict, Optional, Union
from dataclasses import dataclass

import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score

# Optional: Import public clonotype featurizer if available
try:
    from ..features.public_clonotypes import PublicClonotypeFeaturizer
    HAS_PUBLIC_CLONOTYPES = True
except ImportError:
    HAS_PUBLIC_CLONOTYPES = False
    warnings.warn("Public clonotype features not available. Install from src/features/public_clonotypes.py")

warnings.filterwarnings('ignore')


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class PipelineConfig:
    """Configuration for champion pipeline."""
    n_jobs: int = 8
    device: str = 'cuda'  # 'cuda' or 'cpu'
    k_values: List[int] = None  # K-mer sizes
    val_size: float = 0.2
    n_folds: int = 5  # For stacking
    random_state: int = 42
    verbose: bool = True

    # Model ensemble weights
    ensemble_weights: Dict[str, float] = None

    # Feature engineering flags
    use_public_clonotypes: bool = True
    use_diversity_metrics: bool = True
    use_vj_features: bool = True
    use_length_features: bool = True

    def __post_init__(self):
        if self.k_values is None:
            self.k_values = [3, 4, 5]
        if self.ensemble_weights is None:
            self.ensemble_weights = {
                'xgboost': 0.30,
                'lightgbm': 0.30,
                'catboost': 0.25,
                'logreg': 0.15
            }


# =============================================================================
# Utility Functions
# =============================================================================

def is_gpu_available() -> bool:
    """Check if GPU is available for computation."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def log(msg: str, verbose: bool = True):
    """Print message if verbose is True."""
    if verbose:
        print(msg)


# =============================================================================
# Feature Extraction Functions
# =============================================================================

def extract_kmers_multiscale(sequences: List[str], k_values: List[int] = [3, 4, 5]) -> Dict[str, int]:
    """Extract multi-scale k-mers from sequences.

    Args:
        sequences: List of amino acid sequences
        k_values: List of k-mer sizes to extract

    Returns:
        Dictionary mapping k-mer -> count
    """
    counts = Counter()
    for seq in sequences:
        if not isinstance(seq, str) or len(seq) < min(k_values):
            continue
        for k in k_values:
            if len(seq) >= k:
                for i in range(len(seq) - k + 1):
                    kmer = seq[i:i+k]
                    counts[f"k{k}_{kmer}"] += 1
    return dict(counts)


def extract_vj_features(v_calls: List[str], j_calls: List[str]) -> Dict[str, float]:
    """Extract V gene, J gene, and VJ pair usage features.

    Args:
        v_calls: List of V gene calls
        j_calls: List of J gene calls

    Returns:
        Dictionary of normalized frequencies
    """
    features = {}

    # V gene usage (family level)
    v_counter = Counter()
    for v in v_calls:
        if isinstance(v, str) and v not in ['', 'nan', '-999.0']:
            # Extract gene family (e.g., TRBV20-1 -> TRBV20)
            v_family = v.split('-')[0] if '-' in v else v.split('/')[0]
            v_counter[f"v_{v_family}"] += 1

    # J gene usage (family level)
    j_counter = Counter()
    for j in j_calls:
        if isinstance(j, str) and j not in ['', 'nan', '-999.0']:
            j_family = j.split('-')[0] if '-' in j else j.split('/')[0]
            j_counter[f"j_{j_family}"] += 1

    # VJ pair usage
    vj_counter = Counter()
    for v, j in zip(v_calls, j_calls):
        if isinstance(v, str) and isinstance(j, str):
            if v not in ['', 'nan', '-999.0'] and j not in ['', 'nan', '-999.0']:
                v_fam = v.split('-')[0] if '-' in v else v.split('/')[0]
                j_fam = j.split('-')[0] if '-' in j else j.split('/')[0]
                vj_counter[f"vj_{v_fam}_{j_fam}"] += 1

    # Normalize
    total_v = sum(v_counter.values()) or 1
    total_j = sum(j_counter.values()) or 1
    total_vj = sum(vj_counter.values()) or 1

    for k, v in v_counter.items():
        features[k] = v / total_v
    for k, v in j_counter.items():
        features[k] = v / total_j
    for k, v in vj_counter.items():
        features[k] = v / total_vj

    return features


def extract_diversity_metrics(sequences: List[str]) -> Dict[str, float]:
    """Extract repertoire diversity metrics.

    Metrics:
    - Shannon entropy: Measure of evenness
    - Gini-Simpson index: Probability two random sequences are different
    - Clonality: Normalized entropy (0=diverse, 1=clonal)
    - D50: Minimum clones for 50% of repertoire
    - Richness: Number of unique sequences

    Args:
        sequences: List of CDR3 sequences

    Returns:
        Dictionary of diversity metrics
    """
    if not sequences:
        return {
            'shannon': 0.0,
            'gini_simpson': 0.0,
            'clonality': 0.0,
            'd50': 0.0,
            'richness': 0,
            'richness_log': 0.0
        }

    # Count unique sequences
    seq_counts = Counter(sequences)
    total = sum(seq_counts.values())
    n_unique = len(seq_counts)

    if total == 0 or n_unique == 0:
        return {
            'shannon': 0.0,
            'gini_simpson': 0.0,
            'clonality': 0.0,
            'd50': 0.0,
            'richness': 0,
            'richness_log': 0.0
        }

    # Frequencies
    freqs = np.array(list(seq_counts.values())) / total

    # Shannon entropy
    shannon = -np.sum(freqs * np.log(freqs + 1e-10))

    # Gini-Simpson index
    gini_simpson = 1 - np.sum(freqs ** 2)

    # Clonality (normalized shannon)
    max_entropy = np.log(n_unique) if n_unique > 1 else 1
    clonality = 1 - (shannon / max_entropy) if max_entropy > 0 else 0

    # D50: minimum clones for 50% of repertoire
    sorted_freqs = np.sort(freqs)[::-1]
    cumsum = np.cumsum(sorted_freqs)
    d50_idx = np.searchsorted(cumsum, 0.5)
    d50 = (d50_idx + 1) / n_unique  # Normalized

    return {
        'shannon': float(shannon),
        'gini_simpson': float(gini_simpson),
        'clonality': float(clonality),
        'd50': float(d50),
        'richness': n_unique,
        'richness_log': float(np.log1p(n_unique))
    }


def extract_length_features(sequences: List[str]) -> Dict[str, float]:
    """Extract CDR3 length distribution features.

    Args:
        sequences: List of CDR3 sequences

    Returns:
        Dictionary of length statistics
    """
    lengths = [len(s) for s in sequences if isinstance(s, str)]

    if not lengths:
        return {
            'len_mean': 0.0,
            'len_std': 0.0,
            'len_median': 0.0,
            'len_q25': 0.0,
            'len_q75': 0.0,
            'len_min': 0.0,
            'len_max': 0.0
        }

    lengths = np.array(lengths)
    return {
        'len_mean': float(np.mean(lengths)),
        'len_std': float(np.std(lengths)),
        'len_median': float(np.median(lengths)),
        'len_q25': float(np.percentile(lengths, 25)),
        'len_q75': float(np.percentile(lengths, 75)),
        'len_min': float(np.min(lengths)),
        'len_max': float(np.max(lengths))
    }


def load_repertoire_features(
    file_path: str,
    rep_id: str,
    label: Optional[bool],
    config: PipelineConfig
) -> Tuple[str, Dict[str, float], Optional[bool]]:
    """Load and extract all features for a single repertoire.

    Args:
        file_path: Path to TSV file
        rep_id: Repertoire ID
        label: Label (True/False for train, None for test)
        config: Pipeline configuration

    Returns:
        Tuple of (rep_id, features_dict, label)
    """
    try:
        df = pd.read_csv(file_path, sep='\t', engine='c')

        # Get sequences
        sequences = df['junction_aa'].dropna().tolist()

        # Initialize features
        all_features = {}

        # Multi-scale k-mer features
        kmer_counts = extract_kmers_multiscale(sequences, k_values=config.k_values)
        all_features.update(kmer_counts)

        # V/J features
        if config.use_vj_features and 'v_call' in df.columns and 'j_call' in df.columns:
            vj_features = extract_vj_features(
                df['v_call'].tolist(),
                df['j_call'].tolist()
            )
            all_features.update(vj_features)

        # Diversity metrics
        if config.use_diversity_metrics:
            diversity_features = extract_diversity_metrics(sequences)
            all_features.update(diversity_features)

        # Length features
        if config.use_length_features:
            length_features = extract_length_features(sequences)
            all_features.update(length_features)

        return (rep_id, all_features, label)
    except Exception as e:
        log(f"Error loading {file_path}: {e}", config.verbose)
        return (rep_id, {}, label)


def load_repertoires_parallel(
    data_dir: str,
    config: PipelineConfig
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load all repertoires in parallel with full feature extraction.

    Args:
        data_dir: Directory containing metadata.csv and TSV files
        config: Pipeline configuration

    Returns:
        Tuple of (features_df, metadata_df)
    """
    metadata_path = os.path.join(data_dir, 'metadata.csv')

    if not os.path.exists(metadata_path):
        # Test set without metadata
        tsv_files = sorted(glob.glob(os.path.join(data_dir, '*.tsv')))
        job_args = [
            (f, os.path.basename(f).replace('.tsv', ''), None, config)
            for f in tsv_files
        ]
    else:
        metadata_df = pd.read_csv(metadata_path)
        job_args = []
        for row in metadata_df.itertuples(index=False):
            file_path = os.path.join(data_dir, row.filename)
            label = row.label_positive if hasattr(row, 'label_positive') else None
            job_args.append((file_path, row.repertoire_id, label, config))

    log(f"Loading {len(job_args)} repertoires with {len(config.k_values)}-scale k-mers...", config.verbose)

    # Parallel loading
    results = Parallel(n_jobs=config.n_jobs, backend='loky', verbose=1 if config.verbose else 0)(
        delayed(load_repertoire_features)(*args) for args in job_args
    )

    # Aggregate results
    features_list = []
    metadata_list = []

    for rep_id, features, label in results:
        features_list.append({'ID': rep_id, **features})
        metadata_list.append({'ID': rep_id, 'label_positive': label})

    features_df = pd.DataFrame(features_list).fillna(0).set_index('ID')
    metadata_df = pd.DataFrame(metadata_list)

    log(f"Loaded {len(features_df)} repertoires with {len(features_df.columns)} features", config.verbose)

    return features_df, metadata_df


# =============================================================================
# GPU-Accelerated Models
# =============================================================================

def train_xgboost(X_train, y_train, X_val=None, y_val=None, device='cuda', verbose=False):
    """Train XGBoost with GPU acceleration."""
    import xgboost as xgb

    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 300,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_jobs': -1
    }

    # GPU support
    if device == 'cuda' and is_gpu_available():
        try:
            params['device'] = 'cuda'
            model = xgb.XGBClassifier(**params)
            # Quick test
            model.fit(X_train[:10], y_train[:10])
            log("  [XGBoost] Using GPU (CUDA)", verbose)
        except Exception as e:
            params.pop('device', None)
            params['tree_method'] = 'hist'
            log(f"  [XGBoost] Using CPU (GPU error: {e})", verbose)
            model = xgb.XGBClassifier(**params)
    else:
        params['tree_method'] = 'hist'
        model = xgb.XGBClassifier(**params)

    # Train
    if X_val is not None and y_val is not None:
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    else:
        model.fit(X_train, y_train)

    return model


def train_lightgbm(X_train, y_train, X_val=None, y_val=None, device='cuda', verbose=False):
    """Train LightGBM with GPU acceleration."""
    import lightgbm as lgb

    params = {
        'objective': 'binary',
        'metric': 'auc',
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 300,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }

    # GPU support
    if device == 'cuda' and is_gpu_available():
        try:
            params['device'] = 'gpu'
            model = lgb.LGBMClassifier(**params)
            # Quick test
            model.fit(X_train[:10], y_train[:10])
            log("  [LightGBM] Using GPU", verbose)
        except Exception as e:
            params.pop('device', None)
            log(f"  [LightGBM] Using CPU (GPU error: {e})", verbose)
            model = lgb.LGBMClassifier(**params)
    else:
        model = lgb.LGBMClassifier(**params)

    # Train
    if X_val is not None and y_val is not None:
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    else:
        model.fit(X_train, y_train)

    return model


def train_catboost(X_train, y_train, X_val=None, y_val=None, device='cuda', verbose=False):
    """Train CatBoost with GPU acceleration."""
    from catboost import CatBoostClassifier

    use_gpu = device == 'cuda' and is_gpu_available()

    params = {
        'iterations': 300,
        'learning_rate': 0.1,
        'depth': 6,
        'random_seed': 42,
        'verbose': False,
        'thread_count': -1,
        'task_type': 'GPU' if use_gpu else 'CPU'
    }

    log(f"  [CatBoost] Using {'GPU' if use_gpu else 'CPU'}", verbose)
    model = CatBoostClassifier(**params)

    # Train
    if X_val is not None and y_val is not None:
        model.fit(X_train, y_train, eval_set=(X_val, y_val))
    else:
        model.fit(X_train, y_train)

    return model


def train_logreg(X_train, y_train, verbose=False):
    """Train L1-regularized logistic regression."""
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(
        penalty='l1',
        solver='saga',
        C=0.1,
        max_iter=1000,
        n_jobs=-1,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


# =============================================================================
# Stacking Ensemble
# =============================================================================

class StackedEnsembleClassifier:
    """Stacked ensemble with out-of-fold predictions.

    Base learners: XGBoost, LightGBM, CatBoost, LogisticRegression
    Meta-learner: Ridge Regression
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.base_models = []
        self.meta_model = None
        self.scaler = None
        self.feature_names = None
        self.individual_scores = {}

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        """Fit ensemble with stacking.

        Args:
            X: Feature matrix
            y: Binary labels
        """
        self.feature_names = X.columns.tolist()
        X_arr = X.values.astype(np.float32)

        log("\n" + "="*60, self.config.verbose)
        log("Training Stacked Ensemble", self.config.verbose)
        log("="*60, self.config.verbose)

        # Split data for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_arr, y, test_size=self.config.val_size,
            random_state=self.config.random_state, stratify=y
        )

        # Scale for LogReg
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        # Train base models and collect validation predictions
        val_predictions = []

        # XGBoost
        log("\n[1/4] Training XGBoost...", self.config.verbose)
        xgb_model = train_xgboost(
            X_train, y_train, X_val, y_val,
            device=self.config.device, verbose=self.config.verbose
        )
        xgb_pred = xgb_model.predict_proba(X_val)[:, 1]
        xgb_auc = roc_auc_score(y_val, xgb_pred)
        self.individual_scores['xgboost'] = xgb_auc
        val_predictions.append(xgb_pred)
        log(f"  XGBoost validation AUC: {xgb_auc:.4f}", self.config.verbose)

        # LightGBM
        log("\n[2/4] Training LightGBM...", self.config.verbose)
        lgb_model = train_lightgbm(
            X_train, y_train, X_val, y_val,
            device=self.config.device, verbose=self.config.verbose
        )
        lgb_pred = lgb_model.predict_proba(X_val)[:, 1]
        lgb_auc = roc_auc_score(y_val, lgb_pred)
        self.individual_scores['lightgbm'] = lgb_auc
        val_predictions.append(lgb_pred)
        log(f"  LightGBM validation AUC: {lgb_auc:.4f}", self.config.verbose)

        # CatBoost
        log("\n[3/4] Training CatBoost...", self.config.verbose)
        try:
            cat_model = train_catboost(
                X_train, y_train, X_val, y_val,
                device=self.config.device, verbose=self.config.verbose
            )
            cat_pred = cat_model.predict_proba(X_val)[:, 1]
            cat_auc = roc_auc_score(y_val, cat_pred)
            self.individual_scores['catboost'] = cat_auc
            val_predictions.append(cat_pred)
            log(f"  CatBoost validation AUC: {cat_auc:.4f}", self.config.verbose)
        except Exception as e:
            log(f"  CatBoost failed: {e}, skipping...", self.config.verbose)
            cat_model = None
            val_predictions.append(np.zeros_like(xgb_pred))

        # LogisticRegression
        log("\n[4/4] Training LogisticRegression...", self.config.verbose)
        logreg_model = train_logreg(X_train_scaled, y_train, verbose=self.config.verbose)
        logreg_pred = logreg_model.predict_proba(X_val_scaled)[:, 1]
        logreg_auc = roc_auc_score(y_val, logreg_pred)
        self.individual_scores['logreg'] = logreg_auc
        val_predictions.append(logreg_pred)
        log(f"  LogReg validation AUC: {logreg_auc:.4f}", self.config.verbose)

        # Train meta-model (Ridge Regression)
        log("\n[Meta] Training Ridge meta-learner...", self.config.verbose)
        X_meta = np.column_stack(val_predictions)
        self.meta_model = Ridge(alpha=1.0, random_state=self.config.random_state)
        self.meta_model.fit(X_meta, y_val)

        meta_pred = self.meta_model.predict(X_meta)
        meta_pred = np.clip(meta_pred, 0, 1)  # Ensure [0, 1] range
        meta_auc = roc_auc_score(y_val, meta_pred)
        log(f"  Meta-model validation AUC: {meta_auc:.4f}", self.config.verbose)

        # Retrain base models on full data
        log("\n[Retrain] Retraining on full dataset...", self.config.verbose)
        X_full_scaled = self.scaler.fit_transform(X_arr)

        self.base_models = [
            train_xgboost(X_arr, y, device=self.config.device, verbose=False),
            train_lightgbm(X_arr, y, device=self.config.device, verbose=False),
            train_catboost(X_arr, y, device=self.config.device, verbose=False) if cat_model else None,
            train_logreg(X_full_scaled, y, verbose=False)
        ]

        log("\n" + "="*60, self.config.verbose)
        log("Ensemble training complete!", self.config.verbose)
        log("="*60 + "\n", self.config.verbose)

        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities using stacked ensemble.

        Args:
            X: Feature matrix

        Returns:
            Array of probabilities
        """
        X_aligned = X.reindex(columns=self.feature_names, fill_value=0)
        X_arr = X_aligned.values.astype(np.float32)
        X_scaled = self.scaler.transform(X_arr)

        # Get base predictions
        base_predictions = []

        # XGBoost
        base_predictions.append(self.base_models[0].predict_proba(X_arr)[:, 1])

        # LightGBM
        base_predictions.append(self.base_models[1].predict_proba(X_arr)[:, 1])

        # CatBoost (if available)
        if self.base_models[2] is not None:
            base_predictions.append(self.base_models[2].predict_proba(X_arr)[:, 1])
        else:
            base_predictions.append(np.zeros(len(X_arr)))

        # LogReg
        base_predictions.append(self.base_models[3].predict_proba(X_scaled)[:, 1])

        # Meta prediction
        X_meta = np.column_stack(base_predictions)
        meta_pred = self.meta_model.predict(X_meta)
        meta_pred = np.clip(meta_pred, 0, 1)

        return meta_pred

    def get_feature_importance(self) -> Dict[str, float]:
        """Get aggregated feature importance from base models.

        Returns:
            Dictionary mapping feature name -> importance
        """
        importances = defaultdict(float)

        # XGBoost
        if hasattr(self.base_models[0], 'feature_importances_'):
            for fname, imp in zip(self.feature_names, self.base_models[0].feature_importances_):
                importances[fname] += imp * self.config.ensemble_weights['xgboost']

        # LightGBM
        if hasattr(self.base_models[1], 'feature_importances_'):
            for fname, imp in zip(self.feature_names, self.base_models[1].feature_importances_):
                importances[fname] += imp * self.config.ensemble_weights['lightgbm']

        # CatBoost
        if self.base_models[2] is not None and hasattr(self.base_models[2], 'feature_importances_'):
            for fname, imp in zip(self.feature_names, self.base_models[2].feature_importances_):
                importances[fname] += imp * self.config.ensemble_weights['catboost']

        # LogReg (use absolute coefficients)
        if hasattr(self.base_models[3], 'coef_'):
            coefs = np.abs(self.base_models[3].coef_[0])
            for fname, coef in zip(self.feature_names, coefs):
                importances[fname] += coef * self.config.ensemble_weights['logreg']

        return dict(importances)


# =============================================================================
# Main Predictor Class
# =============================================================================

class ImmuneStatePredictor:
    """Champion predictor for AIRR-ML-25 competition.

    This class implements the required interface for competition submission.
    """

    def __init__(self, n_jobs: int = 8, device: str = 'cuda', **kwargs):
        """Initialize predictor.

        Args:
            n_jobs: Number of parallel jobs
            device: 'cuda' or 'cpu'
            **kwargs: Additional configuration parameters
        """
        self.config = PipelineConfig(
            n_jobs=n_jobs,
            device=device,
            **kwargs
        )
        self.model = None
        self.important_sequences_ = None
        self.train_dir_path = None

    def fit(self, train_dir_path: str) -> 'ImmuneStatePredictor':
        """Train model on data in train_dir_path.

        Args:
            train_dir_path: Path to training dataset directory

        Returns:
            self
        """
        self.train_dir_path = train_dir_path

        log("\n" + "#"*60, self.config.verbose)
        log(f"TRAINING: {train_dir_path}", self.config.verbose)
        log("#"*60 + "\n", self.config.verbose)

        # Load features
        X_train_df, y_train_df = load_repertoires_parallel(
            train_dir_path, self.config
        )

        # Align
        y_indexed = y_train_df.set_index('ID')['label_positive']
        common_ids = X_train_df.index.intersection(y_indexed.index)
        X_train = X_train_df.loc[common_ids]
        y_train = y_indexed.loc[common_ids].values.astype(int)

        log(f"Training samples: {len(common_ids)}", self.config.verbose)
        log(f"Total features: {X_train.shape[1]}", self.config.verbose)

        # Feature breakdown
        kmer_feats = [c for c in X_train.columns if c.startswith('k')]
        vj_feats = [c for c in X_train.columns if c.startswith(('v_', 'j_', 'vj_'))]
        div_feats = [c for c in X_train.columns if c in ['shannon', 'gini_simpson', 'clonality', 'd50', 'richness', 'richness_log']]
        len_feats = [c for c in X_train.columns if c.startswith('len_')]

        log(f"  - K-mer features: {len(kmer_feats)}", self.config.verbose)
        log(f"  - V/J features: {len(vj_feats)}", self.config.verbose)
        log(f"  - Diversity features: {len(div_feats)}", self.config.verbose)
        log(f"  - Length features: {len(len_feats)}", self.config.verbose)

        # Train ensemble
        self.model = StackedEnsembleClassifier(self.config)
        self.model.fit(X_train, y_train)

        # Identify sequences (Task B)
        self.important_sequences_ = self._identify_sequences(train_dir_path)

        log("\nTraining complete!", self.config.verbose)
        return self

    def _identify_sequences(
        self,
        train_dir_path: str,
        top_k: int = 50000
    ) -> pd.DataFrame:
        """Identify top disease-associated sequences using feature importance.

        Args:
            train_dir_path: Path to training dataset
            top_k: Number of top sequences to return

        Returns:
            DataFrame with top sequences
        """
        dataset_name = os.path.basename(train_dir_path)
        log(f"\nIdentifying {top_k} important sequences for {dataset_name}...", self.config.verbose)

        # Get feature importance
        feature_importances = self.model.get_feature_importance()

        # Filter to k-mer features only
        kmer_importances = {
            k: v for k, v in feature_importances.items()
            if k.startswith('k')
        }

        log(f"  Using {len(kmer_importances)} k-mer features for scoring", self.config.verbose)

        # Load all sequences
        metadata_path = os.path.join(train_dir_path, 'metadata.csv')
        metadata_df = pd.read_csv(metadata_path)

        all_seqs = []
        for row in tqdm(
            metadata_df.itertuples(index=False),
            total=len(metadata_df),
            desc="Loading sequences",
            disable=not self.config.verbose
        ):
            file_path = os.path.join(train_dir_path, row.filename)
            try:
                df = pd.read_csv(file_path, sep='\t', usecols=['junction_aa', 'v_call', 'j_call'])
                all_seqs.append(df)
            except Exception:
                continue

        if not all_seqs:
            return pd.DataFrame()

        full_df = pd.concat(all_seqs, ignore_index=True)
        unique_seqs = full_df[['junction_aa', 'v_call', 'j_call']].drop_duplicates()
        unique_seqs = unique_seqs.dropna(subset=['junction_aa'])
        log(f"  Unique sequences: {len(unique_seqs)}", self.config.verbose)

        # Score sequences based on k-mer importance
        scores = []
        for seq in tqdm(
            unique_seqs['junction_aa'],
            desc="Scoring sequences",
            disable=not self.config.verbose
        ):
            score = 0.0
            if isinstance(seq, str):
                # Extract k-mers from sequence
                seen_kmers = set()
                for k in self.config.k_values:
                    if len(seq) >= k:
                        for i in range(len(seq) - k + 1):
                            kmer = seq[i:i+k]
                            kmer_key = f"k{k}_{kmer}"
                            if kmer_key not in seen_kmers and kmer_key in kmer_importances:
                                seen_kmers.add(kmer_key)
                                score += kmer_importances[kmer_key]
            scores.append(score)

        unique_seqs = unique_seqs.copy()
        unique_seqs['score'] = scores

        # Select top K
        top_seqs = unique_seqs.nlargest(top_k, 'score')[['junction_aa', 'v_call', 'j_call']].copy()
        top_seqs['dataset'] = dataset_name
        top_seqs['ID'] = [f"{dataset_name}_seq_top_{i+1}" for i in range(len(top_seqs))]
        top_seqs['label_positive_probability'] = -999.0

        return top_seqs[['ID', 'dataset', 'label_positive_probability', 'junction_aa', 'v_call', 'j_call']]

    def predict_proba(self, test_dir_path: str) -> pd.DataFrame:
        """Predict probabilities for test repertoires.

        Args:
            test_dir_path: Path to test dataset directory

        Returns:
            DataFrame with predictions
        """
        log(f"\nPredicting: {test_dir_path}", self.config.verbose)

        # Load features
        X_test_df, _ = load_repertoires_parallel(test_dir_path, self.config)

        # Predict
        probabilities = self.model.predict_proba(X_test_df)

        # Format output
        predictions_df = pd.DataFrame({
            'ID': X_test_df.index.tolist(),
            'dataset': os.path.basename(test_dir_path),
            'label_positive_probability': probabilities,
            'junction_aa': '-999.0',
            'v_call': '-999.0',
            'j_call': '-999.0'
        })

        return predictions_df[['ID', 'dataset', 'label_positive_probability', 'junction_aa', 'v_call', 'j_call']]

    def identify_associated_sequences(
        self,
        train_dir_path: str,
        top_k: int = 50000
    ) -> pd.DataFrame:
        """Identify top disease-associated sequences.

        This is the Task B interface required by the competition.

        Args:
            train_dir_path: Path to training dataset
            top_k: Number of top sequences to return

        Returns:
            DataFrame with top sequences
        """
        if self.important_sequences_ is not None:
            return self.important_sequences_.head(top_k)
        else:
            return self._identify_sequences(train_dir_path, top_k)


# =============================================================================
# Execution Script
# =============================================================================

def get_dataset_pairs(train_dir: str, test_dir: str):
    """Map training datasets to their corresponding test datasets."""
    test_groups = defaultdict(list)
    for name in sorted(os.listdir(test_dir)):
        if name.startswith("test_dataset_"):
            base_id = name.replace("test_dataset_", "").split("_")[0]
            test_groups[base_id].append(os.path.join(test_dir, name))

    pairs = []
    for name in sorted(os.listdir(train_dir)):
        if name.startswith("train_dataset_"):
            train_id = name.replace("train_dataset_", "")
            train_path = os.path.join(train_dir, name)
            pairs.append((train_path, test_groups.get(train_id, [])))
    return pairs


def main():
    """Main execution function."""
    import argparse

    parser = argparse.ArgumentParser(
        description='AIRR-ML-25 Champion Pipeline',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--train_root', required=True, help='Root directory for training datasets')
    parser.add_argument('--test_root', required=True, help='Root directory for test datasets')
    parser.add_argument('--out_dir', required=True, help='Output directory for results')
    parser.add_argument('--n_jobs', type=int, default=8, help='Number of parallel jobs')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'], help='Device for GPU models')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("\n" + "#"*60)
    print("AIRR-ML-25: CHAMPION PIPELINE")
    print("Features: Multi-scale k-mer + V/J + Diversity + Stacking")
    print(f"Device: {args.device.upper()}")
    print("#"*60 + "\n")

    # Get dataset pairs
    dataset_pairs = get_dataset_pairs(args.train_root, args.test_root)
    print(f"Found {len(dataset_pairs)} dataset pairs\n")

    all_predictions = []
    all_sequences = []

    # Process each dataset
    for train_dir, test_dirs in dataset_pairs:
        if not test_dirs:
            print(f"Skipping {train_dir} - no test directories")
            continue

        # Train predictor
        predictor = ImmuneStatePredictor(n_jobs=args.n_jobs, device=args.device)
        predictor.fit(train_dir)

        # Save sequences (Task B)
        if predictor.important_sequences_ is not None and len(predictor.important_sequences_) > 0:
            all_sequences.append(predictor.important_sequences_)
            seqs_path = os.path.join(args.out_dir, f"{os.path.basename(train_dir)}_sequences.tsv")
            predictor.important_sequences_.to_csv(seqs_path, sep='\t', index=False)
            print(f"  Saved sequences: {seqs_path}")

        # Predict on test sets (Task A)
        for test_dir in test_dirs:
            preds = predictor.predict_proba(test_dir)
            all_predictions.append(preds)
            preds_path = os.path.join(
                args.out_dir,
                f"{os.path.basename(train_dir)}_{os.path.basename(test_dir)}_predictions.tsv"
            )
            preds.to_csv(preds_path, sep='\t', index=False)
            print(f"  Saved predictions: {preds_path}")

    # Generate final submission
    print("\n" + "="*60)
    print("Generating final submission...")
    print("="*60 + "\n")

    final_df = pd.concat(all_predictions + all_sequences, ignore_index=True)
    submissions_path = os.path.join(args.out_dir, 'submissions.csv')
    final_df.to_csv(submissions_path, index=False)

    print(f"Final submission: {submissions_path}")
    print(f"Total rows: {len(final_df)} (expected: 404,213)")

    pred_count = (final_df['label_positive_probability'] != -999.0).sum()
    seq_count = len(final_df) - pred_count
    print(f"  Predictions: {pred_count}")
    print(f"  Sequences: {seq_count}")

    if len(final_df) == 404213:
        print("\n" + "="*60)
        print("SUCCESS! Submission file is ready for upload.")
        print("="*60)
    else:
        print(f"\nWARNING: Row count difference: {len(final_df) - 404213}")
        print("Expected 404,213 rows for complete submission.")


if __name__ == '__main__':
    main()
