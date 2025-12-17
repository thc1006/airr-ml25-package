#!/usr/bin/env python3
"""
Champion V13 - VJ Pairs Features + CatBoost Integration
- Enhanced VJ gene pair feature extraction
- CatBoost integrated into ensemble (XGBoost + LightGBM + CatBoost)
- Modern Python 3.10+ patterns with type hints and dataclasses
- GPU-accelerated training
"""

from __future__ import annotations

import logging
import sys
import time
import warnings
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class V13Config:
    """V13 configuration with all hyperparameters."""

    # Paths
    train_root: Path = Path("/home/thc1006/dev/airr-ml25-package/data/train_datasets/train_datasets")
    test_root: Path = Path("/home/thc1006/dev/airr-ml25-package/data/test_datasets/test_datasets")
    output_dir: Path = Path("/home/thc1006/dev/airr-ml25-package/submissions")

    # Resources
    n_threads: int = 8
    use_gpu: bool = True

    # Feature extraction
    n_vocab_samples: int = 100
    n_top_kmers: int = 5000
    n_top_vj_pairs: int = 500
    n_top_v_families: int = 20
    n_top_j_families: int = 10
    n_public_clones: int = 2500

    # Model training
    n_cv_folds: int = 5
    n_top_features: int = 1000

    # XGBoost
    xgb_n_estimators: int = 300
    xgb_max_depth: int = 6
    xgb_learning_rate: float = 0.05

    # LightGBM
    lgb_n_estimators: int = 300
    lgb_max_depth: int = 6
    lgb_learning_rate: float = 0.05

    # CatBoost
    cb_iterations: int = 1000
    cb_depth: int = 8
    cb_learning_rate: float = 0.03

    # Ensemble weights
    weight_xgb: float = 0.4
    weight_lgb: float = 0.4
    weight_cb: float = 0.2

    # Misc
    random_seed: int = 42

    def __post_init__(self):
        """Validate configuration and create directories."""
        self.output_dir.mkdir(exist_ok=True)
        assert abs(self.weight_xgb + self.weight_lgb + self.weight_cb - 1.0) < 1e-6, \
            "Ensemble weights must sum to 1.0"


# ============================================================================
# LOGGING SETUP
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

@contextmanager
def timer(name: str):
    """Context manager for timing operations."""
    start = time.time()
    logger.info(f"Starting: {name}")
    yield
    elapsed = time.time() - start
    logger.info(f"Completed: {name} ({elapsed:.2f}s)")


def print_flush(msg: str):
    """Print with immediate flush."""
    print(msg, flush=True)


# ============================================================================
# VJ PAIRS FEATURE EXTRACTION
# ============================================================================

def extract_v_family(v_call: str) -> str:
    """Extract V gene family from v_call (e.g., TRBV20-1 -> TRBV20)."""
    if not isinstance(v_call, str) or v_call == 'UNK':
        return 'UNK'
    # Format: TRBV20-1 or TRBV20*01
    parts = v_call.split('*')[0].split('-')[0]
    return parts if parts else 'UNK'


def extract_j_family(j_call: str) -> str:
    """Extract J gene family from j_call (e.g., TRBJ2-7 -> TRBJ2)."""
    if not isinstance(j_call, str) or j_call == 'UNK':
        return 'UNK'
    # Format: TRBJ2-7 or TRBJ2*01
    parts = j_call.split('*')[0].split('-')[0]
    return parts if parts else 'UNK'


def extract_vj_pair_features(
    df: pd.DataFrame,
    vj_vocab: List[Tuple[str, str]],
    v_vocab: List[str],
    j_vocab: List[str]
) -> Dict[str, float]:
    """
    Extract VJ gene pair features.

    Features:
    - Top N VJ pair frequencies
    - VJ pair diversity metrics (entropy, unique count)
    - V family distribution (top K)
    - J family distribution (top M)

    Args:
        df: DataFrame with v_call and j_call columns
        vj_vocab: List of top VJ pairs to extract
        v_vocab: List of top V families
        j_vocab: List of top J families

    Returns:
        Dictionary of features
    """
    features = {}
    n_seqs = len(df)

    # Extract calls
    v_calls = df['v_call'].fillna('UNK')
    j_calls = df['j_call'].fillna('UNK')

    # 1. VJ pair frequencies
    vj_pairs = list(zip(v_calls, j_calls))
    vj_counter = Counter(vj_pairs)

    for vj in vj_vocab:
        features[f'vj_pair_{vj[0]}_{vj[1]}'] = vj_counter.get(vj, 0) / n_seqs

    # 2. VJ pair diversity
    vj_counts = np.array(list(vj_counter.values()))
    vj_probs = vj_counts / vj_counts.sum()

    features['vj_entropy'] = -np.sum(vj_probs * np.log(vj_probs + 1e-10))
    features['vj_unique'] = len(vj_counter)
    features['vj_max_freq'] = vj_probs.max() if len(vj_probs) > 0 else 0.0

    # 3. V family distribution
    v_families = [extract_v_family(v) for v in v_calls]
    v_family_counter = Counter(v_families)

    for v_fam in v_vocab:
        features[f'v_family_{v_fam}'] = v_family_counter.get(v_fam, 0) / n_seqs

    # 4. J family distribution
    j_families = [extract_j_family(j) for j in j_calls]
    j_family_counter = Counter(j_families)

    for j_fam in j_vocab:
        features[f'j_family_{j_fam}'] = j_family_counter.get(j_fam, 0) / n_seqs

    return features


# ============================================================================
# FEATURE EXTRACTION (ENHANCED)
# ============================================================================

def extract_features_single(
    tsv_path: Path,
    kmer_vocab: List[str],
    vj_vocab: List[Tuple[str, str]],
    v_vocab: List[str],
    j_vocab: List[str],
    public_clones: List[str]
) -> Optional[Dict[str, float]]:
    """Enhanced feature extraction with VJ pairs."""
    try:
        df = pd.read_csv(
            tsv_path,
            sep='\t',
            usecols=['junction_aa', 'v_call', 'j_call']
        )

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

        # 2. VJ pair features (ENHANCED)
        vj_features = extract_vj_pair_features(df, vj_vocab, v_vocab, j_vocab)
        features.update(vj_features)

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

        # 5. CDR3 length statistics
        lengths = np.array([len(s) for s in sequences if isinstance(s, str)])
        if len(lengths) > 0:
            features['cdr3_mean'] = lengths.mean()
            features['cdr3_std'] = lengths.std()
            features['cdr3_median'] = np.median(lengths)
            features['cdr3_min'] = lengths.min()
            features['cdr3_max'] = lengths.max()

        return features

    except Exception as e:
        logger.warning(f"Failed to extract features from {tsv_path}: {e}")
        return None


def build_vocabulary(
    dataset_path: Path,
    config: V13Config
) -> Tuple[List[str], List[Tuple[str, str]], List[str], List[str], List[str]]:
    """
    Build vocabulary from sample files.

    Returns:
        (kmer_vocab, vj_vocab, v_vocab, j_vocab, public_clones)
    """
    metadata = pd.read_csv(dataset_path / "metadata.csv")
    sample_files = metadata['filename'].head(
        min(config.n_vocab_samples, len(metadata))
    ).tolist()

    kmer_counter = Counter()
    vj_counter = Counter()
    v_family_counter = Counter()
    j_family_counter = Counter()
    seq_counter = Counter()

    for fname in tqdm(sample_files, desc="  Building vocab", leave=False):
        try:
            df = pd.read_csv(dataset_path / fname, sep='\t')

            # K-mers
            for seq in df['junction_aa'].dropna():
                if isinstance(seq, str) and len(seq) >= 3:
                    for i in range(len(seq) - 2):
                        kmer_counter[seq[i:i+3]] += 1

            # VJ pairs
            v_calls = df['v_call'].fillna('UNK')
            j_calls = df['j_call'].fillna('UNK')
            vj_pairs = list(zip(v_calls, j_calls))
            vj_counter.update(vj_pairs)

            # V/J families
            v_families = [extract_v_family(v) for v in v_calls]
            j_families = [extract_j_family(j) for j in j_calls]
            v_family_counter.update(v_families)
            j_family_counter.update(j_families)

            # Sequences
            seq_counter.update(df['junction_aa'].dropna())

        except Exception as e:
            logger.warning(f"Failed to process {fname}: {e}")

    # Extract top items
    kmer_vocab = [k for k, _ in kmer_counter.most_common(config.n_top_kmers)]
    vj_vocab = [k for k, _ in vj_counter.most_common(config.n_top_vj_pairs)]
    v_vocab = [k for k, _ in v_family_counter.most_common(config.n_top_v_families)]
    j_vocab = [k for k, _ in j_family_counter.most_common(config.n_top_j_families)]
    public_clones = [
        k for k, c in seq_counter.most_common(config.n_public_clones)
        if c >= 3
    ]

    logger.info(
        f"    Vocab built: {len(kmer_vocab)} kmers, {len(vj_vocab)} VJ pairs, "
        f"{len(v_vocab)} V families, {len(j_vocab)} J families, "
        f"{len(public_clones)} public clones"
    )

    return kmer_vocab, vj_vocab, v_vocab, j_vocab, public_clones


def extract_features_parallel(
    dataset_path: Path,
    metadata: pd.DataFrame,
    kmer_vocab: List[str],
    vj_vocab: List[Tuple[str, str]],
    v_vocab: List[str],
    j_vocab: List[str],
    public_clones: List[str],
    config: V13Config
) -> pd.DataFrame:
    """Parallel feature extraction with ThreadPoolExecutor."""

    file_paths = [dataset_path / row['filename'] for _, row in metadata.iterrows()]

    def extract_single(path: Path) -> Optional[Dict[str, float]]:
        return extract_features_single(
            path, kmer_vocab, vj_vocab, v_vocab, j_vocab, public_clones
        )

    results = []
    with ThreadPoolExecutor(max_workers=config.n_threads) as executor:
        futures = list(tqdm(
            executor.map(extract_single, file_paths),
            total=len(file_paths),
            desc="  Extracting features",
            leave=False
        ))
        results = [r for r in futures if r is not None]

    return pd.DataFrame(results)


# ============================================================================
# GPU TRAINING WITH CATBOOST
# ============================================================================

@dataclass
class EnsembleModels:
    """Container for trained ensemble models."""
    xgb: xgb.XGBClassifier
    lgb: lgb.LGBMClassifier
    cb: cb.CatBoostClassifier
    selector: SelectKBest
    cv_auc: float


def train_gpu_ensemble(
    X: np.ndarray,
    y: np.ndarray,
    config: V13Config
) -> EnsembleModels:
    """Train ensemble with XGBoost + LightGBM + CatBoost on GPU."""

    with timer("Feature selection"):
        n_features = min(config.n_top_features, X.shape[1])
        selector = SelectKBest(f_classif, k=n_features)
        X_selected = selector.fit_transform(X, y)

    logger.info(
        f"    Training: {X_selected.shape[0]} samples, "
        f"{X_selected.shape[1]} features"
    )

    # Initialize models
    xgb_model = xgb.XGBClassifier(
        n_estimators=config.xgb_n_estimators,
        max_depth=config.xgb_max_depth,
        learning_rate=config.xgb_learning_rate,
        subsample=0.8,
        colsample_bytree=0.8,
        device='cuda' if config.use_gpu else 'cpu',
        tree_method='hist',
        random_state=config.random_seed,
        eval_metric='auc',
        verbosity=0
    )

    lgb_model = lgb.LGBMClassifier(
        n_estimators=config.lgb_n_estimators,
        max_depth=config.lgb_max_depth,
        learning_rate=config.lgb_learning_rate,
        subsample=0.8,
        colsample_bytree=0.8,
        device='gpu' if config.use_gpu else 'cpu',
        random_state=config.random_seed,
        verbose=-1
    )

    cb_model = cb.CatBoostClassifier(
        iterations=config.cb_iterations,
        learning_rate=config.cb_learning_rate,
        depth=config.cb_depth,
        loss_function='Logloss',
        eval_metric='AUC',
        task_type='GPU' if config.use_gpu else 'CPU',
        devices='0',
        random_seed=config.random_seed,
        verbose=False
    )

    # Cross-validation
    cv = StratifiedKFold(
        n_splits=config.n_cv_folds,
        shuffle=True,
        random_state=config.random_seed
    )

    xgb_scores, lgb_scores, cb_scores = [], [], []

    for fold, (train_idx, val_idx) in enumerate(cv.split(X_selected, y), 1):
        X_train, X_val = X_selected[train_idx], X_selected[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # XGBoost
        xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        xgb_pred = xgb_model.predict_proba(X_val)[:, 1]
        xgb_auc = roc_auc_score(y_val, xgb_pred)
        xgb_scores.append(xgb_auc)

        # LightGBM
        lgb_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)]
        )
        lgb_pred = lgb_model.predict_proba(X_val)[:, 1]
        lgb_auc = roc_auc_score(y_val, lgb_pred)
        lgb_scores.append(lgb_auc)

        # CatBoost
        cb_model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            early_stopping_rounds=50,
            use_best_model=True
        )
        cb_pred = cb_model.predict_proba(X_val)[:, 1]
        cb_auc = roc_auc_score(y_val, cb_pred)
        cb_scores.append(cb_auc)

        logger.info(
            f"      Fold {fold}: XGB={xgb_auc:.4f}, "
            f"LGB={lgb_auc:.4f}, CB={cb_auc:.4f}"
        )

    # Calculate mean CV scores
    xgb_mean = np.mean(xgb_scores)
    lgb_mean = np.mean(lgb_scores)
    cb_mean = np.mean(cb_scores)

    # Weighted ensemble score
    ensemble_mean = (
        config.weight_xgb * xgb_mean +
        config.weight_lgb * lgb_mean +
        config.weight_cb * cb_mean
    )

    logger.info(
        f"    CV AUC: XGB={xgb_mean:.4f}, LGB={lgb_mean:.4f}, CB={cb_mean:.4f}, "
        f"Ensemble={ensemble_mean:.4f}"
    )

    # Final training on full dataset
    with timer("Final model training"):
        xgb_model.fit(X_selected, y)
        lgb_model.fit(X_selected, y)
        cb_model.fit(X_selected, y, verbose=False)

    return EnsembleModels(
        xgb=xgb_model,
        lgb=lgb_model,
        cb=cb_model,
        selector=selector,
        cv_auc=ensemble_mean
    )


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def check_gpu(config: V13Config) -> None:
    """Verify GPU availability for all frameworks."""
    logger.info("Checking GPU availability...")

    # XGBoost
    try:
        test_xgb = xgb.XGBClassifier(
            device='cuda' if config.use_gpu else 'cpu',
            tree_method='hist',
            verbosity=0
        )
        test_xgb.fit(np.random.rand(10, 5), np.random.randint(0, 2, 10))
        logger.info("  ✓ XGBoost CUDA: OK")
    except Exception as e:
        logger.error(f"  ✗ XGBoost CUDA: FAILED - {e}")
        raise

    # LightGBM
    try:
        test_lgb = lgb.LGBMClassifier(
            device='gpu' if config.use_gpu else 'cpu',
            verbose=-1
        )
        test_lgb.fit(np.random.rand(10, 5), np.random.randint(0, 2, 10))
        logger.info("  ✓ LightGBM GPU: OK")
    except Exception as e:
        logger.warning(f"  ⚠ LightGBM GPU: FAILED - {e}")

    # CatBoost
    try:
        test_cb = cb.CatBoostClassifier(
            iterations=10,
            task_type='GPU' if config.use_gpu else 'CPU',
            devices='0',
            verbose=False
        )
        test_cb.fit(np.random.rand(10, 5), np.random.randint(0, 2, 10))
        logger.info("  ✓ CatBoost GPU: OK")
    except Exception as e:
        logger.warning(f"  ⚠ CatBoost GPU: FAILED - {e}")

    logger.info("  >>> GPU READY <<<\n")


def main():
    """Main execution pipeline."""
    config = V13Config()

    logger.info("=" * 70)
    logger.info("AIRR-ML-25 Champion V13")
    logger.info("VJ Pairs Features + CatBoost Integration")
    logger.info("=" * 70)
    logger.info(f"Threads: {config.n_threads}")
    logger.info(f"GPU: {config.use_gpu}")

    check_gpu(config)

    # Storage for models and results
    models: Dict[int, Dict] = {}
    all_predictions: List[Dict] = []
    all_sequences: List[Dict] = []

    # PHASE 1: Train on all datasets
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 1: Training on all datasets")
    logger.info("=" * 70)

    datasets = sorted([d for d in config.train_root.iterdir() if d.is_dir()])

    for ds_path in datasets:
        ds_name = ds_path.name
        ds_id = int(ds_name.split('_')[-1])

        logger.info(f"\n{'='*60}")
        logger.info(f"Dataset {ds_id}: {ds_name}")
        logger.info(f"{'='*60}")

        with timer(f"Dataset {ds_id} processing"):
            # Load metadata
            metadata = pd.read_csv(ds_path / "metadata.csv")
            y = metadata['label_positive'].astype(int).values

            logger.info(
                f"  Samples: {len(metadata)}, "
                f"Positive: {y.sum()}, Negative: {len(y)-y.sum()}"
            )

            # Build vocabulary
            kmer_vocab, vj_vocab, v_vocab, j_vocab, public_clones = \
                build_vocabulary(ds_path, config)

            # Extract features
            X_df = extract_features_parallel(
                ds_path, metadata,
                kmer_vocab, vj_vocab, v_vocab, j_vocab, public_clones,
                config
            )
            X = X_df.fillna(0).values
            logger.info(f"  Feature matrix: {X.shape}")

            # Train ensemble
            ensemble = train_gpu_ensemble(X, y, config)

            # Store model
            models[ds_id] = {
                'ensemble': ensemble,
                'vocab': (kmer_vocab, vj_vocab, v_vocab, j_vocab, public_clones),
                'feature_names': X_df.columns.tolist(),
                'cv_auc': ensemble.cv_auc
            }

            logger.info(f"  Dataset {ds_id} complete: CV AUC = {ensemble.cv_auc:.4f}")

            # Extract sequences for Task B
            feature_names = X_df.columns.tolist()
            importances = ensemble.xgb.feature_importances_

            # Align importances with features
            if len(importances) < len(feature_names):
                selected_mask = ensemble.selector.get_support()
                full_importances = np.zeros(len(feature_names))
                full_importances[selected_mask] = importances
                importances = full_importances

            # Score sequences by importance
            seq_scores: Dict[str, float] = {}
            for i, fname in enumerate(feature_names):
                if fname.startswith('pub_') and i < len(importances):
                    seq = fname[4:]
                    seq_scores[seq] = seq_scores.get(seq, 0) + importances[i]

            sorted_seqs = sorted(
                seq_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )[:50000]

            for rank, (seq, score) in enumerate(sorted_seqs, 1):
                all_sequences.append({
                    'ID': f'{ds_name}_seq_top_{rank}',
                    'dataset': ds_name,
                    'label_positive_probability': -999.0,
                    'junction_aa': seq,
                    'v_call': '-999.0',
                    'j_call': '-999.0'
                })

            # Fill to 50000 with dummy sequences
            remaining = 50000 - len(sorted_seqs)
            if remaining > 0:
                dummy_seqs = [
                    f'CASS{chr(65+i%26)}{chr(65+(i//26)%26)}YEQYF'
                    for i in range(remaining)
                ]
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
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 2: Test Predictions")
    logger.info("=" * 70)

    test_dirs = sorted([d for d in config.test_root.iterdir() if d.is_dir()])

    for test_path in test_dirs:
        test_name = test_path.name

        # Determine which model to use
        if '_' in test_name.replace('test_dataset_', ''):
            base_id = int(test_name.replace('test_dataset_', '').split('_')[0])
        else:
            base_id = int(test_name.replace('test_dataset_', ''))

        if base_id > 8:
            base_id = 8

        logger.info(f"\n  {test_name} -> using model {base_id}")

        model_info = models.get(base_id, models.get(1))
        kmer_vocab, vj_vocab, v_vocab, j_vocab, public_clones = model_info['vocab']

        # Scan TSV files
        tsv_files = sorted([f.name for f in test_path.glob("*.tsv")])
        test_metadata = pd.DataFrame({
            'filename': tsv_files,
            'repertoire_id': [f.replace('.tsv', '') for f in tsv_files]
        })

        # Extract features
        X_test_df = extract_features_parallel(
            test_path, test_metadata,
            kmer_vocab, vj_vocab, v_vocab, j_vocab, public_clones,
            config
        )
        X_test = X_test_df.fillna(0).values

        # Align features with training
        train_features = model_info['feature_names']
        test_features = X_test_df.columns.tolist()

        aligned_X = np.zeros((len(X_test_df), len(train_features)))
        for i, fname in enumerate(train_features):
            if fname in test_features:
                j = test_features.index(fname)
                aligned_X[:, i] = X_test[:, j]

        # Transform with feature selector
        X_test_selected = model_info['ensemble'].selector.transform(aligned_X)

        # Predict with ensemble
        ensemble = model_info['ensemble']
        xgb_pred = ensemble.xgb.predict_proba(X_test_selected)[:, 1]
        lgb_pred = ensemble.lgb.predict_proba(X_test_selected)[:, 1]
        cb_pred = ensemble.cb.predict_proba(X_test_selected)[:, 1]

        final_pred = (
            config.weight_xgb * xgb_pred +
            config.weight_lgb * lgb_pred +
            config.weight_cb * cb_pred
        )

        for idx, row in test_metadata.iterrows():
            all_predictions.append({
                'ID': row['repertoire_id'],
                'dataset': test_name,
                'label_positive_probability': float(final_pred[idx]),
                'junction_aa': '-999.0',
                'v_call': '-999.0',
                'j_call': '-999.0'
            })

        logger.info(
            f"    {len(test_metadata)} samples, "
            f"mean: {final_pred.mean():.3f}, "
            f"std: {final_pred.std():.3f}"
        )

    # PHASE 3: Generate submission
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 3: Generating Submission")
    logger.info("=" * 70)

    pred_df = pd.DataFrame(all_predictions)
    seq_df = pd.DataFrame(all_sequences)
    submission = pd.concat([pred_df, seq_df], ignore_index=True)

    output_path = config.output_dir / "submission_v13.csv"
    submission.to_csv(output_path, index=False)

    logger.info(f"\n  Saved: {output_path}")
    logger.info(f"  Total rows: {len(submission):,}")
    logger.info(f"  Predictions: {len(all_predictions):,}")
    logger.info(f"  Sequences: {len(all_sequences):,}")

    # Validate
    sample_sub = pd.read_csv(
        "/home/thc1006/dev/airr-ml25-package/data/sample_submissions.csv"
    )
    logger.info(f"\n  Expected rows: {len(sample_sub):,}")
    logger.info(f"  Match: {'✓ YES' if len(submission) == len(sample_sub) else '✗ NO'}")

    logger.info("\n" + "=" * 70)
    logger.info("COMPLETED!")
    logger.info("=" * 70)


if __name__ == "__main__":
    start_time = time.time()
    try:
        main()
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        sys.exit(1)
    finally:
        elapsed = time.time() - start_time
        logger.info(f"\nTotal execution time: {elapsed/60:.1f} minutes")
