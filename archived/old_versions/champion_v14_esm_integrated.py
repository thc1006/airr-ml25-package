#!/usr/bin/env python3
"""
Champion V14 - ESM2 + Traditional Features Integration
========================================================
Combines ESM2 protein embeddings with traditional k-mer and VJ features.

Architecture:
- ESM2 features: 1280 dimensions (mean, std, max, q75 of 320-dim embeddings)
- Traditional features: ~1000 dimensions (k-mers, VJ pairs, diversity)
- Total: ~2280 dimensions
- Models: XGBoost + LightGBM ensemble with GPU acceleration
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple, Optional
import warnings
import pickle
import logging
warnings.filterwarnings('ignore')

from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# ML
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import lightgbm as lgb

# ESM2 extractor
from champion_v13_esm2 import ESM2FeatureExtractor, ESM2Config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
TRAIN_ROOT = Path("/home/thc1006/dev/airr-ml25-package/data/train_datasets/train_datasets")
TEST_ROOT = Path("/home/thc1006/dev/airr-ml25-package/data/test_datasets/test_datasets")
OUTPUT_DIR = Path("/home/thc1006/dev/airr-ml25-package/submissions")
ESM2_CHECKPOINT_DIR = Path("/home/thc1006/dev/airr-ml25-package/checkpoints_esm2")
OUTPUT_DIR.mkdir(exist_ok=True)

N_THREADS = 8


# ============================================================================
# TRADITIONAL FEATURE EXTRACTION (from champion_v12)
# ============================================================================

def extract_traditional_features(
    tsv_path: Path,
    kmer_vocab: List,
    vj_vocab: List,
    public_clones: List
) -> Optional[Dict]:
    """Extract traditional features (k-mers, VJ, diversity)"""
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
        logger.error(f"Error extracting traditional features: {e}")
        return None


def build_vocabulary(dataset_path: Path, n_samples: int = 100) -> Tuple[List, List, List]:
    """Build k-mer, VJ, and public clone vocabularies"""
    metadata = pd.read_csv(dataset_path / "metadata.csv")

    # Sample files
    sample_files = metadata['filename'].sample(min(n_samples, len(metadata))).tolist()

    all_kmers = Counter()
    all_vj = Counter()
    all_seqs = Counter()

    for fname in tqdm(sample_files, desc="Building vocabulary"):
        try:
            df = pd.read_csv(dataset_path / fname, sep='\t')

            # K-mers
            for seq in df['junction_aa'].dropna():
                if isinstance(seq, str) and len(seq) >= 3:
                    for i in range(len(seq) - 2):
                        all_kmers[seq[i:i+3]] += 1

            # VJ pairs
            for v, j in zip(df['v_call'].fillna('UNK'), df['j_call'].fillna('UNK')):
                all_vj[(v, j)] += 1

            # Sequences (for public clones)
            all_seqs.update(df['junction_aa'].dropna())

        except Exception:
            continue

    # Top features
    kmer_vocab = [k for k, _ in all_kmers.most_common(500)]
    vj_vocab = [vj for vj, _ in all_vj.most_common(100)]
    public_clones = [seq for seq, count in all_seqs.items() if count >= 3][:100]

    logger.info(f"Vocabulary: {len(kmer_vocab)} k-mers, {len(vj_vocab)} VJ pairs, {len(public_clones)} public clones")

    return kmer_vocab, vj_vocab, public_clones


def extract_dataset_traditional_features(
    dataset_path: Path,
    kmer_vocab: List,
    vj_vocab: List,
    public_clones: List
) -> pd.DataFrame:
    """Extract traditional features for all repertoires"""
    metadata = pd.read_csv(dataset_path / "metadata.csv")

    results = []
    with ThreadPoolExecutor(max_workers=N_THREADS) as executor:
        futures = []
        for idx, row in metadata.iterrows():
            tsv_path = dataset_path / row['filename']
            future = executor.submit(
                extract_traditional_features,
                tsv_path,
                kmer_vocab,
                vj_vocab,
                public_clones
            )
            futures.append((row['repertoire_id'], future))

        for rep_id, future in tqdm(futures, desc="Extracting traditional features"):
            features = future.result()
            if features is not None:
                features['repertoire_id'] = rep_id
                results.append(features)

    return pd.DataFrame(results)


# ============================================================================
# ESM2 FEATURE LOADING
# ============================================================================

def load_esm2_features(dataset_id: str) -> pd.DataFrame:
    """Load pre-computed ESM2 features"""
    checkpoint_path = ESM2_CHECKPOINT_DIR / f"esm2_features_{dataset_id}.pkl"

    if not checkpoint_path.exists():
        logger.warning(f"ESM2 checkpoint not found: {checkpoint_path}")
        return None

    with open(checkpoint_path, 'rb') as f:
        df = pickle.load(f)

    logger.info(f"Loaded ESM2 features: {df.shape}")
    return df


# ============================================================================
# FEATURE INTEGRATION
# ============================================================================

def integrate_features(
    traditional_df: pd.DataFrame,
    esm2_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge traditional and ESM2 features.

    Args:
        traditional_df: DataFrame with traditional features
        esm2_df: DataFrame with ESM2 features

    Returns:
        Merged DataFrame
    """
    if esm2_df is None:
        logger.warning("No ESM2 features - using traditional only")
        return traditional_df

    # Merge on repertoire_id
    merged = traditional_df.merge(
        esm2_df,
        on='repertoire_id',
        how='left'
    )

    # Fill missing ESM2 features with 0
    esm_cols = [c for c in esm2_df.columns if c.startswith('esm_')]
    merged[esm_cols] = merged[esm_cols].fillna(0)

    logger.info(f"Integrated features: {merged.shape}")
    logger.info(f"Traditional: {len([c for c in traditional_df.columns if c != 'repertoire_id'])}")
    logger.info(f"ESM2: {len(esm_cols)}")

    return merged


# ============================================================================
# TRAINING
# ============================================================================

def train_models(X_train, y_train, X_val, y_val):
    """Train XGBoost and LightGBM with GPU"""

    # XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    xgb_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'tree_method': 'hist',
        'device': 'cuda',
        'max_depth': 6,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'gamma': 0.1
    }

    xgb_model = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=500,
        evals=[(dval, 'val')],
        early_stopping_rounds=50,
        verbose_eval=50
    )

    # LightGBM
    lgb_params = {
        'objective': 'binary',
        'metric': 'auc',
        'device': 'gpu',
        'gpu_platform_id': 0,
        'gpu_device_id': 0,
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5
    }

    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)

    lgb_model = lgb.train(
        lgb_params,
        lgb_train,
        num_boost_round=500,
        valid_sets=[lgb_val],
        callbacks=[
            lgb.early_stopping(50),
            lgb.log_evaluation(50)
        ]
    )

    return xgb_model, lgb_model


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    logger.info("="*60)
    logger.info("Champion V14 - ESM2 Integrated Pipeline")
    logger.info("="*60)

    # Dataset IDs
    train_ids = [f"train_dataset_{i}" for i in range(1, 9)]

    all_features = []
    all_labels = []
    all_rep_ids = []

    # Process each dataset
    for dataset_id in train_ids:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {dataset_id}")
        logger.info(f"{'='*60}")

        dataset_path = TRAIN_ROOT / dataset_id

        if not dataset_path.exists():
            logger.warning(f"Dataset not found: {dataset_path}")
            continue

        # 1. Build vocabulary
        logger.info("Building vocabulary...")
        kmer_vocab, vj_vocab, public_clones = build_vocabulary(dataset_path)

        # 2. Extract traditional features
        logger.info("Extracting traditional features...")
        trad_features = extract_dataset_traditional_features(
            dataset_path,
            kmer_vocab,
            vj_vocab,
            public_clones
        )

        # 3. Load ESM2 features
        logger.info("Loading ESM2 features...")
        esm2_features = load_esm2_features(dataset_id)

        # 4. Integrate features
        logger.info("Integrating features...")
        features = integrate_features(trad_features, esm2_features)

        # 5. Load labels
        metadata = pd.read_csv(dataset_path / "metadata.csv")
        labels = metadata.set_index('repertoire_id')['label_positive']

        # 6. Align features and labels
        features = features[features['repertoire_id'].isin(labels.index)]
        features = features.set_index('repertoire_id').reindex(labels.index).reset_index()

        # Store
        all_features.append(features)
        all_labels.extend(labels.values)
        all_rep_ids.extend(labels.index)

    # Concatenate all datasets
    logger.info("\n" + "="*60)
    logger.info("Concatenating datasets...")
    X_all = pd.concat(all_features, ignore_index=True)
    y_all = np.array(all_labels)

    logger.info(f"Total samples: {len(X_all)}")
    logger.info(f"Total features: {len(X_all.columns) - 1}")  # Exclude repertoire_id
    logger.info(f"Positive ratio: {y_all.mean():.3f}")

    # Feature selection
    logger.info("\nFeature selection...")
    feature_cols = [c for c in X_all.columns if c != 'repertoire_id']
    X_features = X_all[feature_cols].fillna(0).values

    selector = SelectKBest(f_classif, k=min(1000, len(feature_cols)))
    X_selected = selector.fit_transform(X_features, y_all)
    selected_features = np.array(feature_cols)[selector.get_support()]

    logger.info(f"Selected features: {len(selected_features)}")

    # Cross-validation
    logger.info("\nCross-validation training...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    cv_scores = []
    for fold, (train_idx, val_idx) in enumerate(cv.split(X_selected, y_all)):
        logger.info(f"\n--- Fold {fold + 1}/5 ---")

        X_train, X_val = X_selected[train_idx], X_selected[val_idx]
        y_train, y_val = y_all[train_idx], y_all[val_idx]

        # Train models
        xgb_model, lgb_model = train_models(X_train, y_train, X_val, y_val)

        # Predictions
        xgb_pred = xgb_model.predict(xgb.DMatrix(X_val))
        lgb_pred = lgb_model.predict(X_val)

        # Ensemble
        ensemble_pred = (xgb_pred + lgb_pred) / 2

        # Score
        score = roc_auc_score(y_val, ensemble_pred)
        cv_scores.append(score)

        logger.info(f"Fold {fold + 1} AUC: {score:.4f}")

    logger.info("\n" + "="*60)
    logger.info(f"Mean CV AUC: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
    logger.info("="*60)


if __name__ == "__main__":
    main()
