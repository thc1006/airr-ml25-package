#!/usr/bin/env python3
"""
AIRR-ML-25 Champion v8 - Ultimate Winning Strategy
===================================================
Based on deep analysis of V5 success and V6/V7 failures.

Key Strategy:
1. PRESERVE V5's exact feature extraction (proven to work)
2. Multi-seed ensemble (3 different seeds averaged)
3. Add CatBoost for model diversity
4. Improved Task B with feature importance mapping
5. Robust sequence validation

Target: Beat 0.84590 (1st place SajayR)
Current best: 0.74006 (V5)

Analysis Results:
- Task A contributes ~95% of final score
- Task B contributes ~5% of final score
- V5's feature extraction is critical - don't change it!
- V7 failed due to CV overfitting (high CV, low LB)
"""

import os
import gc
import re
import warnings
from pathlib import Path
from collections import Counter
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import entropy
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from joblib import Parallel, delayed

import xgboost as xgb
import lightgbm as lgb

# Try CatBoost
try:
    from catboost import CatBoostClassifier, Pool
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

warnings.filterwarnings('ignore')

# ============================================================================
# Configuration - EXACTLY same as V5 (proven to work)
# ============================================================================
class Config:
    TRAIN_ROOT = Path('./data/train_datasets/train_datasets')
    TEST_ROOT = Path('./data/test_datasets/test_datasets')
    SAMPLE_SUBMISSION = Path('./data/sample_submissions.csv')
    SUBMISSION_DIR = Path('./submissions')
    CHECKPOINT_DIR = Path('./checkpoints_v8')

    # Feature settings - SAME as V5
    K_LIST = [3, 4]
    TOP_KMER = 500
    MAX_SEQUENCES_PER_FILE = 50000

    # Public clone settings - SAME as V5 (V7 changed these and failed!)
    PUB_MAX_FILES = 30
    PUB_MIN_FREQ = 0.15  # V7 used 0.05 and failed
    PUB_ENRICH = 5.0
    PUB_TOP_N = {1: 2000, 2: 2000, 3: 2000, 4: 2000, 5: 2000,
                 6: 2000, 7: 5000, 8: 3000}

    # Training settings
    N_SPLITS = 5
    EARLY_STOP = 100

    # Multi-seed ensemble - KEY IMPROVEMENT
    RANDOM_SEEDS = [42, 123, 456]  # 3 seeds for stability

    # Per-dataset class weights - SAME as V5
    SCALE_POS_WEIGHT = {
        1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0,
        7: 5.0,
        8: 2.0
    }

    # Top K sequences for Task B
    TOP_K_SEQUENCES = 50000

# Amino acid properties - SAME as V5
AA_PROPERTIES = {
    'A': {'hydro': 1.8, 'vol': 88.6, 'charge': 0, 'polar': 0},
    'R': {'hydro': -4.5, 'vol': 173.4, 'charge': 1, 'polar': 1},
    'N': {'hydro': -3.5, 'vol': 114.1, 'charge': 0, 'polar': 1},
    'D': {'hydro': -3.5, 'vol': 111.1, 'charge': -1, 'polar': 1},
    'C': {'hydro': 2.5, 'vol': 108.5, 'charge': 0, 'polar': 0},
    'Q': {'hydro': -3.5, 'vol': 143.8, 'charge': 0, 'polar': 1},
    'E': {'hydro': -3.5, 'vol': 138.4, 'charge': -1, 'polar': 1},
    'G': {'hydro': -0.4, 'vol': 60.1, 'charge': 0, 'polar': 0},
    'H': {'hydro': -3.2, 'vol': 153.2, 'charge': 0.5, 'polar': 1},
    'I': {'hydro': 4.5, 'vol': 166.7, 'charge': 0, 'polar': 0},
    'L': {'hydro': 3.8, 'vol': 166.7, 'charge': 0, 'polar': 0},
    'K': {'hydro': -3.9, 'vol': 168.6, 'charge': 1, 'polar': 1},
    'M': {'hydro': 1.9, 'vol': 162.9, 'charge': 0, 'polar': 0},
    'F': {'hydro': 2.8, 'vol': 189.9, 'charge': 0, 'polar': 0},
    'P': {'hydro': -1.6, 'vol': 112.7, 'charge': 0, 'polar': 0},
    'S': {'hydro': -0.8, 'vol': 89.0, 'charge': 0, 'polar': 1},
    'T': {'hydro': -0.7, 'vol': 116.1, 'charge': 0, 'polar': 1},
    'W': {'hydro': -0.9, 'vol': 227.8, 'charge': 0, 'polar': 0},
    'Y': {'hydro': -1.3, 'vol': 193.6, 'charge': 0, 'polar': 1},
    'V': {'hydro': 4.2, 'vol': 140.0, 'charge': 0, 'polar': 0},
}

VALID_AA = set(AA_PROPERTIES.keys())
VALID_AA_PATTERN = re.compile(r'^[ACDEFGHIKLMNPQRSTVWY]+$')

# ============================================================================
# Data Utilities - EXACTLY same as V5
# ============================================================================
def dataset_id_from_name(name: str) -> int:
    """Extract dataset ID from name like 'train_dataset_7' -> 7"""
    for part in name.replace('_', ' ').split():
        if part.isdigit():
            return int(part)
    return 1

def read_repertoire(tsv_path: Path, max_seqs: Optional[int] = None) -> pd.DataFrame:
    """Read a repertoire TSV file with optional sampling - SAME as V5."""
    cols = ['junction_aa', 'v_call', 'j_call', 'templates']
    try:
        header = pd.read_csv(tsv_path, sep='\t', nrows=0)
        usecols = [c for c in cols if c in header.columns]
        df = pd.read_csv(tsv_path, sep='\t', usecols=usecols)
    except Exception:
        return pd.DataFrame(columns=cols)

    if max_seqs and len(df) > max_seqs:
        if 'templates' in df.columns:
            weights = pd.to_numeric(df['templates'], errors='coerce').fillna(1.0).values
            s = weights.sum()
            if s > 0:
                weights = weights / s
                idx = np.random.choice(len(df), max_seqs, replace=False, p=weights)
                df = df.iloc[idx].reset_index(drop=True)
            else:
                df = df.sample(n=max_seqs, random_state=42).reset_index(drop=True)
        else:
            df = df.sample(n=max_seqs, random_state=42).reset_index(drop=True)

    for col in cols:
        if col not in df.columns:
            df[col] = '' if col != 'templates' else 1.0

    df['junction_aa'] = df['junction_aa'].fillna('').astype(str)
    df['templates'] = pd.to_numeric(df['templates'], errors='coerce').fillna(1.0)
    return df

# ============================================================================
# Public Clone Mining - EXACTLY same as V5
# ============================================================================
def mine_public_clones(
    dataset_path: Path,
    max_files: int = 20,
    min_freq: float = 0.18,
    enrichment: float = 6.0,
    top_n: int = 2000
) -> Dict[str, Dict]:
    """Mine sequences enriched in positive samples - SAME as V5."""
    meta = pd.read_csv(dataset_path / 'metadata.csv')
    pos_files = meta[meta['label_positive'] == True]['filename'].tolist()[:max_files]
    neg_files = meta[meta['label_positive'] == False]['filename'].tolist()[:max_files]

    if not pos_files:
        return {}

    def get_seqs(files):
        c = Counter()
        for f in files:
            try:
                df = pd.read_csv(dataset_path / f, sep='\t', usecols=['junction_aa'])
                c.update(df['junction_aa'].dropna().unique())
            except Exception:
                pass
        return c

    pos_c = get_seqs(pos_files)
    neg_c = get_seqs(neg_files)

    scored = []
    n_pos = max(1, len(pos_files))
    n_neg = max(1, len(neg_files))

    for seq, count in pos_c.items():
        pf = count / n_pos
        nf = neg_c.get(seq, 0) / n_neg
        if pf >= min_freq and pf > nf * enrichment:
            score = float(np.log((pf + 1e-6) / (nf + 1e-6)))
            scored.append({'seq': seq, 'score': score, 'pos_freq': pf, 'neg_freq': nf})

    scored.sort(key=lambda x: -x['score'])
    return {item['seq']: item for item in scored[:top_n]}

# ============================================================================
# Feature Extraction - EXACTLY same as V5 (CRITICAL!)
# ============================================================================
class FeatureExtractor:
    """Enhanced feature extraction - EXACTLY same as V5."""

    def __init__(self, k_list: List[int] = [3, 4]):
        self.k_list = k_list

    def gene_family(self, gene_call: str) -> str:
        """Extract gene family from call like TRBV20-1*01 -> TRBV20"""
        if not isinstance(gene_call, str) or not gene_call:
            return 'UNK'
        return gene_call.split('*')[0].split('-')[0].upper() or 'UNK'

    def extract_all(
        self,
        df: pd.DataFrame,
        pub_dict: Optional[Dict] = None,
        meta_row: Optional[pd.Series] = None,
        ds_id: int = 1
    ) -> Dict[str, float]:
        """Extract all features from a repertoire - SAME as V5."""
        seqs = df['junction_aa'].dropna().astype(str).tolist()
        seqs = [s for s in seqs if len(s) > 0 and s.isalpha()]
        features: Dict[str, float] = {}

        # 1) K-mers - SAME as V5
        for k in self.k_list:
            c = Counter()
            total = 0
            for seq in seqs:
                if len(seq) < k:
                    continue
                for i in range(len(seq) - k + 1):
                    kmer = seq[i:i + k]
                    if all(ch in AA_PROPERTIES for ch in kmer):
                        c[kmer] += 1
                        total += 1
            if total > 0:
                features.update({f'kmer_{k}_{km}': v / total for km, v in c.items()})

        # 2) Positional k-mers (start/end) - SAME as V5
        k_pos = 3
        start_c, end_c = Counter(), Counter()
        ns, ne = 0, 0
        for seq in seqs:
            if len(seq) < k_pos:
                continue
            sk, ek = seq[:k_pos], seq[-k_pos:]
            if all(ch in AA_PROPERTIES for ch in sk):
                start_c[sk] += 1
                ns += 1
            if all(ch in AA_PROPERTIES for ch in ek):
                end_c[ek] += 1
                ne += 1
        if ns > 0:
            features.update({f'pos_start_{km}': v / ns for km, v in start_c.most_common(30)})
        if ne > 0:
            features.update({f'pos_end_{km}': v / ne for km, v in end_c.most_common(30)})

        # 3) Physicochemical properties - SAME as V5
        hydro, vol, charge = [], [], []
        for seq in seqs:
            h, v, c = 0.0, 0.0, 0.0
            cnt = 0
            for aa in seq:
                if aa in AA_PROPERTIES:
                    h += AA_PROPERTIES[aa]['hydro']
                    v += AA_PROPERTIES[aa]['vol']
                    c += AA_PROPERTIES[aa]['charge']
                    cnt += 1
            if cnt > 0:
                hydro.append(h / cnt)
                vol.append(v / cnt)
                charge.append(c / cnt)

        if hydro:
            features['phys_hydro_mean'] = float(np.mean(hydro))
            features['phys_hydro_std'] = float(np.std(hydro))
            features['phys_vol_mean'] = float(np.mean(vol))
            features['phys_vol_std'] = float(np.std(vol))
            features['phys_charge_mean'] = float(np.mean(charge))

        # 4) V gene families - SAME as V5
        if 'v_call' in df.columns:
            v_fam = df['v_call'].apply(self.gene_family)
            for fam, freq in v_fam.value_counts(normalize=True).head(40).items():
                features[f'v_fam_{fam}'] = float(freq)

        # 5) J gene families - SAME as V5
        if 'j_call' in df.columns:
            j_fam = df['j_call'].apply(self.gene_family)
            for fam, freq in j_fam.value_counts(normalize=True).head(20).items():
                features[f'j_fam_{fam}'] = float(freq)

        # 6) Length statistics - SAME as V5
        lens = [len(s) for s in seqs]
        if lens:
            features['len_mean'] = float(np.mean(lens))
            features['len_std'] = float(np.std(lens))
            features['len_min'] = float(np.min(lens))
            features['len_max'] = float(np.max(lens))
            features['len_p25'] = float(np.percentile(lens, 25))
            features['len_p75'] = float(np.percentile(lens, 75))

        # 7) Diversity metrics - SAME as V5
        features['n_unique_seqs'] = float(len(set(seqs)))
        features['n_total_seqs'] = float(len(seqs))
        if len(seqs) > 0:
            features['diversity_ratio'] = features['n_unique_seqs'] / features['n_total_seqs']

        # Clone size distribution - SAME as V5
        if 'templates' in df.columns:
            temps = df['templates'].values
            if temps.sum() > 0:
                freq = temps / temps.sum()
                features['clone_entropy'] = float(entropy(freq + 1e-10))
                features['clone_gini'] = float(1 - np.sum(freq ** 2))
                features['clone_max_freq'] = float(freq.max())

        # 8) Metadata features - SAME as V5
        if meta_row is not None:
            if 'sex' in meta_row.index:
                sex_val = str(meta_row['sex']).upper()
                features['meta_sex_male'] = 1.0 if sex_val in ['M', 'MALE'] else 0.0

            if ds_id == 7:
                if 'race' in meta_row.index:
                    features['meta_race_white'] = 1.0 if 'white' in str(meta_row['race']).lower() else 0.0
                if 'sequencing_run_id' in meta_row.index:
                    features['meta_run_hash'] = (hash(str(meta_row['sequencing_run_id'])) % 100) / 100.0

            if ds_id == 8:
                for hla in ['A', 'B', 'C', 'DRB1']:
                    if hla in meta_row.index:
                        features[f'meta_hla_{hla}'] = 1.0 if pd.notna(meta_row[hla]) else 0.0

        # 9) Public clone features - SAME as V5
        if pub_dict:
            seq_set = set(seqs)
            hits = [pub_dict[s]['score'] for s in seq_set if s in pub_dict]
            features['pub_score_sum'] = float(sum(hits))
            features['pub_score_max'] = float(max(hits)) if hits else 0.0
            features['pub_hits'] = float(len(hits))
            features['pub_hit_ratio'] = float(len(hits) / len(seq_set)) if seq_set else 0.0

        return features

# ============================================================================
# Multi-Seed Ensemble Trainer - KEY IMPROVEMENT over V5
# ============================================================================
class MultiSeedEnsembleTrainer:
    """XGBoost + LightGBM + CatBoost ensemble with multiple seeds."""

    def __init__(self, use_gpu: bool = True, seeds: List[int] = [42, 123, 456]):
        self.use_gpu = use_gpu
        self.seeds = seeds
        self.models = {seed: {} for seed in seeds}
        self.feature_cols = []
        self.feature_importance = {}

    def select_features_gpu(self, X_df: pd.DataFrame, y: np.ndarray, top_k: int = 500):
        """Select top features using GPU XGBoost - with memory handling."""
        print(f'    Selecting top {top_k} features...', end=' ')
        all_cols = X_df.columns.tolist()

        # For large datasets, use CPU to avoid GPU memory issues
        n_samples = len(X_df)
        use_device = 'cuda' if (self.use_gpu and n_samples < 600) else 'cpu'

        dtrain = xgb.DMatrix(X_df, label=y)

        params = {
            'tree_method': 'hist',
            'device': use_device,
            'max_depth': 4,
            'learning_rate': 0.1,
            'reg_lambda': 1.0,
            'verbosity': 0,
        }

        bst = xgb.train(params, dtrain, num_boost_round=30)
        scores = bst.get_score(importance_type='gain')

        # Store feature importance for Task B
        self.feature_importance = scores.copy()

        sorted_feats = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        selected = [f[0] for f in sorted_feats[:top_k]]

        if len(selected) < top_k:
            remaining = [c for c in all_cols if c not in selected]
            selected.extend(remaining[:top_k - len(selected)])

        print(f'Done ({len(selected)} features)')
        return selected

    def train(self, df: pd.DataFrame, ds_id: int) -> Tuple['MultiSeedEnsembleTrainer', List[str], float]:
        """Train multi-seed ensemble model."""
        y = df['label_positive'].values.astype(np.float32)
        X_df = df.drop(columns=['ID', 'dataset', 'label_positive'], errors='ignore')

        # Feature selection
        self.feature_cols = self.select_features_gpu(X_df, y, top_k=Config.TOP_KMER)
        X = X_df[self.feature_cols].values.astype(np.float32)

        print(f'    Training multi-seed ensemble on {len(X)} samples, {len(self.feature_cols)} features')
        print(f'    Seeds: {self.seeds}')

        # For large datasets, use CPU to avoid GPU memory issues
        use_gpu_for_training = self.use_gpu and len(X) < 600
        if not use_gpu_for_training and self.use_gpu:
            print(f'    Note: Using CPU for large dataset ({len(X)} samples)')

        all_cv_scores = []

        for seed in self.seeds:
            print(f'    --- Seed {seed} ---')

            # XGBoost params
            xgb_params = {
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'max_depth': 6,
                'learning_rate': 0.03,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 15,
                'seed': seed,
                'scale_pos_weight': Config.SCALE_POS_WEIGHT.get(ds_id, 1.0),
                'tree_method': 'hist',
                'device': 'cuda' if use_gpu_for_training else 'cpu',
                'verbosity': 0,
            }

            # LightGBM params
            lgb_params = {
                'objective': 'binary',
                'metric': 'auc',
                'device': 'gpu' if use_gpu_for_training else 'cpu',
                'max_depth': 6,
                'learning_rate': 0.02,
                'num_leaves': 31,
                'min_child_samples': 20,
                'scale_pos_weight': Config.SCALE_POS_WEIGHT.get(ds_id, 1.0),
                'verbosity': -1,
                'force_col_wise': True,
                'seed': seed,
            }

            # Determine n_splits
            pos = int((y == 1).sum())
            neg = int((y == 0).sum())
            min_class = max(2, min(pos, neg))
            n_splits = min(Config.N_SPLITS, min_class)

            kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

            oof_xgb = np.zeros(len(y), dtype=np.float32)
            oof_lgb = np.zeros(len(y), dtype=np.float32)
            best_iters = []

            for tr_idx, va_idx in kf.split(X, y):
                X_tr, X_val = X[tr_idx], X[va_idx]
                y_tr, y_val = y[tr_idx], y[va_idx]

                # XGBoost
                dtr = xgb.DMatrix(X_tr, label=y_tr)
                dval = xgb.DMatrix(X_val, label=y_val)
                bst = xgb.train(
                    xgb_params, dtr,
                    num_boost_round=1000,
                    evals=[(dval, 'v')],
                    early_stopping_rounds=Config.EARLY_STOP,
                    verbose_eval=False,
                )
                oof_xgb[va_idx] = bst.predict(dval)
                best_iters.append(int(bst.best_iteration or 0))

                # LightGBM
                lgb_tr = lgb.Dataset(X_tr, label=y_tr)
                lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_tr)
                lgb_bst = lgb.train(
                    lgb_params, lgb_tr,
                    num_boost_round=1000,
                    valid_sets=[lgb_val],
                    callbacks=[lgb.early_stopping(Config.EARLY_STOP, verbose=False)],
                )
                oof_lgb[va_idx] = lgb_bst.predict(X_val)

            # Ensemble OOF
            oof_ensemble = 0.5 * oof_xgb + 0.5 * oof_lgb
            cv_score = roc_auc_score(y, oof_ensemble)
            all_cv_scores.append(cv_score)
            print(f'      CV AUC: {cv_score:.4f}')

            # Train final models for this seed
            rounds = max(int(np.mean(best_iters)) + 50, 100)
            self.models[seed]['xgb'] = xgb.train(xgb_params, xgb.DMatrix(X, label=y), num_boost_round=rounds)
            self.models[seed]['lgb'] = lgb.train(lgb_params, lgb.Dataset(X, label=y), num_boost_round=800)

            # CatBoost if available
            if HAS_CATBOOST:
                cat_model = CatBoostClassifier(
                    iterations=500,
                    learning_rate=0.03,
                    depth=6,
                    task_type='GPU' if use_gpu_for_training else 'CPU',
                    random_seed=seed,
                    verbose=False,
                    scale_pos_weight=Config.SCALE_POS_WEIGHT.get(ds_id, 1.0),
                )
                cat_model.fit(X, y)
                self.models[seed]['cat'] = cat_model

        mean_cv = float(np.mean(all_cv_scores))
        print(f'    Mean CV AUC across seeds: {mean_cv:.4f}')

        return self, self.feature_cols, mean_cv

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities with multi-seed averaging."""
        X = X.astype(np.float32)
        predictions = []

        for seed in self.seeds:
            p_xgb = self.models[seed]['xgb'].predict(xgb.DMatrix(X))
            p_lgb = self.models[seed]['lgb'].predict(X)

            if HAS_CATBOOST and 'cat' in self.models[seed]:
                p_cat = self.models[seed]['cat'].predict_proba(X)[:, 1]
                seed_pred = (p_xgb + p_lgb + p_cat) / 3.0
            else:
                seed_pred = 0.5 * p_xgb + 0.5 * p_lgb

            predictions.append(seed_pred)

        # Average across seeds
        return np.mean(predictions, axis=0)

# ============================================================================
# Parallel Processing Helpers
# ============================================================================
def process_file_parallel(row, path: Path, ds_id: int, pub_dict: Dict, extractor: FeatureExtractor):
    """Process a single repertoire file - SAME as V5."""
    try:
        df = read_repertoire(path / row['filename'], Config.MAX_SEQUENCES_PER_FILE)
        feats = extractor.extract_all(df, pub_dict, row, ds_id)
        return {
            **feats,
            'ID': row.get('repertoire_id', Path(row['filename']).stem),
            'label_positive': int(row['label_positive']),
            'dataset': path.name,
        }
    except Exception:
        return None

def process_test_file_parallel(row, path: Path, ds_id: int, pub_dict: Dict, extractor: FeatureExtractor):
    """Process a test file from metadata row."""
    try:
        df = read_repertoire(path / row['filename'], Config.MAX_SEQUENCES_PER_FILE)
        feats = extractor.extract_all(df, pub_dict, row, ds_id)
        return {
            **feats,
            'ID': row.get('repertoire_id', Path(row['filename']).stem),
            'dataset': path.name,
        }
    except Exception:
        return None


def process_test_tsv_parallel(tsv_path: Path, test_path: Path, ds_id: int, pub_dict: Dict, extractor: FeatureExtractor):
    """Process a test TSV file directly (for test datasets without metadata.csv) - SAME as V5."""
    try:
        df = read_repertoire(tsv_path, Config.MAX_SEQUENCES_PER_FILE)
        feats = extractor.extract_all(df, pub_dict, None, ds_id)
        return {
            **feats,
            'ID': tsv_path.stem,  # Filename without extension is the ID
            'dataset': test_path.name,
        }
    except Exception:
        return None

# ============================================================================
# Task B: Sequence Identification - IMPROVED with feature importance
# ============================================================================
def identify_sequences_v8(
    dataset_path: Path,
    pub_dict: Dict,
    feature_importance: Dict,
    top_k: int = 50000
) -> pd.DataFrame:
    """Identify top disease-associated sequences using feature importance."""
    meta = pd.read_csv(dataset_path / 'metadata.csv')
    ds_name = dataset_path.name

    # Extract important k-mers from feature importance
    kmer_importance = {}
    for feat, score in feature_importance.items():
        if feat.startswith('kmer_'):
            parts = feat.split('_')
            if len(parts) >= 3:
                kmer = '_'.join(parts[2:])  # Handle kmers with underscores
                if len(kmer) in [3, 4]:
                    kmer_importance[kmer] = score

    # Score sequences
    seq_scores = {}
    seq_info = {}

    # 1) Public clones get base score
    for seq, info in pub_dict.items():
        seq_scores[seq] = info['score'] * 2.0  # Weight public clones

    # 2) Sample from positive repertoires and score by k-mer importance
    pos_files = meta[meta['label_positive'] == True]['filename'].tolist()

    for f in pos_files[:50]:
        try:
            df = pd.read_csv(dataset_path / f, sep='\t',
                           usecols=['junction_aa', 'v_call', 'j_call'])
            for _, row in df.iterrows():
                seq = str(row['junction_aa'])
                # Validate sequence
                if not seq or not VALID_AA_PATTERN.match(seq):
                    continue

                # Calculate k-mer-based importance score
                kmer_score = 0.0
                for k in [3, 4]:
                    for i in range(len(seq) - k + 1):
                        kmer = seq[i:i + k]
                        if kmer in kmer_importance:
                            kmer_score += kmer_importance[kmer]

                # Combine with public clone score
                if seq in seq_scores:
                    seq_scores[seq] = max(seq_scores[seq], kmer_score + seq_scores[seq])
                else:
                    seq_scores[seq] = kmer_score

                # Store v_call/j_call
                if seq not in seq_info:
                    v_call = str(row.get('v_call', ''))
                    j_call = str(row.get('j_call', ''))
                    if v_call and v_call != 'nan':
                        seq_info[seq] = {'v_call': v_call, 'j_call': j_call}

        except Exception:
            continue

        if len(seq_scores) >= top_k * 3:
            break

    # Sort by score and filter valid sequences
    valid_sorted_seqs = [(s, sc) for s, sc in sorted(seq_scores.items(), key=lambda x: -x[1])
                         if VALID_AA_PATTERN.match(s)]

    # Get V/J calls for sequences that don't have them
    seqs_need_calls = set(s for s, _ in valid_sorted_seqs[:top_k] if s not in seq_info)
    if seqs_need_calls:
        for f in meta['filename'].tolist()[:30]:
            try:
                df = pd.read_csv(dataset_path / f, sep='\t',
                               usecols=['junction_aa', 'v_call', 'j_call'])
                for _, row in df.iterrows():
                    seq = str(row['junction_aa'])
                    if seq in seqs_need_calls and seq not in seq_info:
                        v_call = str(row.get('v_call', ''))
                        j_call = str(row.get('j_call', ''))
                        if v_call and v_call != 'nan':
                            seq_info[seq] = {'v_call': v_call, 'j_call': j_call}
                            seqs_need_calls.remove(seq)
            except Exception:
                continue
            if not seqs_need_calls:
                break

    # Build result DataFrame
    results = []
    for i, (seq, score) in enumerate(valid_sorted_seqs[:top_k]):
        info = seq_info.get(seq, {})
        v_call = info.get('v_call', '-999.0')
        j_call = info.get('j_call', '-999.0')

        # Ensure valid format
        if not v_call or v_call == 'nan' or v_call == 'None':
            v_call = '-999.0'
        if not j_call or j_call == 'nan' or j_call == 'None':
            j_call = '-999.0'

        results.append({
            'ID': f'{ds_name}_seq_top_{i+1}',
            'dataset': ds_name,
            'label_positive_probability': -999.0,
            'junction_aa': seq,
            'v_call': v_call,
            'j_call': j_call,
        })

    # Pad if needed
    while len(results) < top_k:
        results.append({
            'ID': f'{ds_name}_seq_top_{len(results)+1}',
            'dataset': ds_name,
            'label_positive_probability': -999.0,
            'junction_aa': 'CASSXXXXF',
            'v_call': '-999.0',
            'j_call': '-999.0',
        })

    return pd.DataFrame(results[:top_k])

# ============================================================================
# Main Pipeline
# ============================================================================
def main():
    print('='*70)
    print('AIRR-ML-25 Champion v8 - Ultimate Winning Strategy')
    print('='*70)
    print('Strategy:')
    print('  - PRESERVE V5 feature extraction (proven to work)')
    print('  - Multi-seed ensemble (3 seeds)')
    print(f'  - CatBoost: {"Available" if HAS_CATBOOST else "Not available"}')
    print('  - Improved Task B with feature importance')
    print('='*70)

    # Check GPU
    import subprocess
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, timeout=5)
        use_gpu = result.returncode == 0
    except Exception:
        use_gpu = False

    print(f'GPU available: {use_gpu}')

    # Create directories
    Config.SUBMISSION_DIR.mkdir(exist_ok=True)
    Config.CHECKPOINT_DIR.mkdir(exist_ok=True)

    # Discover datasets
    train_sets = sorted([d.name for d in Config.TRAIN_ROOT.glob('train_dataset_*')])
    test_sets = sorted([d.name for d in Config.TEST_ROOT.glob('test_dataset_*')])
    print(f'Train datasets: {train_sets}')
    print(f'Test datasets: {test_sets}')

    # Storage
    trainers = {}
    feature_cols_map = {}
    public_clones = {}
    feature_importance_map = {}
    all_test_predictions = []
    cv_scores = []

    extractor = FeatureExtractor(Config.K_LIST)

    # ======== TRAINING ========
    print('\n' + '='*70)
    print('TRAINING PHASE')
    print('='*70)

    for ds_name in train_sets:
        ds_path = Config.TRAIN_ROOT / ds_name
        ds_id = dataset_id_from_name(ds_name)
        print(f'\n[{ds_name}] Training (id={ds_id})')

        # Mine public clones
        print('  Mining public clones...')
        pub_dict = mine_public_clones(
            ds_path,
            max_files=Config.PUB_MAX_FILES,
            min_freq=Config.PUB_MIN_FREQ,
            enrichment=Config.PUB_ENRICH,
            top_n=Config.PUB_TOP_N.get(ds_id, 2000),
        )
        public_clones[ds_name] = pub_dict
        print(f'    Found {len(pub_dict)} public clones')

        # Extract features
        print('  Extracting features...')
        meta = pd.read_csv(ds_path / 'metadata.csv')

        results = Parallel(n_jobs=8, backend='loky')(
            delayed(process_file_parallel)(row, ds_path, ds_id, pub_dict, extractor)
            for _, row in tqdm(meta.iterrows(), total=len(meta), desc='    ')
        )

        df_train = pd.DataFrame([r for r in results if r is not None])
        print(f'    Extracted features for {len(df_train)} repertoires')

        # Train ensemble
        print('  Training multi-seed ensemble...')
        trainer = MultiSeedEnsembleTrainer(use_gpu=use_gpu, seeds=Config.RANDOM_SEEDS)
        trainer, fcols, cv_auc = trainer.train(df_train, ds_id)

        trainers[ds_name] = trainer
        feature_cols_map[ds_name] = fcols
        feature_importance_map[ds_name] = trainer.feature_importance.copy()
        cv_scores.append(cv_auc)

        print(f'    Dataset {ds_id} CV AUC: {cv_auc:.4f}')
        gc.collect()

    print(f'\n{"="*70}')
    print(f'Mean CV AUC: {np.mean(cv_scores):.4f}')
    print(f'{"="*70}')

    # ======== PREDICTION ========
    print('\n' + '='*70)
    print('PREDICTION PHASE')
    print('='*70)

    for test_name in test_sets:
        test_path = Config.TEST_ROOT / test_name
        test_id = dataset_id_from_name(test_name)
        train_name = f'train_dataset_{test_id}'

        print(f'\n[{test_name}] Predicting using {train_name}')

        if train_name not in trainers:
            print(f'  WARNING: No trainer for {train_name}, skipping')
            continue

        trainer = trainers[train_name]
        fcols = feature_cols_map[train_name]
        pub_dict = public_clones[train_name]

        # Extract test features - iterate over TSV files directly (no metadata.csv in test sets)
        tsv_files = sorted(test_path.glob('*.tsv'))

        results = Parallel(n_jobs=8, backend='loky')(
            delayed(process_test_tsv_parallel)(f, test_path, test_id, pub_dict, extractor)
            for f in tqdm(tsv_files, total=len(tsv_files), desc='    ')
        )

        df_test = pd.DataFrame([r for r in results if r is not None])

        # Align features
        for col in fcols:
            if col not in df_test.columns:
                df_test[col] = 0.0

        X_test = df_test[fcols].values.astype(np.float32)
        preds = trainer.predict(X_test)

        sub_part = df_test[['ID']].copy()
        sub_part['dataset'] = test_name
        sub_part['label_positive_probability'] = preds
        sub_part['junction_aa'] = '-999.0'
        sub_part['v_call'] = '-999.0'
        sub_part['j_call'] = '-999.0'

        all_test_predictions.append(sub_part)
        print(f'    Predicted {len(sub_part)} repertoires')
        gc.collect()

    # Combine predictions
    task_a = pd.concat(all_test_predictions, ignore_index=True)
    task_a = task_a.sort_values(['dataset', 'ID']).reset_index(drop=True)

    # ======== SEQUENCE IDENTIFICATION (Task B) ========
    print('\n' + '='*70)
    print('SEQUENCE IDENTIFICATION (Task B)')
    print('='*70)

    task_b_parts = []
    for ds_name in train_sets:
        ds_path = Config.TRAIN_ROOT / ds_name
        pub_dict = public_clones[ds_name]
        feat_imp = feature_importance_map.get(ds_name, {})

        print(f'  [{ds_name}] Identifying sequences...')
        seq_df = identify_sequences_v8(
            ds_path, pub_dict, feat_imp,
            top_k=Config.TOP_K_SEQUENCES,
        )
        task_b_parts.append(seq_df)
        print(f'    Generated {len(seq_df)} sequences')

    task_b = pd.concat(task_b_parts, ignore_index=True)

    # Combine Task A and Task B
    submission = pd.concat([task_a, task_b], ignore_index=True)
    submission = submission[['ID', 'dataset', 'label_positive_probability', 'junction_aa', 'v_call', 'j_call']]

    # Validate
    print(f'\nSubmission shape: {submission.shape}')
    print(f'Task A rows: {len(task_a)}')
    print(f'Task B rows: {len(task_b)}')

    # Final validation - ensure no invalid sequences
    task_b_seqs = submission[submission['junction_aa'] != '-999.0']['junction_aa']
    invalid_seqs = [s for s in task_b_seqs if not VALID_AA_PATTERN.match(str(s))]
    if invalid_seqs:
        print(f'WARNING: Found {len(invalid_seqs)} invalid sequences, fixing...')
        for idx in submission[submission['junction_aa'].isin(invalid_seqs)].index:
            submission.loc[idx, 'junction_aa'] = 'CASSXXXXF'

    # Save
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = Config.SUBMISSION_DIR / f'v8_submission_{timestamp}.csv'
    submission.to_csv(out_path, index=False)

    print(f'\nSaved submission: {out_path}')
    print(f'Expected rows: 404213, Actual: {len(submission)}')

    # Save CV scores
    print('\n' + '='*70)
    print('RESULTS SUMMARY')
    print('='*70)
    for i, (ds, score) in enumerate(zip(train_sets, cv_scores)):
        print(f'  {ds}: CV AUC = {score:.4f}')
    print(f'\n  Mean CV AUC: {np.mean(cv_scores):.4f}')
    print('='*70)

    return str(out_path)

if __name__ == '__main__':
    main()
