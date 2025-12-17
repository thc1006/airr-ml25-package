#!/usr/bin/env python3
"""
AIRR-ML-25 Champion v5 - Based on 69% Notebook with Enhancements
================================================================
Key improvements over our previous approaches:
1. Public clone mining (sequences enriched in positive samples)
2. Positional k-mers (start/end of sequences)
3. Physicochemical properties
4. XGBoost + LightGBM ensemble with GPU
5. Per-dataset scale_pos_weight
6. Better metadata features

Target: Beat 0.69806 -> aim for 0.75+
"""

import os
import gc
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

warnings.filterwarnings('ignore')

# ============================================================================
# Configuration
# ============================================================================
class Config:
    TRAIN_ROOT = Path('./data/train_datasets/train_datasets')
    TEST_ROOT = Path('./data/test_datasets/test_datasets')
    SAMPLE_SUBMISSION = Path('./data/sample_submissions.csv')
    SUBMISSION_DIR = Path('./submissions')
    CHECKPOINT_DIR = Path('./checkpoints_v5')

    # Feature settings
    K_LIST = [3, 4]
    TOP_KMER = 500  # More than 69% notebook's 400
    MAX_SEQUENCES_PER_FILE = 50000

    # Public clone settings
    PUB_MAX_FILES = 30
    PUB_MIN_FREQ = 0.15
    PUB_ENRICH = 5.0
    PUB_TOP_N = {1: 2000, 2: 2000, 3: 2000, 4: 2000, 5: 2000,
                 6: 2000, 7: 5000, 8: 3000}

    # Training settings
    N_SPLITS = 5
    RANDOM_STATE = 42
    EARLY_STOP = 100

    # Per-dataset class weights
    SCALE_POS_WEIGHT = {
        1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0,
        7: 5.0,  # Imbalanced
        8: 2.0   # Imbalanced
    }

    # Top K sequences for Task B
    TOP_K_SEQUENCES = 50000

# Amino acid properties
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

# ============================================================================
# Data Utilities
# ============================================================================
def dataset_id_from_name(name: str) -> int:
    """Extract dataset ID from name like 'train_dataset_7' -> 7"""
    for part in name.replace('_', ' ').split():
        if part.isdigit():
            return int(part)
    return 1

def read_repertoire(tsv_path: Path, max_seqs: Optional[int] = None) -> pd.DataFrame:
    """Read a repertoire TSV file with optional sampling."""
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
# Public Clone Mining
# ============================================================================
def mine_public_clones(
    dataset_path: Path,
    max_files: int = 20,
    min_freq: float = 0.18,
    enrichment: float = 6.0,
    top_n: int = 2000
) -> Dict[str, Dict]:
    """Mine sequences enriched in positive samples."""
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
# Feature Extraction
# ============================================================================
class FeatureExtractor:
    """Enhanced feature extraction with public clones and positional k-mers."""

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
        """Extract all features from a repertoire."""
        seqs = df['junction_aa'].dropna().astype(str).tolist()
        seqs = [s for s in seqs if len(s) > 0 and s.isalpha()]
        features: Dict[str, float] = {}

        # 1) K-mers
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

        # 2) Positional k-mers (start/end)
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

        # 3) Physicochemical properties
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

        # 4) V gene families
        if 'v_call' in df.columns:
            v_fam = df['v_call'].apply(self.gene_family)
            for fam, freq in v_fam.value_counts(normalize=True).head(40).items():
                features[f'v_fam_{fam}'] = float(freq)

        # 5) J gene families
        if 'j_call' in df.columns:
            j_fam = df['j_call'].apply(self.gene_family)
            for fam, freq in j_fam.value_counts(normalize=True).head(20).items():
                features[f'j_fam_{fam}'] = float(freq)

        # 6) Length statistics
        lens = [len(s) for s in seqs]
        if lens:
            features['len_mean'] = float(np.mean(lens))
            features['len_std'] = float(np.std(lens))
            features['len_min'] = float(np.min(lens))
            features['len_max'] = float(np.max(lens))
            features['len_p25'] = float(np.percentile(lens, 25))
            features['len_p75'] = float(np.percentile(lens, 75))

        # 7) Diversity metrics
        features['n_unique_seqs'] = float(len(set(seqs)))
        features['n_total_seqs'] = float(len(seqs))
        if len(seqs) > 0:
            features['diversity_ratio'] = features['n_unique_seqs'] / features['n_total_seqs']

        # Clone size distribution (if templates available)
        if 'templates' in df.columns:
            temps = df['templates'].values
            if temps.sum() > 0:
                freq = temps / temps.sum()
                features['clone_entropy'] = float(entropy(freq + 1e-10))
                features['clone_gini'] = float(1 - np.sum(freq ** 2))
                features['clone_max_freq'] = float(freq.max())

        # 8) Metadata features (dataset-specific)
        if meta_row is not None:
            if 'sex' in meta_row.index:
                sex_val = str(meta_row['sex']).upper()
                features['meta_sex_male'] = 1.0 if sex_val in ['M', 'MALE'] else 0.0

            # Dataset 7 specific
            if ds_id == 7:
                if 'race' in meta_row.index:
                    features['meta_race_white'] = 1.0 if 'white' in str(meta_row['race']).lower() else 0.0
                if 'sequencing_run_id' in meta_row.index:
                    features['meta_run_hash'] = (hash(str(meta_row['sequencing_run_id'])) % 100) / 100.0

            # Dataset 8 specific (HLA types)
            if ds_id == 8:
                for hla in ['A', 'B', 'C', 'DRB1']:
                    if hla in meta_row.index:
                        features[f'meta_hla_{hla}'] = 1.0 if pd.notna(meta_row[hla]) else 0.0

        # 9) Public clone features
        if pub_dict:
            seq_set = set(seqs)
            hits = [pub_dict[s]['score'] for s in seq_set if s in pub_dict]
            features['pub_score_sum'] = float(sum(hits))
            features['pub_score_max'] = float(max(hits)) if hits else 0.0
            features['pub_hits'] = float(len(hits))
            features['pub_hit_ratio'] = float(len(hits) / len(seq_set)) if seq_set else 0.0

        return features

# ============================================================================
# Ensemble Trainer
# ============================================================================
class EnsembleTrainer:
    """XGBoost + LightGBM ensemble with GPU support."""

    def __init__(self, use_gpu: bool = True, random_state: int = 42):
        self.use_gpu = use_gpu
        self.random_state = random_state
        self.models = {}
        self.weights = {'xgb': 0.5, 'lgb': 0.5}
        self.feature_cols = []

    def select_features_gpu(self, X_df: pd.DataFrame, y: np.ndarray, top_k: int = 500):
        """Select top features using GPU XGBoost."""
        print(f'    Selecting top {top_k} features...', end=' ')
        all_cols = X_df.columns.tolist()
        dtrain = xgb.DMatrix(X_df, label=y)

        params = {
            'tree_method': 'hist',
            'device': 'cuda' if self.use_gpu else 'cpu',
            'max_depth': 4,
            'learning_rate': 0.1,
            'reg_lambda': 1.0,
            'verbosity': 0,
        }

        bst = xgb.train(params, dtrain, num_boost_round=30)
        scores = bst.get_score(importance_type='gain')

        sorted_feats = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        selected = [f[0] for f in sorted_feats[:top_k]]

        # Fill with remaining if needed
        if len(selected) < top_k:
            remaining = [c for c in all_cols if c not in selected]
            selected.extend(remaining[:top_k - len(selected)])

        print(f'Done ({len(selected)} features)')
        return selected

    def train(self, df: pd.DataFrame, ds_id: int) -> Tuple['EnsembleTrainer', List[str], float]:
        """Train ensemble model."""
        y = df['label_positive'].values.astype(np.float32)
        X_df = df.drop(columns=['ID', 'dataset', 'label_positive'], errors='ignore')

        # Feature selection
        self.feature_cols = self.select_features_gpu(X_df, y, top_k=Config.TOP_KMER)
        X = X_df[self.feature_cols].values.astype(np.float32)

        print(f'    Training ensemble on {len(X)} samples, {len(self.feature_cols)} features')

        # XGBoost params
        xgb_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'max_depth': 6,
            'learning_rate': 0.03,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 15,
            'seed': self.random_state,
            'scale_pos_weight': Config.SCALE_POS_WEIGHT.get(ds_id, 1.0),
            'tree_method': 'hist',
            'device': 'cuda' if self.use_gpu else 'cpu',
            'verbosity': 0,
        }

        # LightGBM params
        lgb_params = {
            'objective': 'binary',
            'metric': 'auc',
            'device': 'gpu' if self.use_gpu else 'cpu',
            'max_depth': 6,
            'learning_rate': 0.02,
            'num_leaves': 31,
            'min_child_samples': 20,
            'scale_pos_weight': Config.SCALE_POS_WEIGHT.get(ds_id, 1.0),
            'verbosity': -1,
            'force_col_wise': True,
        }

        # Determine n_splits based on class balance
        pos = int((y == 1).sum())
        neg = int((y == 0).sum())
        min_class = max(2, min(pos, neg))
        n_splits = min(Config.N_SPLITS, min_class)

        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)

        oof_xgb = np.zeros(len(y), dtype=np.float32)
        oof_lgb = np.zeros(len(y), dtype=np.float32)
        cv_xgb, cv_lgb, best_iters = [], [], []

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
            cv_xgb.append(roc_auc_score(y_val, oof_xgb[va_idx]))
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
            cv_lgb.append(roc_auc_score(y_val, oof_lgb[va_idx]))

        mean_xgb = float(np.mean(cv_xgb))
        mean_lgb = float(np.mean(cv_lgb))
        print(f'    CV AUC: XGB={mean_xgb:.4f}, LGB={mean_lgb:.4f}')

        # Learn stacking weights
        meta = LogisticRegression(max_iter=2000)
        meta.fit(np.column_stack([oof_xgb, oof_lgb]), y)
        w = np.clip(meta.coef_[0], 0, None)
        if float(w.sum()) <= 0:
            self.weights = {'xgb': 0.5, 'lgb': 0.5}
        else:
            self.weights = {'xgb': float(w[0] / w.sum()), 'lgb': float(w[1] / w.sum())}

        # Train final models
        rounds = max(int(np.mean(best_iters)) + 50, 100)
        self.models['xgb'] = xgb.train(xgb_params, xgb.DMatrix(X, label=y), num_boost_round=rounds)
        self.models['lgb'] = lgb.train(lgb_params, lgb.Dataset(X, label=y), num_boost_round=800)

        return self, self.feature_cols, mean_xgb

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities."""
        X = X.astype(np.float32)
        p_xgb = self.models['xgb'].predict(xgb.DMatrix(X))
        p_lgb = self.models['lgb'].predict(X)
        return p_xgb * self.weights['xgb'] + p_lgb * self.weights['lgb']

# ============================================================================
# Parallel Processing Helpers
# ============================================================================
def process_file_parallel(row, path: Path, ds_id: int, pub_dict: Dict, extractor: FeatureExtractor):
    """Process a single repertoire file."""
    try:
        df = read_repertoire(path / row['filename'], Config.MAX_SEQUENCES_PER_FILE)
        feats = extractor.extract_all(df, pub_dict, row, ds_id)
        return {
            **feats,
            'ID': row.get('repertoire_id', Path(row['filename']).stem),
            'label_positive': int(row['label_positive']),
            'dataset': path.name,
        }
    except Exception as e:
        return None

def process_test_parallel(tsv: Path, path: Path, ds_id: int, pub_dict: Dict, extractor: FeatureExtractor):
    """Process a single test repertoire file."""
    try:
        df = read_repertoire(tsv, Config.MAX_SEQUENCES_PER_FILE)
        feats = extractor.extract_all(df, pub_dict, None, ds_id)
        return {**feats, 'ID': tsv.stem, 'dataset': path.name}
    except Exception:
        return None

# ============================================================================
# Task B: Sequence Identification
# ============================================================================
def identify_sequences(
    dataset_path: Path,
    pub_dict: Dict,
    top_k: int = 50000
) -> pd.DataFrame:
    """Identify top disease-associated sequences for Task B."""
    meta = pd.read_csv(dataset_path / 'metadata.csv')
    ds_name = dataset_path.name

    # Strategy: Use public clone scores + k-mer importance
    seq_scores = {}

    # Add public clones with their scores
    for seq, info in pub_dict.items():
        seq_scores[seq] = info['score']

    # If not enough sequences, sample from positive repertoires
    if len(seq_scores) < top_k:
        pos_files = meta[meta['label_positive'] == True]['filename'].tolist()
        for f in pos_files[:50]:
            try:
                df = pd.read_csv(dataset_path / f, sep='\t', usecols=['junction_aa', 'v_call', 'j_call'])
                for _, row in df.iterrows():
                    seq = str(row['junction_aa'])
                    if seq and seq not in seq_scores:
                        seq_scores[seq] = 0.0  # Default score
            except Exception:
                continue
            if len(seq_scores) >= top_k * 2:
                break

    # Sort by score and take top_k
    sorted_seqs = sorted(seq_scores.items(), key=lambda x: -x[1])[:top_k]

    # Get V/J calls for top sequences
    v_calls = {}
    j_calls = {}
    for f in meta['filename'].tolist()[:30]:
        try:
            df = pd.read_csv(dataset_path / f, sep='\t', usecols=['junction_aa', 'v_call', 'j_call'])
            for _, row in df.iterrows():
                seq = str(row['junction_aa'])
                if seq in seq_scores:
                    if seq not in v_calls:
                        v_calls[seq] = str(row.get('v_call', '-999.0'))
                        j_calls[seq] = str(row.get('j_call', '-999.0'))
        except Exception:
            continue

    # Build result DataFrame
    results = []
    for i, (seq, score) in enumerate(sorted_seqs[:top_k]):
        results.append({
            'ID': f'{ds_name}_seq_top_{i+1}',
            'dataset': ds_name,
            'label_positive_probability': -999.0,
            'junction_aa': seq,
            'v_call': v_calls.get(seq, '-999.0'),
            'j_call': j_calls.get(seq, '-999.0'),
        })

    # Pad if needed
    while len(results) < top_k:
        results.append({
            'ID': f'{ds_name}_seq_top_{len(results)+1}',
            'dataset': ds_name,
            'label_positive_probability': -999.0,
            'junction_aa': 'CASSXXXXF',  # Placeholder
            'v_call': '-999.0',
            'j_call': '-999.0',
        })

    return pd.DataFrame(results[:top_k])

# ============================================================================
# Main Pipeline
# ============================================================================
def main():
    print('='*70)
    print('AIRR-ML-25 Champion v5 - Enhanced Pipeline')
    print('='*70)

    # Check GPU
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi'], capture_output=True, timeout=5)
        use_gpu = result.returncode == 0
    except Exception:
        use_gpu = False
    print(f'GPU available: {use_gpu}')

    # Create directories
    Config.CHECKPOINT_DIR.mkdir(exist_ok=True)
    Config.SUBMISSION_DIR.mkdir(exist_ok=True)

    # Find datasets
    train_sets = sorted([d.name for d in Config.TRAIN_ROOT.glob('train_dataset_*')])
    test_sets = sorted([d.name for d in Config.TEST_ROOT.glob('test_dataset_*')])

    print(f'Train datasets: {train_sets}')
    print(f'Test datasets: {test_sets}')

    extractor = FeatureExtractor(Config.K_LIST)
    bundles = {}

    # ======== TRAINING ========
    for ds_name in train_sets:
        ds_id = dataset_id_from_name(ds_name)
        ds_path = Config.TRAIN_ROOT / ds_name

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
        print(f'    Found {len(pub_dict)} public clones')

        # Extract features
        print('  Extracting features...')
        meta = pd.read_csv(ds_path / 'metadata.csv')

        results = Parallel(n_jobs=8, backend='loky')(
            delayed(process_file_parallel)(row, ds_path, ds_id, pub_dict, extractor)
            for _, row in tqdm(meta.iterrows(), total=len(meta), leave=False, desc='    ')
        )

        df = pd.DataFrame([r for r in results if r is not None])
        print(f'    Extracted {len(df)} repertoires with {len(df.columns)} raw features')

        # Train model
        trainer = EnsembleTrainer(use_gpu=use_gpu, random_state=Config.RANDOM_STATE)
        trainer, fcols, score = trainer.train(df, ds_id)

        bundles[ds_name] = {
            'trainer': trainer,
            'cols': fcols,
            'pub': pub_dict,
            'score': score
        }
        print(f'  Stored bundle for {ds_name} | CV AUC: {score:.4f}')

        del df, results
        gc.collect()

    # ======== PREDICTION (Task A) ========
    print('\n' + '='*70)
    print('PREDICTING (Task A)')
    print('='*70)

    all_preds = []

    for test_name in test_sets:
        ds_id = dataset_id_from_name(test_name)
        train_key = f'train_dataset_{ds_id}'

        if train_key not in bundles:
            train_key = train_sets[0] if train_sets else None

        if train_key is None:
            raise RuntimeError('No training bundles available')

        bundle = bundles[train_key]
        test_path = Config.TEST_ROOT / test_name

        print(f'  {test_name} -> {train_key}')

        files = sorted(test_path.glob('*.tsv'))

        results = Parallel(n_jobs=8, backend='loky')(
            delayed(process_test_parallel)(f, test_path, ds_id, bundle['pub'], extractor)
            for f in tqdm(files, leave=False, desc='    ')
        )

        test_df = pd.DataFrame([r for r in results if r is not None])

        # Align features
        X = pd.DataFrame(0.0, index=np.arange(len(test_df)), columns=bundle['cols'])
        for c in bundle['cols']:
            if c in test_df.columns:
                X[c] = test_df[c].astype(np.float32)

        # Predict
        probs = bundle['trainer'].predict(X.values)

        sub_part = test_df[['ID', 'dataset']].copy()
        sub_part['label_positive_probability'] = probs.astype(float)
        sub_part['junction_aa'] = '-999.0'
        sub_part['v_call'] = '-999.0'
        sub_part['j_call'] = '-999.0'

        all_preds.append(sub_part)

        del test_df, X, results
        gc.collect()

    # ======== SEQUENCE IDENTIFICATION (Task B) ========
    print('\n' + '='*70)
    print('SEQUENCE IDENTIFICATION (Task B)')
    print('='*70)

    all_seqs = []

    for ds_name in train_sets:
        ds_path = Config.TRAIN_ROOT / ds_name
        bundle = bundles[ds_name]

        print(f'  {ds_name}...')
        seq_df = identify_sequences(ds_path, bundle['pub'], Config.TOP_K_SEQUENCES)
        all_seqs.append(seq_df)
        print(f'    Generated {len(seq_df)} sequences')

    # ======== CREATE SUBMISSION ========
    print('\n' + '='*70)
    print('CREATING SUBMISSION')
    print('='*70)

    # Combine Task A and Task B
    task_a = pd.concat(all_preds, ignore_index=True)
    task_b = pd.concat(all_seqs, ignore_index=True)

    submission = pd.concat([task_a, task_b], ignore_index=True)

    # Ensure correct column order
    submission = submission[['ID', 'dataset', 'label_positive_probability', 'junction_aa', 'v_call', 'j_call']]

    # Save
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = Config.SUBMISSION_DIR / f'v5_submission_{timestamp}.csv'
    submission.to_csv(output_path, index=False)

    print(f'\nSubmission saved to: {output_path}')
    print(f'Total rows: {len(submission)} (expected: 404213)')
    print(f'Task A rows: {len(task_a)}')
    print(f'Task B rows: {len(task_b)}')

    # Validate
    if len(submission) == 404213:
        print('\nValidation PASSED!')
    else:
        print(f'\nValidation FAILED! Expected 404213, got {len(submission)}')

    # Summary
    print('\n' + '='*70)
    print('TRAINING SUMMARY')
    print('='*70)
    for ds_name, bundle in bundles.items():
        print(f"  {ds_name}: CV AUC = {bundle['score']:.4f}")

if __name__ == '__main__':
    main()
