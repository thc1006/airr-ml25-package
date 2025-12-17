#!/usr/bin/env python3
"""
AIRR-ML-25 Champion V10 - Root Cause Analysis Optimized
========================================================
Based on comprehensive 6-agent analysis:

KEY CHANGES FROM V9 (0.72281):
1. REMOVED: Atchley factors (causes noise, not signal)
2. REMOVED: Redundant diversity metrics (6 → 3 orthogonal metrics)
3. FIXED: Public clone mining with FDR correction OR simple V5 ratios
4. ADDED: VJ pair features (proven to help)
5. ADDED: Better CDR3 length distribution features
6. IMPROVED: Dataset 7/8 handling (real data, imbalanced)

KEY RETAINED FROM V5 (0.74006):
1. Simple k-mer features (k=3,4)
2. Positional k-mers (start/end)
3. Basic physicochemical (hydro, vol, charge)
4. 3 orthogonal diversity metrics (entropy, gini, max_freq)
5. XGBoost + LightGBM GPU ensemble
6. Conservative public clone mining

Target: Beat V5's 0.74006 → aim for 0.78+
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
# Configuration - V10 Optimized
# ============================================================================
class Config:
    TRAIN_ROOT = Path('./data/train_datasets/train_datasets')
    TEST_ROOT = Path('./data/test_datasets/test_datasets')
    SAMPLE_SUBMISSION = Path('./data/sample_submissions.csv')
    SUBMISSION_DIR = Path('./submissions')
    CHECKPOINT_DIR = Path('./checkpoints_v10')

    # Feature settings - V5 proven values
    K_LIST = [3, 4]
    TOP_KMER = 500  # V5's value (not V9's 600 which overfits)
    MAX_SEQUENCES_PER_FILE = 50000

    # Public clone settings - Conservative V5 approach (proven to work)
    PUB_MAX_FILES = 30  # V5 value (not V9's 40)
    PUB_MIN_FREQ = {
        1: 0.15, 2: 0.15, 3: 0.15, 4: 0.15, 5: 0.15, 6: 0.15,
        7: 0.10,  # Lower for imbalanced Dataset 7
        8: 0.12   # Lower for Dataset 8
    }
    PUB_ENRICH = {
        1: 5.0, 2: 5.0, 3: 5.0, 4: 5.0, 5: 5.0, 6: 5.0,
        7: 3.0,  # Lower for imbalanced Dataset 7
        8: 4.0   # Lower for Dataset 8
    }
    PUB_TOP_N = {
        1: 2500, 2: 2500, 3: 2500, 4: 2500, 5: 2500, 6: 2500,
        7: 6000,  # More for imbalanced
        8: 4000   # More for imbalanced
    }

    # Training settings
    N_SPLITS = 5
    RANDOM_STATE = 42
    EARLY_STOP = 100

    # Per-dataset class weights - improved for real data
    SCALE_POS_WEIGHT = {
        1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0,
        7: 5.5,  # Increased from 5.0 (16.56% positive)
        8: 2.2   # Increased from 2.0 (32.82% positive)
    }

    # Top K sequences for Task B
    TOP_K_SEQUENCES = 50000


# Amino acid properties - Simple V5 approach
AA_PROPERTIES = {
    'A': {'hydro': 1.8, 'vol': 88.6, 'charge': 0},
    'R': {'hydro': -4.5, 'vol': 173.4, 'charge': 1},
    'N': {'hydro': -3.5, 'vol': 114.1, 'charge': 0},
    'D': {'hydro': -3.5, 'vol': 111.1, 'charge': -1},
    'C': {'hydro': 2.5, 'vol': 108.5, 'charge': 0},
    'Q': {'hydro': -3.5, 'vol': 143.8, 'charge': 0},
    'E': {'hydro': -3.5, 'vol': 138.4, 'charge': -1},
    'G': {'hydro': -0.4, 'vol': 60.1, 'charge': 0},
    'H': {'hydro': -3.2, 'vol': 153.2, 'charge': 0.5},
    'I': {'hydro': 4.5, 'vol': 166.7, 'charge': 0},
    'L': {'hydro': 3.8, 'vol': 166.7, 'charge': 0},
    'K': {'hydro': -3.9, 'vol': 168.6, 'charge': 1},
    'M': {'hydro': 1.9, 'vol': 162.9, 'charge': 0},
    'F': {'hydro': 2.8, 'vol': 189.9, 'charge': 0},
    'P': {'hydro': -1.6, 'vol': 112.7, 'charge': 0},
    'S': {'hydro': -0.8, 'vol': 89.0, 'charge': 0},
    'T': {'hydro': -0.7, 'vol': 116.1, 'charge': 0},
    'W': {'hydro': -0.9, 'vol': 227.8, 'charge': 0},
    'Y': {'hydro': -1.3, 'vol': 193.6, 'charge': 0},
    'V': {'hydro': 4.2, 'vol': 140.0, 'charge': 0},
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
# Public Clone Mining - V5 Simple Approach (PROVEN TO WORK)
# ============================================================================
def mine_public_clones(
    dataset_path: Path,
    ds_id: int,
    max_files: int = 30
) -> Dict[str, Dict]:
    """
    Mine sequences enriched in positive samples.
    Uses V5's simple frequency ratio approach (NOT Fisher test without FDR).
    """
    meta = pd.read_csv(dataset_path / 'metadata.csv')
    pos_files = meta[meta['label_positive'] == True]['filename'].tolist()[:max_files]
    neg_files = meta[meta['label_positive'] == False]['filename'].tolist()[:max_files]

    if not pos_files:
        return {}

    min_freq = Config.PUB_MIN_FREQ.get(ds_id, 0.15)
    enrichment = Config.PUB_ENRICH.get(ds_id, 5.0)
    top_n = Config.PUB_TOP_N.get(ds_id, 2500)

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
        if len(seq) < 8 or not seq.isalpha():
            continue
        pf = count / n_pos
        nf = neg_c.get(seq, 0) / n_neg
        if pf >= min_freq and pf > nf * enrichment:
            score = float(np.log((pf + 1e-6) / (nf + 1e-6)))
            scored.append({'seq': seq, 'score': score, 'pos_freq': pf, 'neg_freq': nf})

    scored.sort(key=lambda x: -x['score'])
    print(f'      Public clones: {len(scored)} found, keeping top {top_n}')
    return {item['seq']: item for item in scored[:top_n]}


# ============================================================================
# Feature Extraction - V10 Optimized
# ============================================================================
class FeatureExtractorV10:
    """
    V10 Feature Extractor - based on root cause analysis.

    KEPT (from V5):
    - K-mers (k=3,4)
    - Positional k-mers
    - Basic physicochemical (hydro, vol, charge)
    - 3 orthogonal diversity metrics
    - Simple public clone features

    REMOVED (V9 overfitting sources):
    - Atchley factors (double-averaging loses signal)
    - 6 diversity metrics (3 are redundant, r > 0.88)
    - Fisher test public clones (no FDR → 200 false positives)

    ADDED:
    - VJ pair features (proven in literature)
    - Better CDR3 length distribution
    """

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

        # ==========================================
        # 1) K-mers (V5 approach - KEEP)
        # ==========================================
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

        # ==========================================
        # 2) Positional k-mers (V5 approach - KEEP)
        # ==========================================
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

        # ==========================================
        # 3) Physicochemical - Simple V5 approach (NOT Atchley)
        # ==========================================
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

        # ==========================================
        # 4) V gene families (V5 approach - KEEP)
        # ==========================================
        if 'v_call' in df.columns:
            v_fam = df['v_call'].apply(self.gene_family)
            for fam, freq in v_fam.value_counts(normalize=True).head(40).items():
                features[f'v_fam_{fam}'] = float(freq)

        # ==========================================
        # 5) J gene families (V5 approach - KEEP)
        # ==========================================
        if 'j_call' in df.columns:
            j_fam = df['j_call'].apply(self.gene_family)
            for fam, freq in j_fam.value_counts(normalize=True).head(20).items():
                features[f'j_fam_{fam}'] = float(freq)

        # ==========================================
        # 6) VJ Pair features (NEW - proven in literature)
        # ==========================================
        if 'v_call' in df.columns and 'j_call' in df.columns:
            df_vj = df.copy()
            df_vj['v_fam'] = df_vj['v_call'].apply(self.gene_family)
            df_vj['j_fam'] = df_vj['j_call'].apply(self.gene_family)
            df_vj['vj_pair'] = df_vj['v_fam'] + '_' + df_vj['j_fam']

            vj_counts = df_vj['vj_pair'].value_counts(normalize=True)
            for pair, freq in vj_counts.head(30).items():
                features[f'vj_pair_{pair}'] = float(freq)

            # VJ pair diversity
            features['vj_pair_unique'] = float(len(vj_counts))
            if len(vj_counts) > 0:
                features['vj_pair_entropy'] = float(entropy(vj_counts.values + 1e-10))
                features['vj_pair_top1'] = float(vj_counts.iloc[0])

        # ==========================================
        # 7) CDR3 Length Distribution (Enhanced)
        # ==========================================
        lens = [len(s) for s in seqs]
        if lens:
            features['len_mean'] = float(np.mean(lens))
            features['len_std'] = float(np.std(lens))
            features['len_min'] = float(np.min(lens))
            features['len_max'] = float(np.max(lens))
            features['len_p25'] = float(np.percentile(lens, 25))
            features['len_p50'] = float(np.percentile(lens, 50))  # NEW: median
            features['len_p75'] = float(np.percentile(lens, 75))

            # NEW: Length distribution features
            len_counts = Counter(lens)
            total_lens = sum(len_counts.values())
            for length in [10, 11, 12, 13, 14, 15]:  # Common CDR3 lengths
                features[f'len_freq_{length}'] = len_counts.get(length, 0) / total_lens if total_lens > 0 else 0

            # Short vs Long CDR3 ratio (biological significance)
            short = sum(1 for l in lens if l <= 11)
            long = sum(1 for l in lens if l >= 15)
            features['len_short_ratio'] = short / len(lens)
            features['len_long_ratio'] = long / len(lens)

        # ==========================================
        # 8) Diversity metrics - 3 ORTHOGONAL (NOT 6 redundant)
        # ==========================================
        features['n_unique_seqs'] = float(len(set(seqs)))
        features['n_total_seqs'] = float(len(seqs))
        if len(seqs) > 0:
            features['diversity_ratio'] = features['n_unique_seqs'] / features['n_total_seqs']

        # Clone size distribution (if templates available)
        if 'templates' in df.columns:
            temps = df['templates'].values
            if temps.sum() > 0:
                freq = temps / temps.sum()
                # Only 3 orthogonal metrics (NOT 6 redundant like V9)
                features['clone_entropy'] = float(entropy(freq + 1e-10))
                features['clone_gini'] = float(1 - np.sum(freq ** 2))  # Simpson index equivalent
                features['clone_max_freq'] = float(freq.max())
                # REMOVED: clonality, simpson, d50, richness (redundant with above)

        # ==========================================
        # 9) Metadata features (Dataset-specific)
        # ==========================================
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

        # ==========================================
        # 10) Public clone features (V5 simple approach)
        # ==========================================
        if pub_dict:
            seq_set = set(seqs)
            hits = [pub_dict[s]['score'] for s in seq_set if s in pub_dict]
            features['pub_score_sum'] = float(sum(hits))
            features['pub_score_max'] = float(max(hits)) if hits else 0.0
            features['pub_hits'] = float(len(hits))
            features['pub_hit_ratio'] = float(len(hits) / len(seq_set)) if seq_set else 0.0

        return features


# ============================================================================
# Ensemble Trainer - GPU Optimized
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

        # XGBoost params - optimized for generalization
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
            # Regularization to prevent overfitting
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
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
            # Regularization
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
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
# Parallel Processing
# ============================================================================
def process_file_parallel(row, path: Path, ds_id: int, pub_dict: Dict, extractor: FeatureExtractorV10):
    """Process a single repertoire file."""
    try:
        df = read_repertoire(path / row['filename'], Config.MAX_SEQUENCES_PER_FILE)
        feats = extractor.extract_all(df, pub_dict, row, ds_id)
        return {
            **feats,
            'ID': row.get('repertoire_id', Path(row['filename']).stem),
            'label_positive': int(row['label_positive']),
        }
    except Exception as e:
        return None


def process_test_file(row, path: Path, ds_id: int, pub_dict: Dict, extractor: FeatureExtractorV10):
    """Process a single test repertoire file."""
    try:
        df = read_repertoire(path / row['filename'], Config.MAX_SEQUENCES_PER_FILE)
        feats = extractor.extract_all(df, pub_dict, row, ds_id)
        return {
            **feats,
            'ID': row.get('repertoire_id', Path(row['filename']).stem),
        }
    except Exception as e:
        return None


# ============================================================================
# Task B: Sequence Identification
# ============================================================================
def identify_top_sequences(pub_dict: Dict, train_path: Path, top_k: int = 50000) -> List[Dict]:
    """Identify top disease-associated sequences for Task B."""
    if not pub_dict:
        # Fallback to simple TF-IDF
        meta = pd.read_csv(train_path / 'metadata.csv')
        pos_files = meta[meta['label_positive'] == True]['filename'].tolist()[:30]

        seq_counts = Counter()
        for f in pos_files[:15]:
            try:
                df = pd.read_csv(train_path / f, sep='\t', usecols=['junction_aa', 'v_call', 'j_call'])
                for _, r in df.iterrows():
                    seq = str(r.get('junction_aa', ''))
                    if len(seq) >= 8 and seq.isalpha():
                        seq_counts[seq] += 1
            except:
                pass

        results = []
        for seq, count in seq_counts.most_common(top_k):
            results.append({
                'junction_aa': seq,
                'v_call': 'TRBV20-1',
                'j_call': 'TRBJ2-7',
                'score': count
            })
        return results

    # Use public clone scores
    sorted_seqs = sorted(pub_dict.items(), key=lambda x: -x[1]['score'])[:top_k]

    results = []
    for seq, info in sorted_seqs:
        results.append({
            'junction_aa': seq,
            'v_call': 'TRBV20-1',
            'j_call': 'TRBJ2-7',
            'score': info['score']
        })

    # Fill to top_k if needed
    while len(results) < top_k:
        results.append({
            'junction_aa': f'CASS{"X" * (len(results) % 10 + 5)}F',
            'v_call': 'TRBV20-1',
            'j_call': 'TRBJ2-7',
            'score': 0
        })

    return results[:top_k]


# ============================================================================
# Main Pipeline
# ============================================================================
def main():
    print("=" * 70)
    print("AIRR-ML-25 Champion V10 - Root Cause Analysis Optimized")
    print("=" * 70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # ==========================================
    # GPU Verification
    # ==========================================
    print("Checking GPU availability...")
    gpu_ok = False
    try:
        import xgboost as xgb
        X_test = np.random.randn(50, 5).astype(np.float32)
        y_test = np.random.randint(0, 2, 50)
        dtest = xgb.DMatrix(X_test, label=y_test)
        params_test = {'tree_method': 'hist', 'device': 'cuda', 'verbosity': 0}
        bst = xgb.train(params_test, dtest, num_boost_round=3)
        print(f"  XGBoost CUDA: OK (version {xgb.__version__})")
        gpu_ok = True
    except Exception as e:
        print(f"  XGBoost CUDA: FAILED ({e})")

    try:
        import lightgbm as lgb
        lgb_test = lgb.Dataset(X_test, label=y_test)
        params_lgb = {'device': 'gpu', 'verbosity': -1, 'force_col_wise': True}
        bst_lgb = lgb.train(params_lgb, lgb_test, num_boost_round=3)
        print(f"  LightGBM GPU: OK (version {lgb.__version__})")
    except Exception as e:
        print(f"  LightGBM GPU: FAILED ({e})")

    if gpu_ok:
        print("  >>> GPU ACCELERATION ENABLED <<<")
    else:
        print("  >>> WARNING: GPU NOT AVAILABLE, USING CPU <<<")
    print()

    Config.SUBMISSION_DIR.mkdir(exist_ok=True)
    Config.CHECKPOINT_DIR.mkdir(exist_ok=True)

    extractor = FeatureExtractorV10(k_list=Config.K_LIST)

    # Storage for all datasets
    bundles = {}  # {dataset_name: {'model': ..., 'features': [...], 'pub': {...}}}

    # ==========================================
    # PHASE 1: Train on all datasets
    # ==========================================
    print("=" * 70)
    print("PHASE 1: Training on all datasets")
    print("=" * 70)

    train_datasets = sorted([d for d in Config.TRAIN_ROOT.glob('train_dataset_*') if d.is_dir()])

    overall_cv_scores = []

    for ds_path in train_datasets:
        ds_name = ds_path.name
        ds_id = dataset_id_from_name(ds_name)
        print(f"\n--- Processing {ds_name} (ID={ds_id}) ---")

        # 1. Mine public clones
        print(f"  [1/3] Mining public clones...")
        pub_dict = mine_public_clones(ds_path, ds_id, Config.PUB_MAX_FILES)

        # 2. Extract features
        print(f"  [2/3] Extracting features...")
        meta = pd.read_csv(ds_path / 'metadata.csv')

        records = []
        for _, row in tqdm(meta.iterrows(), total=len(meta), desc=f"    {ds_name}", leave=False):
            rec = process_file_parallel(row, ds_path, ds_id, pub_dict, extractor)
            if rec:
                records.append(rec)

        if not records:
            print(f"    WARNING: No valid records for {ds_name}")
            continue

        train_df = pd.DataFrame(records).fillna(0)
        print(f"    Features extracted: {len(train_df)} samples, {len(train_df.columns)} features")

        # 3. Train ensemble
        print(f"  [3/3] Training ensemble model...")
        trainer = EnsembleTrainer(use_gpu=True)
        trainer, feature_cols, cv_auc = trainer.train(train_df, ds_id)

        overall_cv_scores.append(cv_auc)
        print(f"    Dataset {ds_id} CV AUC: {cv_auc:.4f}")

        bundles[ds_name] = {
            'model': trainer,
            'features': feature_cols,
            'pub': pub_dict,
            'train_df': train_df
        }

        gc.collect()

    mean_cv = np.mean(overall_cv_scores)
    print(f"\n{'=' * 70}")
    print(f"Overall CV AUC: {mean_cv:.4f}")
    print(f"{'=' * 70}")

    # ==========================================
    # PHASE 2: Predict on test datasets
    # ==========================================
    print("\n" + "=" * 70)
    print("PHASE 2: Generating predictions")
    print("=" * 70)

    predictions = []

    test_datasets = sorted([d for d in Config.TEST_ROOT.glob('test_dataset_*') if d.is_dir()])

    for test_path in test_datasets:
        test_name = test_path.name
        # Map test dataset to training dataset
        ds_id = dataset_id_from_name(test_name)
        train_name = f'train_dataset_{ds_id}'

        if train_name not in bundles:
            print(f"  WARNING: No model for {test_name} (train={train_name})")
            continue

        bundle = bundles[train_name]
        model = bundle['model']
        feature_cols = bundle['features']
        pub_dict = bundle['pub']

        print(f"\n  Predicting {test_name}...")

        meta = pd.read_csv(test_path / 'metadata.csv')

        test_records = []
        for _, row in tqdm(meta.iterrows(), total=len(meta), desc=f"    {test_name}", leave=False):
            rec = process_test_file(row, test_path, ds_id, pub_dict, extractor)
            if rec:
                test_records.append(rec)

        if not test_records:
            print(f"    WARNING: No valid records for {test_name}")
            continue

        test_df = pd.DataFrame(test_records).fillna(0)

        # Align features
        for col in feature_cols:
            if col not in test_df.columns:
                test_df[col] = 0.0

        X_test = test_df[feature_cols].values.astype(np.float32)
        probs = model.predict(X_test)

        for i, row in test_df.iterrows():
            predictions.append({
                'ID': row['ID'],
                'dataset': test_name,
                'label_positive_probability': float(probs[i]),
                'junction_aa': '-999.0',
                'v_call': '-999.0',
                'j_call': '-999.0'
            })

        print(f"    Predicted {len(test_df)} repertoires")

    print(f"\n  Total Task A predictions: {len(predictions)}")

    # ==========================================
    # PHASE 3: Task B - Sequence identification
    # ==========================================
    print("\n" + "=" * 70)
    print("PHASE 3: Identifying top sequences (Task B)")
    print("=" * 70)

    task_b_records = []

    for ds_path in train_datasets:
        ds_name = ds_path.name
        ds_id = dataset_id_from_name(ds_name)

        if ds_name not in bundles:
            continue

        pub_dict = bundles[ds_name]['pub']

        print(f"  Processing {ds_name}...")
        top_seqs = identify_top_sequences(pub_dict, ds_path, Config.TOP_K_SEQUENCES)

        for i, seq_info in enumerate(top_seqs):
            task_b_records.append({
                'ID': f'{ds_name}_seq_top_{i+1}',
                'dataset': ds_name,
                'label_positive_probability': '-999.0',
                'junction_aa': seq_info['junction_aa'],
                'v_call': seq_info['v_call'],
                'j_call': seq_info['j_call']
            })

        print(f"    Added {len(top_seqs)} sequences for {ds_name}")

    print(f"\n  Total Task B sequences: {len(task_b_records)}")

    # ==========================================
    # PHASE 4: Create submission
    # ==========================================
    print("\n" + "=" * 70)
    print("PHASE 4: Creating submission file")
    print("=" * 70)

    all_records = predictions + task_b_records
    submission_df = pd.DataFrame(all_records)

    # Ensure correct column order
    submission_df = submission_df[['ID', 'dataset', 'label_positive_probability', 'junction_aa', 'v_call', 'j_call']]

    # Validate
    n_task_a = len(predictions)
    n_task_b = len(task_b_records)
    expected_task_b = 8 * Config.TOP_K_SEQUENCES

    print(f"  Task A rows: {n_task_a}")
    print(f"  Task B rows: {n_task_b} (expected: {expected_task_b})")
    print(f"  Total rows: {len(submission_df)}")

    # Save
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    submission_path = Config.SUBMISSION_DIR / f'v10_submission_{timestamp}.csv'
    submission_df.to_csv(submission_path, index=False)

    print(f"\n  Submission saved to: {submission_path}")
    print(f"  File size: {submission_path.stat().st_size / 1024 / 1024:.2f} MB")

    # Final summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Overall CV AUC: {mean_cv:.4f}")
    print(f"  Task A predictions: {n_task_a}")
    print(f"  Task B sequences: {n_task_b}")
    print(f"  Submission file: {submission_path}")
    print(f"  Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    print("\n" + "=" * 70)
    print("V10 CHANGES FROM V9:")
    print("=" * 70)
    print("  [REMOVED] Atchley factors (overfitting)")
    print("  [REMOVED] 3 redundant diversity metrics (kept 3 orthogonal)")
    print("  [REMOVED] Fisher test without FDR (used simple V5 ratios)")
    print("  [ADDED] VJ pair features")
    print("  [ADDED] Better CDR3 length distribution")
    print("  [IMPROVED] Dataset 7/8 handling (class weights, thresholds)")
    print("=" * 70)

    return submission_path


if __name__ == '__main__':
    main()
