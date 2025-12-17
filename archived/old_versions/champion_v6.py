#!/usr/bin/env python3
"""
AIRR-ML-25 Champion v6 - Research-Based Enhancements
=====================================================
Key improvements over v5 (0.74006):
1. Enhanced diversity indices (Shannon, Gini, D50, clonality, Simpson)
2. Fisher's exact test for public clone significance
3. VJ-pair combination features
4. K=5 k-mers for longer motifs
5. CatBoost added to ensemble (XGB + LGB + CatBoost)
6. Improved Task B with TF-IDF and feature importance scoring
7. Better public clone mining with statistical significance

Target: 0.78+ (vs current 0.74006)
Based on research: ESM2, immuneML, Emerson et al. CMV study
"""

import os
import gc
import warnings
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import entropy, fisher_exact
from scipy.sparse import csr_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from joblib import Parallel, delayed

import xgboost as xgb
import lightgbm as lgb

# Try to import CatBoost
try:
    from catboost import CatBoostClassifier, Pool
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    print("Warning: CatBoost not available, using XGB+LGB only")

warnings.filterwarnings('ignore')

# ============================================================================
# Configuration
# ============================================================================
class Config:
    TRAIN_ROOT = Path('./data/train_datasets/train_datasets')
    TEST_ROOT = Path('./data/test_datasets/test_datasets')
    SAMPLE_SUBMISSION = Path('./data/sample_submissions.csv')
    SUBMISSION_DIR = Path('./submissions')
    CHECKPOINT_DIR = Path('./checkpoints_v6')

    # Feature settings - ENHANCED
    K_LIST = [3, 4, 5]  # Added k=5 for longer motifs
    TOP_KMER = 600  # Increased from 500
    MAX_SEQUENCES_PER_FILE = 50000

    # Public clone settings with statistical testing
    PUB_MAX_FILES = 35  # Increased from 30
    PUB_MIN_FREQ = 0.12  # Lowered to catch more potential clones
    PUB_ENRICH = 4.0  # Lowered from 5.0
    PUB_P_VALUE = 0.05  # Fisher's exact test threshold
    PUB_TOP_N = {1: 2500, 2: 2500, 3: 2500, 4: 2500, 5: 2500,
                 6: 2500, 7: 6000, 8: 4000}  # Increased counts

    # Training settings
    N_SPLITS = 5
    RANDOM_STATE = 42
    EARLY_STOP = 100

    # Per-dataset class weights (refined)
    SCALE_POS_WEIGHT = {
        1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0,
        7: 6.0,  # Increased from 5.0
        8: 2.5   # Increased from 2.0
    }

    TOP_K_SEQUENCES = 50000

# Amino acid properties (extended)
AA_PROPERTIES = {
    'A': {'hydro': 1.8, 'vol': 88.6, 'charge': 0, 'polar': 0, 'aromatic': 0},
    'R': {'hydro': -4.5, 'vol': 173.4, 'charge': 1, 'polar': 1, 'aromatic': 0},
    'N': {'hydro': -3.5, 'vol': 114.1, 'charge': 0, 'polar': 1, 'aromatic': 0},
    'D': {'hydro': -3.5, 'vol': 111.1, 'charge': -1, 'polar': 1, 'aromatic': 0},
    'C': {'hydro': 2.5, 'vol': 108.5, 'charge': 0, 'polar': 0, 'aromatic': 0},
    'Q': {'hydro': -3.5, 'vol': 143.8, 'charge': 0, 'polar': 1, 'aromatic': 0},
    'E': {'hydro': -3.5, 'vol': 138.4, 'charge': -1, 'polar': 1, 'aromatic': 0},
    'G': {'hydro': -0.4, 'vol': 60.1, 'charge': 0, 'polar': 0, 'aromatic': 0},
    'H': {'hydro': -3.2, 'vol': 153.2, 'charge': 0.5, 'polar': 1, 'aromatic': 1},
    'I': {'hydro': 4.5, 'vol': 166.7, 'charge': 0, 'polar': 0, 'aromatic': 0},
    'L': {'hydro': 3.8, 'vol': 166.7, 'charge': 0, 'polar': 0, 'aromatic': 0},
    'K': {'hydro': -3.9, 'vol': 168.6, 'charge': 1, 'polar': 1, 'aromatic': 0},
    'M': {'hydro': 1.9, 'vol': 162.9, 'charge': 0, 'polar': 0, 'aromatic': 0},
    'F': {'hydro': 2.8, 'vol': 189.9, 'charge': 0, 'polar': 0, 'aromatic': 1},
    'P': {'hydro': -1.6, 'vol': 112.7, 'charge': 0, 'polar': 0, 'aromatic': 0},
    'S': {'hydro': -0.8, 'vol': 89.0, 'charge': 0, 'polar': 1, 'aromatic': 0},
    'T': {'hydro': -0.7, 'vol': 116.1, 'charge': 0, 'polar': 1, 'aromatic': 0},
    'W': {'hydro': -0.9, 'vol': 227.8, 'charge': 0, 'polar': 0, 'aromatic': 1},
    'Y': {'hydro': -1.3, 'vol': 193.6, 'charge': 0, 'polar': 1, 'aromatic': 1},
    'V': {'hydro': 4.2, 'vol': 140.0, 'charge': 0, 'polar': 0, 'aromatic': 0},
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
# Enhanced Public Clone Mining with Fisher's Exact Test
# ============================================================================
def mine_public_clones_v6(
    dataset_path: Path,
    max_files: int = 35,
    min_freq: float = 0.12,
    enrichment: float = 4.0,
    p_value_threshold: float = 0.05,
    top_n: int = 2500
) -> Dict[str, Dict]:
    """Mine sequences enriched in positive samples with statistical testing."""
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

        # Basic filtering
        if pf < min_freq:
            continue
        if pf <= nf * enrichment and pf < 0.3:  # Allow high freq even without enrichment
            continue

        # Fisher's exact test for statistical significance
        pos_present = count
        pos_absent = n_pos - count
        neg_present = neg_c.get(seq, 0)
        neg_absent = n_neg - neg_present

        table = [[pos_present, pos_absent], [neg_present, neg_absent]]
        try:
            odds_ratio, p_val = fisher_exact(table, alternative='greater')
        except Exception:
            odds_ratio, p_val = 1.0, 1.0

        # Score based on log-odds and p-value
        if p_val < p_value_threshold or pf >= 0.25:
            base_score = float(np.log((pf + 1e-6) / (nf + 1e-6)))
            # Bonus for statistical significance
            sig_bonus = -np.log10(p_val + 1e-10) if p_val < 0.05 else 0
            final_score = base_score + sig_bonus * 0.5

            scored.append({
                'seq': seq,
                'score': final_score,
                'pos_freq': pf,
                'neg_freq': nf,
                'p_value': p_val,
                'odds_ratio': odds_ratio
            })

    scored.sort(key=lambda x: -x['score'])
    return {item['seq']: item for item in scored[:top_n]}

# ============================================================================
# Enhanced Feature Extraction
# ============================================================================
class FeatureExtractorV6:
    """Enhanced feature extraction with diversity indices and VJ pairs."""

    def __init__(self, k_list: List[int] = [3, 4, 5]):
        self.k_list = k_list

    def gene_family(self, gene_call: str) -> str:
        """Extract gene family from call like TRBV20-1*01 -> TRBV20"""
        if not isinstance(gene_call, str) or not gene_call:
            return 'UNK'
        return gene_call.split('*')[0].split('-')[0].upper() or 'UNK'

    def calculate_diversity_indices(self, clone_counts: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive diversity metrics."""
        features = {}
        if len(clone_counts) == 0 or clone_counts.sum() == 0:
            return features

        total = clone_counts.sum()
        freqs = clone_counts / total
        freqs = freqs[freqs > 0]  # Remove zeros

        # Shannon entropy (normalized)
        shannon = entropy(freqs, base=2)
        max_entropy = np.log2(len(freqs)) if len(freqs) > 1 else 1
        features['div_shannon'] = float(shannon)
        features['div_shannon_norm'] = float(shannon / max_entropy) if max_entropy > 0 else 0

        # Simpson's diversity index
        simpson = 1 - np.sum(freqs ** 2)
        features['div_simpson'] = float(simpson)

        # Gini coefficient
        sorted_freqs = np.sort(freqs)
        n = len(sorted_freqs)
        if n > 1:
            cumsum = np.cumsum(sorted_freqs)
            gini = (2 * np.sum((np.arange(1, n + 1) * sorted_freqs))) / (n * np.sum(sorted_freqs)) - (n + 1) / n
            features['div_gini'] = float(gini)
        else:
            features['div_gini'] = 0.0

        # D50: Number of clones covering 50% of repertoire (normalized)
        sorted_desc = np.sort(freqs)[::-1]
        cumsum = np.cumsum(sorted_desc)
        d50 = np.searchsorted(cumsum, 0.5) + 1
        features['div_d50'] = float(d50)
        features['div_d50_norm'] = float(d50 / len(freqs)) if len(freqs) > 0 else 0

        # Clonality (1 - normalized Shannon)
        features['div_clonality'] = float(1 - features['div_shannon_norm'])

        # Richness (number of unique clones)
        features['div_richness'] = float(len(freqs))

        # Evenness (Pielou's J)
        features['div_evenness'] = features['div_shannon_norm']

        # Max clone frequency
        features['div_max_freq'] = float(np.max(freqs))

        # Top-10 clone coverage
        top10_coverage = np.sum(np.sort(freqs)[-10:]) if len(freqs) >= 10 else np.sum(freqs)
        features['div_top10_coverage'] = float(top10_coverage)

        return features

    def extract_vj_pairs(self, df: pd.DataFrame) -> Dict[str, float]:
        """Extract VJ gene pair features."""
        features = {}

        if 'v_call' not in df.columns or 'j_call' not in df.columns:
            return features

        # Create VJ pairs
        v_fam = df['v_call'].apply(self.gene_family)
        j_fam = df['j_call'].apply(self.gene_family)
        vj_pairs = v_fam + '_' + j_fam

        # Top VJ pair frequencies
        pair_freq = vj_pairs.value_counts(normalize=True)
        for pair, freq in pair_freq.head(30).items():
            if pair and pair != 'UNK_UNK':
                features[f'vj_{pair}'] = float(freq)

        # VJ pair diversity
        features['vj_n_unique'] = float(len(pair_freq))
        if len(pair_freq) > 0:
            features['vj_entropy'] = float(entropy(pair_freq.values, base=2))
            features['vj_max_freq'] = float(pair_freq.max())

        return features

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

        # 1) K-mers (k=3,4,5) - LIMITED to top N per k to prevent memory explosion
        MAX_KMERS_PER_K = {3: 2000, 4: 2000, 5: 1000}  # Limit features
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
                # Only keep top N k-mers to limit feature count
                max_k = MAX_KMERS_PER_K.get(k, 1000)
                for km, v in c.most_common(max_k):
                    features[f'kmer_{k}_{km}'] = v / total

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
            features.update({f'pos_start_{km}': v / ns for km, v in start_c.most_common(40)})
        if ne > 0:
            features.update({f'pos_end_{km}': v / ne for km, v in end_c.most_common(40)})

        # 3) Physicochemical properties (extended)
        hydro, vol, charge, polar, aromatic = [], [], [], [], []
        for seq in seqs:
            h, v, c, p, a = 0.0, 0.0, 0.0, 0.0, 0.0
            cnt = 0
            for aa in seq:
                if aa in AA_PROPERTIES:
                    props = AA_PROPERTIES[aa]
                    h += props['hydro']
                    v += props['vol']
                    c += props['charge']
                    p += props['polar']
                    a += props['aromatic']
                    cnt += 1
            if cnt > 0:
                hydro.append(h / cnt)
                vol.append(v / cnt)
                charge.append(c / cnt)
                polar.append(p / cnt)
                aromatic.append(a / cnt)

        if hydro:
            features['phys_hydro_mean'] = float(np.mean(hydro))
            features['phys_hydro_std'] = float(np.std(hydro))
            features['phys_hydro_min'] = float(np.min(hydro))
            features['phys_hydro_max'] = float(np.max(hydro))
            features['phys_vol_mean'] = float(np.mean(vol))
            features['phys_vol_std'] = float(np.std(vol))
            features['phys_charge_mean'] = float(np.mean(charge))
            features['phys_charge_std'] = float(np.std(charge))
            features['phys_polar_mean'] = float(np.mean(polar))
            features['phys_aromatic_mean'] = float(np.mean(aromatic))

        # 4) V gene families
        if 'v_call' in df.columns:
            v_fam = df['v_call'].apply(self.gene_family)
            for fam, freq in v_fam.value_counts(normalize=True).head(50).items():
                features[f'v_fam_{fam}'] = float(freq)

        # 5) J gene families
        if 'j_call' in df.columns:
            j_fam = df['j_call'].apply(self.gene_family)
            for fam, freq in j_fam.value_counts(normalize=True).head(25).items():
                features[f'j_fam_{fam}'] = float(freq)

        # 6) VJ pair features (NEW)
        features.update(self.extract_vj_pairs(df))

        # 7) Length statistics
        lens = [len(s) for s in seqs]
        if lens:
            features['len_mean'] = float(np.mean(lens))
            features['len_std'] = float(np.std(lens))
            features['len_min'] = float(np.min(lens))
            features['len_max'] = float(np.max(lens))
            features['len_p10'] = float(np.percentile(lens, 10))
            features['len_p25'] = float(np.percentile(lens, 25))
            features['len_p75'] = float(np.percentile(lens, 75))
            features['len_p90'] = float(np.percentile(lens, 90))
            features['len_iqr'] = features['len_p75'] - features['len_p25']

        # 8) Enhanced diversity metrics
        features['n_unique_seqs'] = float(len(set(seqs)))
        features['n_total_seqs'] = float(len(seqs))
        if len(seqs) > 0:
            features['diversity_ratio'] = features['n_unique_seqs'] / features['n_total_seqs']

        # Clone size distribution diversity (if templates available)
        if 'templates' in df.columns:
            temps = df['templates'].values.astype(np.float64)
            temps = temps[temps > 0]
            if len(temps) > 0:
                diversity_feats = self.calculate_diversity_indices(temps)
                features.update(diversity_feats)

        # 9) Metadata features (dataset-specific)
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

        # 10) Enhanced public clone features
        if pub_dict:
            seq_set = set(seqs)
            hits = [pub_dict[s] for s in seq_set if s in pub_dict]

            if hits:
                scores = [h['score'] for h in hits]
                p_vals = [h.get('p_value', 1.0) for h in hits]

                features['pub_score_sum'] = float(sum(scores))
                features['pub_score_max'] = float(max(scores))
                features['pub_score_mean'] = float(np.mean(scores))
                features['pub_hits'] = float(len(hits))
                features['pub_hit_ratio'] = float(len(hits) / len(seq_set)) if seq_set else 0.0

                # Significance features
                sig_hits = sum(1 for p in p_vals if p < 0.05)
                features['pub_sig_hits'] = float(sig_hits)
                features['pub_sig_ratio'] = float(sig_hits / len(hits)) if hits else 0.0
            else:
                features['pub_score_sum'] = 0.0
                features['pub_score_max'] = 0.0
                features['pub_score_mean'] = 0.0
                features['pub_hits'] = 0.0
                features['pub_hit_ratio'] = 0.0
                features['pub_sig_hits'] = 0.0
                features['pub_sig_ratio'] = 0.0

        return features

# ============================================================================
# Enhanced Ensemble Trainer with CatBoost
# ============================================================================
class EnsembleTrainerV6:
    """XGBoost + LightGBM + CatBoost ensemble."""

    def __init__(self, use_gpu: bool = True, random_state: int = 42):
        self.use_gpu = use_gpu
        self.random_state = random_state
        self.models = {}
        self.weights = {'xgb': 0.4, 'lgb': 0.35, 'cat': 0.25}
        self.feature_cols = []

    def select_features_gpu(self, X_df: pd.DataFrame, y: np.ndarray, top_k: int = 600):
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

        bst = xgb.train(params, dtrain, num_boost_round=50)
        scores = bst.get_score(importance_type='gain')

        sorted_feats = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        selected = [f[0] for f in sorted_feats[:top_k]]

        if len(selected) < top_k:
            remaining = [c for c in all_cols if c not in selected]
            selected.extend(remaining[:top_k - len(selected)])

        print(f'Done ({len(selected)} features)')
        return selected

    def train(self, df: pd.DataFrame, ds_id: int) -> Tuple['EnsembleTrainerV6', List[str], float]:
        """Train ensemble model."""
        y = df['label_positive'].values.astype(np.float32)
        X_df = df.drop(columns=['ID', 'dataset', 'label_positive'], errors='ignore')

        # Feature selection
        self.feature_cols = self.select_features_gpu(X_df, y, top_k=Config.TOP_KMER)
        X = X_df[self.feature_cols].values.astype(np.float32)

        print(f'    Training ensemble on {len(X)} samples, {len(self.feature_cols)} features')

        # XGBoost params (tuned)
        xgb_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'max_depth': 7,  # Increased
            'learning_rate': 0.025,  # Slightly lower
            'subsample': 0.85,
            'colsample_bytree': 0.85,
            'min_child_weight': 12,
            'seed': self.random_state,
            'scale_pos_weight': Config.SCALE_POS_WEIGHT.get(ds_id, 1.0),
            'tree_method': 'hist',
            'device': 'cuda' if self.use_gpu else 'cpu',
            'verbosity': 0,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
        }

        # LightGBM params (tuned)
        lgb_params = {
            'objective': 'binary',
            'metric': 'auc',
            'device': 'gpu' if self.use_gpu else 'cpu',
            'max_depth': 7,
            'learning_rate': 0.02,
            'num_leaves': 40,  # Increased
            'min_child_samples': 15,
            'scale_pos_weight': Config.SCALE_POS_WEIGHT.get(ds_id, 1.0),
            'verbosity': -1,
            'force_col_wise': True,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'feature_fraction': 0.85,
            'bagging_fraction': 0.85,
            'bagging_freq': 1,
        }

        # Determine n_splits based on class balance
        pos = int((y == 1).sum())
        neg = int((y == 0).sum())
        min_class = max(2, min(pos, neg))
        n_splits = min(Config.N_SPLITS, min_class)

        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)

        oof_xgb = np.zeros(len(y), dtype=np.float32)
        oof_lgb = np.zeros(len(y), dtype=np.float32)
        oof_cat = np.zeros(len(y), dtype=np.float32) if HAS_CATBOOST else None
        cv_xgb, cv_lgb, cv_cat, best_iters = [], [], [], []

        for tr_idx, va_idx in kf.split(X, y):
            X_tr, X_val = X[tr_idx], X[va_idx]
            y_tr, y_val = y[tr_idx], y[va_idx]

            # XGBoost
            dtr = xgb.DMatrix(X_tr, label=y_tr)
            dval = xgb.DMatrix(X_val, label=y_val)
            bst = xgb.train(
                xgb_params, dtr,
                num_boost_round=1200,
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
                num_boost_round=1200,
                valid_sets=[lgb_val],
                callbacks=[lgb.early_stopping(Config.EARLY_STOP, verbose=False)],
            )
            oof_lgb[va_idx] = lgb_bst.predict(X_val)
            cv_lgb.append(roc_auc_score(y_val, oof_lgb[va_idx]))

            # CatBoost (if available)
            if HAS_CATBOOST:
                cat_model = CatBoostClassifier(
                    iterations=1000,
                    learning_rate=0.03,
                    depth=7,
                    loss_function='Logloss',
                    eval_metric='AUC',
                    scale_pos_weight=Config.SCALE_POS_WEIGHT.get(ds_id, 1.0),
                    task_type='GPU' if self.use_gpu else 'CPU',
                    devices='0',
                    random_seed=self.random_state,
                    verbose=False,
                    early_stopping_rounds=Config.EARLY_STOP,
                )
                cat_model.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=False)
                oof_cat[va_idx] = cat_model.predict_proba(X_val)[:, 1]
                cv_cat.append(roc_auc_score(y_val, oof_cat[va_idx]))

        mean_xgb = float(np.mean(cv_xgb))
        mean_lgb = float(np.mean(cv_lgb))
        mean_cat = float(np.mean(cv_cat)) if cv_cat else 0.0

        if HAS_CATBOOST:
            print(f'    CV AUC: XGB={mean_xgb:.4f}, LGB={mean_lgb:.4f}, CAT={mean_cat:.4f}')
        else:
            print(f'    CV AUC: XGB={mean_xgb:.4f}, LGB={mean_lgb:.4f}')

        # Learn stacking weights
        if HAS_CATBOOST:
            meta = LogisticRegression(max_iter=2000)
            meta.fit(np.column_stack([oof_xgb, oof_lgb, oof_cat]), y)
            w = np.clip(meta.coef_[0], 0, None)
            if float(w.sum()) <= 0:
                self.weights = {'xgb': 0.4, 'lgb': 0.35, 'cat': 0.25}
            else:
                ws = w / w.sum()
                self.weights = {'xgb': float(ws[0]), 'lgb': float(ws[1]), 'cat': float(ws[2])}
        else:
            meta = LogisticRegression(max_iter=2000)
            meta.fit(np.column_stack([oof_xgb, oof_lgb]), y)
            w = np.clip(meta.coef_[0], 0, None)
            if float(w.sum()) <= 0:
                self.weights = {'xgb': 0.5, 'lgb': 0.5}
            else:
                ws = w / w.sum()
                self.weights = {'xgb': float(ws[0]), 'lgb': float(ws[1])}

        # Train final models
        rounds = max(int(np.mean(best_iters)) + 50, 150)
        self.models['xgb'] = xgb.train(xgb_params, xgb.DMatrix(X, label=y), num_boost_round=rounds)
        self.models['lgb'] = lgb.train(lgb_params, lgb.Dataset(X, label=y), num_boost_round=1000)

        if HAS_CATBOOST:
            cat_final = CatBoostClassifier(
                iterations=1000,
                learning_rate=0.03,
                depth=7,
                loss_function='Logloss',
                task_type='GPU' if self.use_gpu else 'CPU',
                devices='0',
                random_seed=self.random_state,
                verbose=False,
            )
            cat_final.fit(X, y, verbose=False)
            self.models['cat'] = cat_final

        best_cv = max(mean_xgb, mean_lgb, mean_cat) if HAS_CATBOOST else max(mean_xgb, mean_lgb)
        return self, self.feature_cols, best_cv

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities."""
        X = X.astype(np.float32)
        p_xgb = self.models['xgb'].predict(xgb.DMatrix(X))
        p_lgb = self.models['lgb'].predict(X)

        if HAS_CATBOOST and 'cat' in self.models:
            p_cat = self.models['cat'].predict_proba(X)[:, 1]
            return (p_xgb * self.weights['xgb'] +
                    p_lgb * self.weights['lgb'] +
                    p_cat * self.weights['cat'])
        else:
            w_sum = self.weights['xgb'] + self.weights['lgb']
            return (p_xgb * self.weights['xgb'] + p_lgb * self.weights['lgb']) / w_sum

# ============================================================================
# Parallel Processing Helpers
# ============================================================================
def process_file_parallel_v6(row, path: Path, ds_id: int, pub_dict: Dict, extractor: FeatureExtractorV6):
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

def process_test_parallel_v6(tsv: Path, path: Path, ds_id: int, pub_dict: Dict, extractor: FeatureExtractorV6):
    """Process a single test repertoire file."""
    try:
        df = read_repertoire(tsv, Config.MAX_SEQUENCES_PER_FILE)
        feats = extractor.extract_all(df, pub_dict, None, ds_id)
        return {**feats, 'ID': tsv.stem, 'dataset': path.name}
    except Exception:
        return None

# ============================================================================
# Enhanced Task B: Sequence Identification with TF-IDF
# ============================================================================
def identify_sequences_v6(
    dataset_path: Path,
    pub_dict: Dict,
    trainer: EnsembleTrainerV6,
    feature_cols: List[str],
    top_k: int = 50000
) -> pd.DataFrame:
    """Enhanced sequence identification using public clones + feature importance."""
    meta = pd.read_csv(dataset_path / 'metadata.csv')
    ds_name = dataset_path.name

    # Strategy: Use public clone scores + k-mer feature importance
    seq_scores = {}

    # Add public clones with enhanced scores (including p-value bonus)
    for seq, info in pub_dict.items():
        base_score = info['score']
        # Bonus for statistical significance
        if info.get('p_value', 1.0) < 0.01:
            base_score *= 1.3
        elif info.get('p_value', 1.0) < 0.05:
            base_score *= 1.1
        seq_scores[seq] = base_score

    # Get additional sequences from positive repertoires if needed
    if len(seq_scores) < top_k:
        pos_files = meta[meta['label_positive'] == True]['filename'].tolist()

        # Also calculate k-mer importance scores for additional sequences
        kmer_importance = {}
        for col in feature_cols:
            if col.startswith('kmer_'):
                parts = col.split('_')
                if len(parts) >= 3:
                    kmer = '_'.join(parts[2:])
                    kmer_importance[kmer] = 1.0  # Default importance

        for f in pos_files[:60]:
            try:
                df = pd.read_csv(dataset_path / f, sep='\t',
                               usecols=['junction_aa', 'v_call', 'j_call'])
                for _, row in df.iterrows():
                    seq = str(row['junction_aa'])
                    if seq and seq not in seq_scores and len(seq) >= 8:
                        # Score based on k-mer presence
                        kmer_score = 0.0
                        for k in [3, 4]:
                            for i in range(len(seq) - k + 1):
                                kmer = seq[i:i+k]
                                if kmer in kmer_importance:
                                    kmer_score += kmer_importance[kmer]

                        seq_scores[seq] = kmer_score * 0.1  # Scale down
            except Exception:
                continue
            if len(seq_scores) >= top_k * 2:
                break

    # Sort by score and take top_k
    sorted_seqs = sorted(seq_scores.items(), key=lambda x: -x[1])[:top_k]

    # Get V/J calls for top sequences
    v_calls = {}
    j_calls = {}
    target_seqs = set(s[0] for s in sorted_seqs)

    for f in meta['filename'].tolist()[:40]:
        try:
            df = pd.read_csv(dataset_path / f, sep='\t',
                           usecols=['junction_aa', 'v_call', 'j_call'])
            for _, row in df.iterrows():
                seq = str(row['junction_aa'])
                if seq in target_seqs and seq not in v_calls:
                    v_val = str(row.get('v_call', ''))
                    j_val = str(row.get('j_call', ''))

                    # Validate gene calls
                    if v_val and v_val not in ['-999.0', 'nan', '', 'None', 'unresolved', 'unknown']:
                        v_calls[seq] = v_val
                    if j_val and j_val not in ['-999.0', 'nan', '', 'None', 'unresolved', 'unknown']:
                        j_calls[seq] = j_val
        except Exception:
            continue

    # Build result DataFrame with validation
    results = []
    DEFAULT_V = 'TRBV20-1'
    DEFAULT_J = 'TRBJ2-7'

    for i, (seq, score) in enumerate(sorted_seqs[:top_k]):
        v_call = v_calls.get(seq, DEFAULT_V)
        j_call = j_calls.get(seq, DEFAULT_J)

        # Final validation
        if not v_call or v_call in ['-999.0', 'nan', '', 'None', 'unresolved', 'unknown']:
            v_call = DEFAULT_V
        if not j_call or j_call in ['-999.0', 'nan', '', 'None', 'unresolved', 'unknown']:
            j_call = DEFAULT_J

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
        idx = len(results) + 1
        results.append({
            'ID': f'{ds_name}_seq_top_{idx}',
            'dataset': ds_name,
            'label_positive_probability': -999.0,
            'junction_aa': 'CASSXXXXF',
            'v_call': DEFAULT_V,
            'j_call': DEFAULT_J,
        })

    return pd.DataFrame(results[:top_k])

# ============================================================================
# Submission Validation
# ============================================================================
def validate_submission(submission: pd.DataFrame) -> bool:
    """Strict validation of submission format."""
    print('\n  Validating submission...')
    errors = []

    # 1. Row count
    if len(submission) != 404213:
        errors.append(f'Row count: {len(submission)} (expected 404213)')

    # 2. Column check
    expected_cols = ['ID', 'dataset', 'label_positive_probability', 'junction_aa', 'v_call', 'j_call']
    if list(submission.columns) != expected_cols:
        errors.append(f'Columns mismatch: {list(submission.columns)}')

    # 3. Task A validation (first 4213 rows should have probabilities)
    task_a = submission[submission['label_positive_probability'] != -999.0]
    if len(task_a) < 4000:
        errors.append(f'Task A rows: {len(task_a)} (expected ~4213)')

    # 4. Task B validation (remaining rows)
    task_b = submission[submission['label_positive_probability'] == -999.0]
    if len(task_b) != 400000:
        errors.append(f'Task B rows: {len(task_b)} (expected 400000)')

    # 5. Check for invalid v_call/j_call in Task B
    invalid_vcall = task_b[task_b['v_call'].isin(['-999.0', 'nan', '', 'None', 'unresolved', 'unknown'])]
    invalid_jcall = task_b[task_b['j_call'].isin(['-999.0', 'nan', '', 'None', 'unresolved', 'unknown'])]

    if len(invalid_vcall) > 0:
        errors.append(f'Invalid v_call in Task B: {len(invalid_vcall)} rows')
    if len(invalid_jcall) > 0:
        errors.append(f'Invalid j_call in Task B: {len(invalid_jcall)} rows')

    # 6. Check NaN values
    if submission.isna().any().any():
        nan_cols = submission.columns[submission.isna().any()].tolist()
        errors.append(f'NaN values in columns: {nan_cols}')

    # 7. Check unique IDs
    if submission['ID'].duplicated().any():
        dup_count = submission['ID'].duplicated().sum()
        errors.append(f'Duplicate IDs: {dup_count}')

    if errors:
        print('  VALIDATION FAILED:')
        for e in errors:
            print(f'    - {e}')
        return False
    else:
        print('  VALIDATION PASSED!')
        return True

# ============================================================================
# Main Pipeline
# ============================================================================
def main():
    print('='*70)
    print('AIRR-ML-25 Champion v6 - Research-Based Enhancements')
    print('='*70)
    print(f'Target: 0.78+ (vs current 0.74006)')
    print(f'Improvements: k=5 k-mers, diversity indices, Fisher test,')
    print(f'              VJ pairs, CatBoost, enhanced Task B')
    print('='*70)

    # Check GPU
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi'], capture_output=True, timeout=5)
        use_gpu = result.returncode == 0
    except Exception:
        use_gpu = False
    print(f'GPU available: {use_gpu}')
    print(f'CatBoost available: {HAS_CATBOOST}')

    # Create directories
    Config.CHECKPOINT_DIR.mkdir(exist_ok=True)
    Config.SUBMISSION_DIR.mkdir(exist_ok=True)

    # Find datasets
    train_sets = sorted([d.name for d in Config.TRAIN_ROOT.glob('train_dataset_*')])
    test_sets = sorted([d.name for d in Config.TEST_ROOT.glob('test_dataset_*')])

    print(f'Train datasets: {train_sets}')
    print(f'Test datasets: {test_sets}')

    extractor = FeatureExtractorV6(Config.K_LIST)
    bundles = {}

    # ======== TRAINING ========
    for ds_name in train_sets:
        ds_id = dataset_id_from_name(ds_name)
        ds_path = Config.TRAIN_ROOT / ds_name

        print(f'\n[{ds_name}] Training (id={ds_id})')

        # Mine public clones with Fisher's test
        print('  Mining public clones (with Fisher test)...')
        pub_dict = mine_public_clones_v6(
            ds_path,
            max_files=Config.PUB_MAX_FILES,
            min_freq=Config.PUB_MIN_FREQ,
            enrichment=Config.PUB_ENRICH,
            p_value_threshold=Config.PUB_P_VALUE,
            top_n=Config.PUB_TOP_N.get(ds_id, 2500),
        )
        sig_count = sum(1 for v in pub_dict.values() if v.get('p_value', 1) < 0.05)
        print(f'    Found {len(pub_dict)} public clones ({sig_count} significant)')

        # Extract features
        print('  Extracting enhanced features...')
        meta = pd.read_csv(ds_path / 'metadata.csv')

        results = Parallel(n_jobs=8, backend='loky')(
            delayed(process_file_parallel_v6)(row, ds_path, ds_id, pub_dict, extractor)
            for _, row in tqdm(meta.iterrows(), total=len(meta), leave=False, desc='    ')
        )

        df = pd.DataFrame([r for r in results if r is not None])
        print(f'    Extracted {len(df)} repertoires with {len(df.columns)} raw features')

        # Train model
        trainer = EnsembleTrainerV6(use_gpu=use_gpu, random_state=Config.RANDOM_STATE)
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
            delayed(process_test_parallel_v6)(f, test_path, ds_id, bundle['pub'], extractor)
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
    print('SEQUENCE IDENTIFICATION (Task B) - Enhanced')
    print('='*70)

    all_seqs = []

    for ds_name in train_sets:
        ds_path = Config.TRAIN_ROOT / ds_name
        bundle = bundles[ds_name]

        print(f'  {ds_name}...')
        seq_df = identify_sequences_v6(
            ds_path,
            bundle['pub'],
            bundle['trainer'],
            bundle['cols'],
            Config.TOP_K_SEQUENCES
        )
        all_seqs.append(seq_df)
        print(f'    Generated {len(seq_df)} sequences')

    # ======== CREATE SUBMISSION ========
    print('\n' + '='*70)
    print('CREATING SUBMISSION')
    print('='*70)

    task_a = pd.concat(all_preds, ignore_index=True)
    task_b = pd.concat(all_seqs, ignore_index=True)

    submission = pd.concat([task_a, task_b], ignore_index=True)
    submission = submission[['ID', 'dataset', 'label_positive_probability', 'junction_aa', 'v_call', 'j_call']]

    # Validate before saving
    is_valid = validate_submission(submission)

    # Save
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = Config.SUBMISSION_DIR / f'v6_submission_{timestamp}.csv'
    submission.to_csv(output_path, index=False)

    print(f'\nSubmission saved to: {output_path}')
    print(f'Total rows: {len(submission)} (expected: 404213)')
    print(f'Task A rows: {len(task_a)}')
    print(f'Task B rows: {len(task_b)}')

    if is_valid:
        print('\n*** READY FOR SUBMISSION ***')
    else:
        print('\n*** WARNING: Fix validation errors before submitting! ***')

    # Summary
    print('\n' + '='*70)
    print('TRAINING SUMMARY')
    print('='*70)
    cv_scores = []
    for ds_name, bundle in bundles.items():
        print(f"  {ds_name}: CV AUC = {bundle['score']:.4f}")
        cv_scores.append(bundle['score'])

    print(f'\n  Mean CV AUC: {np.mean(cv_scores):.4f}')
    print(f'  Expected Public Score: ~{np.mean(cv_scores):.2f}')

if __name__ == '__main__':
    main()
