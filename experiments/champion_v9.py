#!/usr/bin/env python3
"""
AIRR-ML-25 Champion v9 - Research-Enhanced Pipeline
====================================================
基於最新研究調研結果的增強版本：

新增特性 (相對於 v5):
1. Atchley Factors - 氨基酸理化性質編碼 (Science 2025 Mal-ID)
2. 改進的多樣性指標 - Shannon, Simpson, Gini, D50, Clonality
3. Fisher Exact Test - 統計顯著的 public clone mining (Emerson 2017)
4. 改進的 Task B - TF-IDF + 模型重要性結合

保留 v5 優勢:
- K-mer 特徵 (k=3,4)
- XGBoost + LightGBM GPU 集成
- Per-dataset 權重
- Public clone 特徵

目標: 0.74006 -> 0.80+
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
from scipy.stats import entropy, fisher_exact
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from joblib import Parallel, delayed

import xgboost as xgb
import lightgbm as lgb

# Numba JIT for acceleration
try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if args and callable(args[0]) is False else decorator(args[0]) if args else decorator
    prange = range

warnings.filterwarnings('ignore')

# ============================================================================
# Numba-Accelerated K-mer Counting
# ============================================================================
if NUMBA_AVAILABLE:
    @njit(cache=True, parallel=True)
    def count_kmers_numba(seq_array: np.ndarray, seq_lengths: np.ndarray, k: int) -> np.ndarray:
        """Numba-accelerated k-mer counting (returns hash array)"""
        n_seqs = len(seq_lengths)
        # Pre-allocate for max possible k-mers
        max_kmers = int(seq_lengths.sum()) * k
        hashes = np.zeros(max_kmers, dtype=np.uint64)
        idx = 0

        for i in prange(n_seqs):
            start = int(seq_lengths[:i].sum()) if i > 0 else 0
            length = seq_lengths[i]
            if length < k:
                continue
            for j in range(length - k + 1):
                # Simple hash for k-mer
                h = np.uint64(0)
                for m in range(k):
                    h = h * 31 + seq_array[start + j + m]
                hashes[idx] = h
                idx += 1

        return hashes[:idx]

# ============================================================================
# Configuration
# ============================================================================
class Config:
    TRAIN_ROOT = Path('./data/train_datasets/train_datasets')
    TEST_ROOT = Path('./data/test_datasets/test_datasets')
    SAMPLE_SUBMISSION = Path('./data/sample_submissions.csv')
    SUBMISSION_DIR = Path('./submissions')
    CHECKPOINT_DIR = Path('./checkpoints_v9')

    # Feature settings
    K_LIST = [3, 4]  # 保持 v5 的最佳設定
    TOP_KMER = 600   # 略增以容納新特徵
    MAX_SEQUENCES_PER_FILE = 50000

    # Public clone settings (使用 Fisher test)
    PUB_MAX_FILES = 40
    PUB_P_THRESHOLD = 0.01  # Fisher exact test p-value 閾值
    PUB_MIN_FREQ = 0.10     # 最低頻率閾值
    PUB_TOP_N = {1: 2500, 2: 2500, 3: 2500, 4: 2500, 5: 2500,
                 6: 2500, 7: 6000, 8: 4000}

    # Training settings
    N_SPLITS = 5
    RANDOM_STATE = 42
    EARLY_STOP = 100

    # Per-dataset class weights (保持 v5)
    SCALE_POS_WEIGHT = {
        1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0,
        7: 5.0,  # Imbalanced
        8: 2.0   # Imbalanced
    }

    # Top K sequences for Task B
    TOP_K_SEQUENCES = 50000

    # Performance settings
    N_JOBS = 12  # Parallel workers (use more for faster extraction)
    BATCH_SIZE = 50  # Files to process in a batch
    USE_NUMBA = NUMBA_AVAILABLE  # Use Numba acceleration if available


# ============================================================================
# Atchley Factors - 氨基酸理化性質編碼
# ============================================================================
# 5 個因子: PAF, PSS, POS, COD, ENT (取自 Atchley et al. 2005)
ATCHLEY_FACTORS = {
    'A': np.array([-0.591, -1.302, -0.733,  1.570, -0.146]),
    'C': np.array([-1.343,  0.465, -0.862, -1.020, -0.255]),
    'D': np.array([ 1.050,  0.302, -3.656, -0.259, -3.242]),
    'E': np.array([ 1.357, -1.453,  1.477,  0.113, -0.837]),
    'F': np.array([-1.006, -0.590,  1.891, -0.397,  0.412]),
    'G': np.array([-0.384,  1.652,  1.330,  1.045,  2.064]),
    'H': np.array([ 0.336, -0.417, -1.673, -1.474, -0.078]),
    'I': np.array([-1.239, -0.547,  2.131,  0.393,  0.816]),
    'K': np.array([ 1.831, -0.561,  0.533, -0.277,  1.648]),
    'L': np.array([-1.019, -0.987, -1.505,  1.266, -0.912]),
    'M': np.array([-0.663, -1.524,  2.219, -1.005,  1.212]),
    'N': np.array([ 0.945,  0.828,  1.299, -0.169,  0.933]),
    'P': np.array([ 0.189,  2.081, -1.628,  0.421, -1.392]),
    'Q': np.array([ 0.931, -0.179, -3.005, -0.503, -1.853]),
    'R': np.array([ 1.538, -0.055,  1.502,  0.440,  2.897]),
    'S': np.array([-0.228,  1.399, -4.760,  0.670, -2.647]),
    'T': np.array([-0.032,  0.326,  2.213,  0.908,  1.313]),
    'V': np.array([-1.337, -0.279, -0.544,  1.242, -1.262]),
    'W': np.array([-0.595,  0.009,  0.672, -2.128, -0.184]),
    'Y': np.array([ 0.260,  0.830,  3.097, -0.838,  1.512]),
}

ATCHLEY_FACTOR_NAMES = ['PAF', 'PSS', 'POS', 'COD', 'ENT']

# 保持 v5 的 AA_PROPERTIES 用於 hydro/vol/charge
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
    """Read a repertoire TSV file with optional sampling (optimized)."""
    cols = ['junction_aa', 'v_call', 'j_call', 'templates']
    try:
        # Read header first to get available columns
        header = pd.read_csv(tsv_path, sep='\t', nrows=0)
        usecols = [c for c in cols if c in header.columns]

        # Use faster C engine with optimized dtypes
        df = pd.read_csv(
            tsv_path, sep='\t', usecols=usecols,
            dtype={'junction_aa': str, 'v_call': str, 'j_call': str},
            engine='c', low_memory=True
        )
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


def read_repertoire_batch(file_list: List[Path], max_seqs: int = 50000) -> List[pd.DataFrame]:
    """Batch read multiple repertoire files in parallel."""
    return Parallel(n_jobs=Config.N_JOBS, backend='loky', prefer='processes')(
        delayed(read_repertoire)(f, max_seqs) for f in file_list
    )


# ============================================================================
# Diversity Metrics (改進版)
# ============================================================================
def calculate_diversity_metrics(clone_counts: np.ndarray) -> Dict[str, float]:
    """計算多樣性指標 (Shannon, Simpson, Gini, D50, Clonality)"""
    if len(clone_counts) == 0 or clone_counts.sum() == 0:
        return {
            'shannon': 0.0, 'simpson': 1.0, 'gini': 0.0,
            'd50': 0.0, 'clonality': 0.0, 'richness': 0.0
        }

    total = clone_counts.sum()
    freqs = clone_counts / total

    # Shannon entropy (normalized)
    shannon = -np.sum(freqs * np.log2(freqs + 1e-10))
    max_shannon = np.log2(len(freqs)) if len(freqs) > 1 else 1.0

    # Simpson index
    simpson = np.sum(freqs ** 2)

    # Gini coefficient
    sorted_freqs = np.sort(freqs)
    n = len(sorted_freqs)
    cumsum = np.cumsum(sorted_freqs)
    gini = (2.0 * np.sum((np.arange(1, n + 1) * sorted_freqs))) / (n * np.sum(sorted_freqs)) - (n + 1) / n

    # D50 (fraction of clones covering 50% of repertoire)
    sorted_desc = np.sort(freqs)[::-1]
    cumsum_desc = np.cumsum(sorted_desc)
    d50_idx = np.searchsorted(cumsum_desc, 0.5) + 1
    d50 = d50_idx / len(freqs)

    # Clonality (1 - normalized Shannon)
    clonality = 1 - (shannon / max_shannon) if max_shannon > 0 else 0.0

    return {
        'shannon': shannon,
        'simpson': simpson,
        'gini': max(0, gini),  # 確保非負
        'd50': d50,
        'clonality': clonality,
        'richness': float(len(freqs)),
    }


# ============================================================================
# Public Clone Mining with Fisher Exact Test
# ============================================================================
def mine_public_clones_fisher(
    dataset_path: Path,
    max_files: int = 40,
    p_threshold: float = 0.01,
    min_freq: float = 0.10,
    top_n: int = 2500
) -> Dict[str, Dict]:
    """使用 Fisher exact test 挖掘統計顯著的 public clones"""
    meta = pd.read_csv(dataset_path / 'metadata.csv')
    pos_files = meta[meta['label_positive'] == True]['filename'].tolist()[:max_files]
    neg_files = meta[meta['label_positive'] == False]['filename'].tolist()[:max_files]

    if not pos_files:
        return {}

    n_pos = len(pos_files)
    n_neg = len(neg_files)

    # 計算每個序列在 pos/neg 中出現的 repertoire 數
    pos_counts = Counter()
    neg_counts = Counter()

    for f in pos_files:
        try:
            df = pd.read_csv(dataset_path / f, sep='\t', usecols=['junction_aa'])
            unique_seqs = set(df['junction_aa'].dropna().astype(str).unique())
            pos_counts.update(unique_seqs)
        except Exception:
            pass

    for f in neg_files:
        try:
            df = pd.read_csv(dataset_path / f, sep='\t', usecols=['junction_aa'])
            unique_seqs = set(df['junction_aa'].dropna().astype(str).unique())
            neg_counts.update(unique_seqs)
        except Exception:
            pass

    # Fisher exact test
    all_seqs = set(pos_counts.keys()) | set(neg_counts.keys())
    significant = []

    for seq in all_seqs:
        if len(seq) < 8:  # 過短的序列跳過
            continue

        pos_c = pos_counts.get(seq, 0)
        neg_c = neg_counts.get(seq, 0)

        # 頻率過濾
        pos_freq = pos_c / n_pos if n_pos > 0 else 0
        neg_freq = neg_c / n_neg if n_neg > 0 else 0

        if pos_freq < min_freq and neg_freq < min_freq:
            continue

        # Fisher exact test
        table = [[pos_c, n_pos - pos_c],
                 [neg_c, n_neg - neg_c]]
        try:
            odds_ratio, p_value = fisher_exact(table)
        except Exception:
            continue

        if p_value < p_threshold:
            # 計算 score: -log10(p) * sign(log(odds))
            log_odds = np.log(odds_ratio + 1e-10)
            score = -np.log10(p_value + 1e-10) * np.sign(log_odds)

            significant.append({
                'seq': seq,
                'score': score,
                'p_value': p_value,
                'odds_ratio': odds_ratio,
                'pos_freq': pos_freq,
                'neg_freq': neg_freq,
                'direction': 'positive' if odds_ratio > 1 else 'negative'
            })

    # 按分數排序，取 top_n
    significant.sort(key=lambda x: -abs(x['score']))
    return {item['seq']: item for item in significant[:top_n]}


# ============================================================================
# Feature Extraction (Enhanced)
# ============================================================================
class FeatureExtractor:
    """增強版特徵提取器 - 整合 Atchley factors 和改進的多樣性指標"""

    def __init__(self, k_list: List[int] = [3, 4]):
        self.k_list = k_list

    def gene_family(self, gene_call: str) -> str:
        """Extract gene family from call like TRBV20-1*01 -> TRBV20"""
        if not isinstance(gene_call, str) or not gene_call:
            return 'UNK'
        return gene_call.split('*')[0].split('-')[0].upper() or 'UNK'

    def extract_atchley_features(self, seqs: List[str]) -> Dict[str, float]:
        """提取 Atchley factor 特徵"""
        all_factors = []

        for seq in seqs:
            factors = [ATCHLEY_FACTORS.get(aa, np.zeros(5)) for aa in seq if aa in ATCHLEY_FACTORS]
            if factors:
                all_factors.append(np.mean(factors, axis=0))

        if not all_factors:
            features = {}
            for i, name in enumerate(ATCHLEY_FACTOR_NAMES):
                features[f'atchley_{name}_mean'] = 0.0
                features[f'atchley_{name}_std'] = 0.0
            return features

        arr = np.array(all_factors)
        features = {}
        for i, name in enumerate(ATCHLEY_FACTOR_NAMES):
            features[f'atchley_{name}_mean'] = float(np.mean(arr[:, i]))
            features[f'atchley_{name}_std'] = float(np.std(arr[:, i]))
            features[f'atchley_{name}_min'] = float(np.min(arr[:, i]))
            features[f'atchley_{name}_max'] = float(np.max(arr[:, i]))

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

        # 1) K-mers (保持 v5 方法)
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

        # 2) Positional k-mers
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

        # 3) Atchley Factors (新增)
        atchley_feats = self.extract_atchley_features(seqs)
        features.update(atchley_feats)

        # 4) 原有的 physicochemical (保留用於比較)
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

        # 5) V gene families
        if 'v_call' in df.columns:
            v_fam = df['v_call'].apply(self.gene_family)
            for fam, freq in v_fam.value_counts(normalize=True).head(40).items():
                features[f'v_fam_{fam}'] = float(freq)

        # 6) J gene families
        if 'j_call' in df.columns:
            j_fam = df['j_call'].apply(self.gene_family)
            for fam, freq in j_fam.value_counts(normalize=True).head(20).items():
                features[f'j_fam_{fam}'] = float(freq)

        # 7) Length statistics
        lens = [len(s) for s in seqs]
        if lens:
            features['len_mean'] = float(np.mean(lens))
            features['len_std'] = float(np.std(lens))
            features['len_min'] = float(np.min(lens))
            features['len_max'] = float(np.max(lens))
            features['len_p25'] = float(np.percentile(lens, 25))
            features['len_p75'] = float(np.percentile(lens, 75))

        # 8) Improved Diversity Metrics (新增/改進)
        features['n_unique_seqs'] = float(len(set(seqs)))
        features['n_total_seqs'] = float(len(seqs))
        if len(seqs) > 0:
            features['diversity_ratio'] = features['n_unique_seqs'] / features['n_total_seqs']

        # Clone 多樣性 (使用 templates)
        if 'templates' in df.columns:
            temps = df['templates'].values
            if temps.sum() > 0:
                diversity_metrics = calculate_diversity_metrics(temps)
                for k, v in diversity_metrics.items():
                    features[f'div_{k}'] = v

        # 9) Metadata features
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

        # 10) Public clone features (使用 Fisher test 結果)
        if pub_dict:
            seq_set = set(seqs)
            pos_hits = []
            neg_hits = []

            for s in seq_set:
                if s in pub_dict:
                    info = pub_dict[s]
                    if info['direction'] == 'positive':
                        pos_hits.append(info['score'])
                    else:
                        neg_hits.append(info['score'])

            features['pub_pos_score_sum'] = float(sum(pos_hits))
            features['pub_pos_score_max'] = float(max(pos_hits)) if pos_hits else 0.0
            features['pub_pos_hits'] = float(len(pos_hits))
            features['pub_neg_score_sum'] = float(sum(neg_hits))
            features['pub_neg_score_max'] = float(min(neg_hits)) if neg_hits else 0.0
            features['pub_neg_hits'] = float(len(neg_hits))
            features['pub_net_score'] = features['pub_pos_score_sum'] - abs(features['pub_neg_score_sum'])
            features['pub_hit_ratio'] = float(len(pos_hits) + len(neg_hits)) / len(seq_set) if seq_set else 0.0

        return features


# ============================================================================
# Ensemble Trainer (保持 v5 架構)
# ============================================================================
class EnsembleTrainer:
    """XGBoost + LightGBM ensemble with GPU support."""

    def __init__(self, use_gpu: bool = True, random_state: int = 42):
        self.use_gpu = use_gpu
        self.random_state = random_state
        self.models = {}
        self.weights = {'xgb': 0.5, 'lgb': 0.5}
        self.feature_cols = []
        self.feature_importance = {}

    def select_features_gpu(self, X_df: pd.DataFrame, y: np.ndarray, top_k: int = 600):
        """Select top features using GPU XGBoost (with memory protection)."""
        print(f'    Selecting top {top_k} features...', end=' ')
        all_cols = X_df.columns.tolist()

        # Memory protection: if too many features, pre-filter using variance
        max_gpu_features = 50000  # Safe limit for 16GB GPU
        if len(all_cols) > max_gpu_features:
            print(f'(pre-filtering {len(all_cols)} -> {max_gpu_features})...', end=' ')
            # Use variance to select initial features
            variances = X_df.var()
            top_var_cols = variances.nlargest(max_gpu_features).index.tolist()
            X_df = X_df[top_var_cols]
            all_cols = top_var_cols

        try:
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

        except Exception as e:
            print(f'GPU failed ({e}), using variance selection...', end=' ')
            # Fallback: use variance-based selection
            variances = X_df.var()
            selected = variances.nlargest(top_k).index.tolist()

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

        # Determine n_splits
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

        # Stacking weights
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

        # Store feature importance for Task B
        xgb_imp = self.models['xgb'].get_score(importance_type='gain')
        for i, col in enumerate(self.feature_cols):
            self.feature_importance[col] = xgb_imp.get(f'f{i}', 0)

        return self, self.feature_cols, max(mean_xgb, mean_lgb)

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
# Task B: Improved Sequence Identification
# ============================================================================
def identify_sequences_improved(
    dataset_path: Path,
    pub_dict: Dict,
    feature_importance: Dict,
    top_k: int = 50000
) -> pd.DataFrame:
    """改進的序列識別 - 結合多種信號"""
    meta = pd.read_csv(dataset_path / 'metadata.csv')
    ds_name = dataset_path.name

    seq_scores = {}
    seq_info = {}  # 儲存 v_call, j_call

    # 1. Public clone scores (from Fisher test)
    for seq, info in pub_dict.items():
        if info['direction'] == 'positive':  # 只取正向關聯
            seq_scores[seq] = info['score'] * 2.0  # 加權

    # 2. 從 positive repertoires 收集序列
    pos_files = meta[meta['label_positive'] == True]['filename'].tolist()
    neg_files = meta[meta['label_positive'] == False]['filename'].tolist()

    all_pos_seqs = Counter()
    all_neg_seqs = Counter()

    for f in pos_files[:50]:
        try:
            df = pd.read_csv(dataset_path / f, sep='\t', usecols=['junction_aa', 'v_call', 'j_call'])
            for _, row in df.iterrows():
                seq = str(row['junction_aa'])
                if seq and len(seq) >= 8:
                    all_pos_seqs[seq] += 1
                    if seq not in seq_info:
                        seq_info[seq] = {
                            'v_call': str(row.get('v_call', '-999.0')),
                            'j_call': str(row.get('j_call', '-999.0'))
                        }
        except Exception:
            continue

    for f in neg_files[:30]:
        try:
            df = pd.read_csv(dataset_path / f, sep='\t', usecols=['junction_aa'])
            for seq in df['junction_aa'].dropna():
                all_neg_seqs[str(seq)] += 1
        except Exception:
            continue

    # 3. 計算差異分數
    for seq, pos_count in all_pos_seqs.items():
        neg_count = all_neg_seqs.get(seq, 0)
        diff_score = (pos_count - neg_count) / (pos_count + neg_count + 1)

        if seq not in seq_scores:
            seq_scores[seq] = diff_score
        else:
            seq_scores[seq] += diff_score

    # 4. K-mer 重要性加權 (使用模型的 feature importance)
    for seq in list(seq_scores.keys())[:100000]:  # 限制計算量
        kmer_score = 0.0
        for k in [3, 4]:
            for i in range(len(seq) - k + 1):
                kmer = seq[i:i+k]
                feat_name = f'kmer_{k}_{kmer}'
                if feat_name in feature_importance:
                    kmer_score += feature_importance[feat_name]
        seq_scores[seq] = seq_scores.get(seq, 0) + kmer_score * 0.1

    # 排序並取 top_k
    sorted_seqs = sorted(seq_scores.items(), key=lambda x: -x[1])[:top_k]

    # 構建結果
    results = []
    for i, (seq, score) in enumerate(sorted_seqs):
        info = seq_info.get(seq, {'v_call': '-999.0', 'j_call': '-999.0'})
        results.append({
            'ID': f'{ds_name}_seq_top_{i+1}',
            'dataset': ds_name,
            'label_positive_probability': -999.0,
            'junction_aa': seq,
            'v_call': info['v_call'],
            'j_call': info['j_call'],
        })

    # 填充
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
    import time
    total_start = time.time()

    print('='*70)
    print('AIRR-ML-25 Champion v9 - Research-Enhanced Pipeline')
    print('='*70)
    print('新增特性: Atchley Factors + Fisher Exact Test + 改進多樣性指標')
    print(f'Numba acceleration: {"ENABLED" if NUMBA_AVAILABLE else "DISABLED"}')
    print(f'Parallel workers: {Config.N_JOBS}')

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

        # Mine public clones with Fisher test
        print('  Mining public clones (Fisher test)...')
        pub_dict = mine_public_clones_fisher(
            ds_path,
            max_files=Config.PUB_MAX_FILES,
            p_threshold=Config.PUB_P_THRESHOLD,
            min_freq=Config.PUB_MIN_FREQ,
            top_n=Config.PUB_TOP_N.get(ds_id, 2500),
        )
        print(f'    Found {len(pub_dict)} significant public clones')

        # Extract features
        print('  Extracting features...')
        meta = pd.read_csv(ds_path / 'metadata.csv')
        feat_start = time.time()

        results = Parallel(n_jobs=Config.N_JOBS, backend='loky', prefer='processes')(
            delayed(process_file_parallel)(row, ds_path, ds_id, pub_dict, extractor)
            for _, row in tqdm(meta.iterrows(), total=len(meta), leave=False, desc='    ')
        )
        print(f'    Feature extraction: {time.time() - feat_start:.1f}s')

        df = pd.DataFrame([r for r in results if r is not None])
        print(f'    Extracted {len(df)} repertoires with {len(df.columns)} raw features')

        # Train model
        trainer = EnsembleTrainer(use_gpu=use_gpu, random_state=Config.RANDOM_STATE)
        trainer, fcols, score = trainer.train(df, ds_id)

        bundles[ds_name] = {
            'trainer': trainer,
            'cols': fcols,
            'pub': pub_dict,
            'score': score,
            'feature_importance': trainer.feature_importance
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
        pred_start = time.time()

        results = Parallel(n_jobs=Config.N_JOBS, backend='loky', prefer='processes')(
            delayed(process_test_parallel)(f, test_path, ds_id, bundle['pub'], extractor)
            for f in tqdm(files, leave=False, desc='    ')
        )
        print(f'    Processed {len(files)} files in {time.time() - pred_start:.1f}s')

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
        seq_df = identify_sequences_improved(
            ds_path,
            bundle['pub'],
            bundle['feature_importance'],
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

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = Config.SUBMISSION_DIR / f'v9_submission_{timestamp}.csv'
    submission.to_csv(output_path, index=False)

    print(f'\nSubmission saved to: {output_path}')
    print(f'Total rows: {len(submission)} (expected: 404213)')
    print(f'Task A rows: {len(task_a)}')
    print(f'Task B rows: {len(task_b)}')

    if len(submission) == 404213:
        print('\nValidation PASSED!')
    else:
        print(f'\nValidation FAILED! Expected 404213, got {len(submission)}')

    # Summary
    print('\n' + '='*70)
    print('TRAINING SUMMARY')
    print('='*70)
    total_score = 0
    for ds_name, bundle in bundles.items():
        print(f"  {ds_name}: CV AUC = {bundle['score']:.4f}")
        total_score += bundle['score']
    print(f'\n  Average CV AUC: {total_score / len(bundles):.4f}')

    total_time = time.time() - total_start
    print(f'\n  Total runtime: {total_time/60:.1f} minutes ({total_time:.0f} seconds)')


if __name__ == '__main__':
    main()
