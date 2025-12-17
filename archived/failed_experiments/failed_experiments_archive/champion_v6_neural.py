#!/usr/bin/env python3
"""
AIRR-ML-25 Champion V6 - Neural Network with V5 Feature Engineering
====================================================================
Combines proven feature engineering from V5 with neural network classifier.

Key features (from V5):
1. Public clone mining (sequences enriched in positive samples)
2. Positional k-mers (start/end of sequences)
3. Physicochemical properties
4. V/J gene usage
5. Diversity metrics

Model: Deep MLP with dropout and batch normalization (GPU accelerated)
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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from joblib import Parallel, delayed

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

warnings.filterwarnings('ignore')

# ============================================================================
# Configuration
# ============================================================================
class Config:
    PROJECT_ROOT = Path(__file__).parent
    TRAIN_ROOT = PROJECT_ROOT / 'data/train_datasets/train_datasets'
    TEST_ROOT = PROJECT_ROOT / 'data/test_datasets/test_datasets'
    SAMPLE_SUBMISSION = PROJECT_ROOT / 'data/sample_submissions.csv'
    SUBMISSION_DIR = PROJECT_ROOT / 'submissions'

    # Feature settings (from V5)
    K_LIST = [3, 4]
    TOP_FEATURES = 500
    MAX_SEQUENCES_PER_FILE = 50000

    # Public clone settings
    PUB_MAX_FILES = 30
    PUB_MIN_FREQ = 0.15
    PUB_ENRICH = 5.0
    PUB_TOP_N = {1: 2000, 2: 2000, 3: 2000, 4: 2000, 5: 2000,
                 6: 2000, 7: 5000, 8: 3000}

    # Neural network settings
    HIDDEN_DIMS = [512, 256, 128, 64]
    DROPOUT = 0.3
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    MAX_EPOCHS = 200
    PATIENCE = 20

    # Per-dataset class weights
    SCALE_POS_WEIGHT = {
        1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0,
        7: 5.0, 8: 2.0
    }

    RANDOM_STATE = 42
    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

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
# Neural Network Model
# ============================================================================
class DeepMLP(nn.Module):
    """Deep MLP with batch normalization and dropout."""

    def __init__(self, input_dim: int, hidden_dims: List[int] = [512, 256, 128, 64],
                 dropout: float = 0.3):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        self.features = nn.Sequential(*layers)
        self.output = nn.Linear(prev_dim, 1)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        h = self.features(x)
        return self.output(h)

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
# Public Clone Mining (from V5)
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
# Feature Extraction (from V5)
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

        # 8) Public clone features
        if pub_dict:
            seq_set = set(seqs)
            hits = [pub_dict[s]['score'] for s in seq_set if s in pub_dict]
            features['pub_score_sum'] = float(sum(hits))
            features['pub_score_max'] = float(max(hits)) if hits else 0.0
            features['pub_hits'] = float(len(hits))
            features['pub_hit_ratio'] = float(len(hits) / len(seq_set)) if seq_set else 0.0

        return features

# ============================================================================
# Neural Network Trainer
# ============================================================================
class NeuralTrainer:
    """GPU-accelerated neural network trainer."""

    def __init__(self, device: str = 'cuda:0'):
        self.device = torch.device(device)
        self.model = None
        self.scaler = StandardScaler()
        self.feature_cols = []

    def select_top_features(self, X_df: pd.DataFrame, y: np.ndarray, top_k: int = 500) -> List[str]:
        """Select top features using variance and correlation."""
        print(f'    Selecting top {top_k} features...', end=' ')

        # Remove constant features
        variances = X_df.var()
        non_constant = variances[variances > 1e-10].index.tolist()

        # Compute correlations with target
        correlations = {}
        for col in non_constant:
            corr = abs(np.corrcoef(X_df[col].fillna(0).values, y)[0, 1])
            if not np.isnan(corr):
                correlations[col] = corr

        # Sort by correlation
        sorted_feats = sorted(correlations.items(), key=lambda x: -x[1])
        selected = [f[0] for f in sorted_feats[:top_k]]

        # Fill with remaining if needed
        if len(selected) < top_k:
            remaining = [c for c in non_constant if c not in selected]
            selected.extend(remaining[:top_k - len(selected)])

        print(f'Done ({len(selected)} features)')
        return selected

    def train(self, df: pd.DataFrame, ds_id: int) -> Tuple['NeuralTrainer', List[str], float]:
        """Train neural network model with cross-validation."""
        y = df['label_positive'].values.astype(np.float32)
        X_df = df.drop(columns=['ID', 'dataset', 'label_positive'], errors='ignore')

        # Feature selection
        self.feature_cols = self.select_top_features(X_df, y, top_k=Config.TOP_FEATURES)
        X = X_df[self.feature_cols].fillna(0).values.astype(np.float32)

        # Scale features
        X = self.scaler.fit_transform(X)

        print(f'    Training neural network on {len(X)} samples, {len(self.feature_cols)} features')

        # Calculate class weight
        pos_weight = Config.SCALE_POS_WEIGHT.get(ds_id, 1.0)

        # Single train with all data (no CV for speed)
        self.model = DeepMLP(
            input_dim=len(self.feature_cols),
            hidden_dims=Config.HIDDEN_DIMS,
            dropout=Config.DROPOUT
        ).to(self.device)

        # Create dataset
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True)

        # Optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=Config.LEARNING_RATE,
                         weight_decay=Config.WEIGHT_DECAY)
        scheduler = CosineAnnealingLR(optimizer, T_max=Config.MAX_EPOCHS)

        # Class weight for loss
        pos_weight_tensor = torch.tensor([pos_weight], device=self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

        # Training loop
        self.model.train()
        for epoch in range(Config.MAX_EPOCHS):
            total_loss = 0
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            scheduler.step()

        # Evaluate on training data
        self.model.eval()
        with torch.no_grad():
            all_preds = torch.sigmoid(self.model(X_tensor.to(self.device))).cpu().numpy().flatten()

        train_auc = roc_auc_score(y, all_preds)
        print(f'    Train AUC: {train_auc:.4f}')

        return self, self.feature_cols, train_auc

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities."""
        X = self.scaler.transform(X.astype(np.float32))
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)

        self.model.eval()
        with torch.no_grad():
            probs = torch.sigmoid(self.model(X_tensor)).cpu().numpy().flatten()

        return probs

# ============================================================================
# Parallel Processing
# ============================================================================
def process_file_parallel(row, path: Path, ds_id: int, pub_dict: Dict, extractor: FeatureExtractor):
    """Process a single repertoire file."""
    try:
        df = read_repertoire(path / row['filename'], Config.MAX_SEQUENCES_PER_FILE)
        feats = extractor.extract_all(df, pub_dict, row, ds_id)
        feats['ID'] = row['repertoire_id']
        feats['label_positive'] = row.get('label_positive', np.nan)
        return feats
    except Exception as e:
        return {'ID': row.get('repertoire_id', ''), 'label_positive': np.nan}

# ============================================================================
# Main Pipeline
# ============================================================================
class ChampionV6Neural:
    """Main pipeline combining V5 features with neural network."""

    def __init__(self):
        self.models: Dict[int, NeuralTrainer] = {}
        self.feature_cols: Dict[int, List[str]] = {}
        self.pub_dicts: Dict[int, Dict] = {}
        self.extractor = FeatureExtractor(k_list=Config.K_LIST)

    def load_train_dataset(self, ds_id: int) -> pd.DataFrame:
        """Load and process a training dataset."""
        ds_name = f'train_dataset_{ds_id}'
        ds_path = Config.TRAIN_ROOT / ds_name

        print(f'\n📦 Loading {ds_name}...')

        # Mine public clones
        top_n = Config.PUB_TOP_N.get(ds_id, 2000)
        pub_dict = mine_public_clones(
            ds_path,
            max_files=Config.PUB_MAX_FILES,
            min_freq=Config.PUB_MIN_FREQ,
            enrichment=Config.PUB_ENRICH,
            top_n=top_n
        )
        self.pub_dicts[ds_id] = pub_dict
        print(f'  Public clones: {len(pub_dict)}')

        # Load metadata
        meta = pd.read_csv(ds_path / 'metadata.csv')

        # Extract features in parallel
        results = Parallel(n_jobs=8, prefer='threads')(
            delayed(process_file_parallel)(row, ds_path, ds_id, pub_dict, self.extractor)
            for _, row in tqdm(meta.iterrows(), total=len(meta), desc='  Extracting features')
        )

        df = pd.DataFrame(results)
        df['dataset'] = ds_name

        print(f'  Loaded {len(df)} repertoires, {len(df.columns)} features')
        return df

    def run_lodo_cv(self):
        """Run Leave-One-Dataset-Out cross-validation."""
        print('\n' + '='*60)
        print('🎓 LODO CROSS-VALIDATION WITH NEURAL NETWORK')
        print('='*60)

        # Load all datasets
        all_dfs = []
        for ds_id in range(1, 9):
            df = self.load_train_dataset(ds_id)
            df['ds_id'] = ds_id
            all_dfs.append(df)

        # Combine
        full_df = pd.concat(all_dfs, ignore_index=True)
        print(f'\n✅ Total: {len(full_df)} repertoires')

        # LODO CV
        fold_aucs = []
        for val_ds_id in range(1, 9):
            print(f'\n{"="*50}')
            print(f'🎯 FOLD {val_ds_id}: Validate on dataset {val_ds_id}')
            print(f'{"="*50}')

            # Split
            train_mask = full_df['ds_id'] != val_ds_id
            val_mask = full_df['ds_id'] == val_ds_id

            train_df = full_df[train_mask].copy()
            val_df = full_df[val_mask].copy()

            print(f'  Train: {len(train_df)}, Val: {len(val_df)}')

            # Prepare features
            y_train = train_df['label_positive'].values.astype(np.float32)
            y_val = val_df['label_positive'].values.astype(np.float32)

            # Drop metadata columns
            drop_cols = ['ID', 'dataset', 'label_positive', 'ds_id']
            X_train_df = train_df.drop(columns=drop_cols, errors='ignore')
            X_val_df = val_df.drop(columns=drop_cols, errors='ignore')

            # Align columns
            all_cols = list(set(X_train_df.columns) | set(X_val_df.columns))
            for col in all_cols:
                if col not in X_train_df.columns:
                    X_train_df[col] = 0.0
                if col not in X_val_df.columns:
                    X_val_df[col] = 0.0
            X_val_df = X_val_df[X_train_df.columns]

            # Train
            trainer = NeuralTrainer(device=Config.DEVICE)

            # Temporarily add label back for training
            train_df_temp = X_train_df.copy()
            train_df_temp['label_positive'] = y_train

            trainer, feature_cols, _ = trainer.train(train_df_temp, val_ds_id)

            # Predict
            X_val = X_val_df[feature_cols].fillna(0).values
            preds = trainer.predict(X_val)

            # Evaluate
            auc = roc_auc_score(y_val, preds)
            fold_aucs.append(auc)
            print(f'  🎯 Fold {val_ds_id} AUC: {auc:.4f}')

        mean_auc = np.mean(fold_aucs)
        std_auc = np.std(fold_aucs)
        print(f'\n{"="*60}')
        print(f'🏆 LODO CV Results:')
        for i, auc in enumerate(fold_aucs, 1):
            print(f'   Fold {i}: {auc:.4f}')
        print(f'\n🎯 Mean LODO AUC: {mean_auc:.4f} ± {std_auc:.4f}')
        print(f'{"="*60}')

        return mean_auc, fold_aucs

# ============================================================================
# Main
# ============================================================================
if __name__ == '__main__':
    print('🔥' * 30)
    print('  CHAMPION V6: NEURAL NETWORK + V5 FEATURES')
    print('🔥' * 30)
    print(f'  Device: {Config.DEVICE}')
    print(f'  Hidden dims: {Config.HIDDEN_DIMS}')
    print(f'  Dropout: {Config.DROPOUT}')
    print('🔥' * 30)

    np.random.seed(Config.RANDOM_STATE)
    torch.manual_seed(Config.RANDOM_STATE)

    pipeline = ChampionV6Neural()
    mean_auc, fold_aucs = pipeline.run_lodo_cv()

    print(f'\n✅ LODO CV Complete')
    print(f'   Mean AUC: {mean_auc:.4f}')
