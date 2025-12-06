#!/usr/bin/env python3
"""
AIRR-ML-25 GPU XGBoost Championship Pipeline
=============================================
Uses GPU-accelerated XGBoost with optimized features.
"""

import os
import sys
import gc
import glob
import warnings
from pathlib import Path
from collections import Counter
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import xgboost as xgb

warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# =============================================================================
# Feature Extraction - Multi-scale k-mers
# =============================================================================

def extract_features(sequences: List[str]) -> Dict[str, float]:
    """Extract multi-scale k-mer features from sequences."""
    features = {}

    # K-mers: k=3,4,5 for better discrimination
    for k in [3, 4, 5]:
        counts = Counter()
        total = 0
        for seq in sequences:
            if isinstance(seq, str) and len(seq) >= k:
                for i in range(len(seq) - k + 1):
                    counts[seq[i:i+k]] += 1
                    total += 1

        # Normalize
        if total > 0:
            for kmer, count in counts.items():
                features[f"k{k}_{kmer}"] = count / total

    # CDR3 length statistics
    lengths = [len(s) for s in sequences if isinstance(s, str)]
    if lengths:
        features['cdr3_len_mean'] = np.mean(lengths)
        features['cdr3_len_std'] = np.std(lengths)
        features['cdr3_len_min'] = np.min(lengths)
        features['cdr3_len_max'] = np.max(lengths)

    return features


def load_repertoire(file_path: str, repertoire_id: str) -> Tuple[str, Dict]:
    """Load a single repertoire and extract features."""
    try:
        df = pd.read_csv(file_path, sep='\t', usecols=['junction_aa'])
        sequences = df['junction_aa'].dropna().tolist()
        features = extract_features(sequences)
        return repertoire_id, features
    except Exception as e:
        return repertoire_id, {}


def load_all_repertoires(data_dir: str, n_jobs: int = 8) -> Tuple:
    """Load all repertoires and create sparse feature matrix."""
    metadata_path = os.path.join(data_dir, 'metadata.csv')

    if os.path.exists(metadata_path):
        metadata_df = pd.read_csv(metadata_path)
        job_args = [
            (os.path.join(data_dir, row.filename), row.repertoire_id)
            for row in metadata_df.itertuples(index=False)
        ]
        labels = {row.repertoire_id: row.label_positive for row in metadata_df.itertuples(index=False)}
    else:
        tsv_files = sorted(glob.glob(os.path.join(data_dir, '*.tsv')))
        job_args = [
            (f, os.path.basename(f).replace('.tsv', ''))
            for f in tsv_files
        ]
        labels = {}

    print(f"  Loading {len(job_args)} repertoires with multi-scale k-mers...")

    # Parallel loading
    results = Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(load_repertoire)(*args) for args in tqdm(job_args, disable=False)
    )

    # Build sparse matrix using DictVectorizer
    rep_ids = [r[0] for r in results]
    features = [r[1] for r in results]

    print(f"  Creating sparse feature matrix...")
    vectorizer = DictVectorizer(sparse=True)
    X = vectorizer.fit_transform(features)

    print(f"  Shape: {X.shape} (sparse)")

    if labels:
        y = np.array([labels.get(rid, 0) for rid in rep_ids])
    else:
        y = None

    return X, y, rep_ids, vectorizer


# =============================================================================
# GPU XGBoost Predictor
# =============================================================================

class GPUXGBoostPredictor:
    """GPU-accelerated XGBoost predictor."""

    def __init__(self, n_jobs: int = 8):
        self.n_jobs = n_jobs
        self.model = None
        self.vectorizer = None
        self.train_dir = None

    def fit(self, train_dir: str):
        """Train model on data in train_dir."""
        self.train_dir = train_dir
        print(f"\n{'#'*60}")
        print(f"TRAINING: {train_dir}")
        print(f"{'#'*60}\n")

        # Load data
        X, y, rep_ids, self.vectorizer = load_all_repertoires(train_dir, n_jobs=self.n_jobs)

        print(f"  Training samples: {len(y)}")
        print(f"  Features: {X.shape[1]}")
        print(f"  Class balance: {np.mean(y):.2%} positive")

        # Split for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Train XGBoost with GPU
        print(f"  Training XGBoost with GPU...")

        # Convert to DMatrix for XGBoost
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'tree_method': 'hist',  # Use hist for speed
            'device': 'cuda',  # GPU acceleration
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 1,
            'gamma': 0,
            'reg_alpha': 0.1,
            'reg_lambda': 1,
            'random_state': 42,
        }

        # Train with early stopping
        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=500,
            evals=[(dval, 'val')],
            early_stopping_rounds=50,
            verbose_eval=False
        )

        # Validate
        y_pred = self.model.predict(dval)
        auc = roc_auc_score(y_val, y_pred)
        print(f"  Validation AUC: {auc:.4f}")
        print(f"  Best iteration: {self.model.best_iteration}")

        return self

    def predict_proba(self, test_dir: str) -> pd.DataFrame:
        """Predict probabilities for test data."""
        dataset_name = os.path.basename(test_dir)
        print(f"\n  Predicting: {dataset_name}")

        # Load test data
        test_files = sorted(glob.glob(os.path.join(test_dir, '*.tsv')))

        predictions = []
        for file_path in tqdm(test_files, desc="Predicting", disable=False):
            rep_id = os.path.basename(file_path).replace('.tsv', '')
            _, features = load_repertoire(file_path, rep_id)

            # Transform to same feature space
            X = self.vectorizer.transform([features])
            dtest = xgb.DMatrix(X)

            # Predict
            prob = self.model.predict(dtest)[0]

            predictions.append({
                'ID': rep_id,
                'dataset': dataset_name,
                'label_positive_probability': prob,
                'junction_aa': -999.0,
                'v_call': -999.0,
                'j_call': -999.0
            })

        print(f"    Got {len(predictions)} predictions")
        return pd.DataFrame(predictions)

    def identify_sequences(self, top_k: int = 50000) -> pd.DataFrame:
        """Identify top disease-associated sequences."""
        dataset_name = os.path.basename(self.train_dir)
        print(f"\n  Identifying sequences for {dataset_name}...")

        # Get feature importances from XGBoost
        importance = self.model.get_score(importance_type='gain')
        feature_names = self.vectorizer.get_feature_names_out()

        # Create importance dict
        kmer_importance = {}
        for fname in feature_names:
            if fname in importance:
                kmer_importance[fname] = importance[fname]

        # Load all sequences
        metadata_path = os.path.join(self.train_dir, 'metadata.csv')
        metadata_df = pd.read_csv(metadata_path)

        all_seqs = []
        for row in tqdm(metadata_df.itertuples(index=False), total=len(metadata_df), desc="Loading sequences"):
            file_path = os.path.join(self.train_dir, row.filename)
            try:
                df = pd.read_csv(file_path, sep='\t', usecols=['junction_aa', 'v_call', 'j_call'])
                all_seqs.append(df)
            except:
                continue

        if not all_seqs:
            return pd.DataFrame()

        full_df = pd.concat(all_seqs, ignore_index=True)
        unique_seqs = full_df[['junction_aa', 'v_call', 'j_call']].drop_duplicates()
        unique_seqs = unique_seqs.dropna(subset=['junction_aa'])

        # Score sequences based on k-mer importance
        scores = []
        for seq in unique_seqs['junction_aa']:
            score = 0.0
            if isinstance(seq, str):
                for k in [3, 4, 5]:
                    if len(seq) >= k:
                        for i in range(len(seq) - k + 1):
                            kmer = f"k{k}_{seq[i:i+k]}"
                            score += kmer_importance.get(kmer, 0)
            scores.append(score)

        unique_seqs = unique_seqs.copy()
        unique_seqs['score'] = scores

        # Get top sequences
        top_seqs = unique_seqs.nlargest(top_k, 'score')

        # Format output
        result = []
        for i, row in enumerate(top_seqs.itertuples(index=False)):
            result.append({
                'ID': f"{dataset_name}_seq_{i+1}",
                'dataset': dataset_name,
                'label_positive_probability': -999.0,
                'junction_aa': row.junction_aa,
                'v_call': row.v_call if pd.notna(row.v_call) else '-999.0',
                'j_call': row.j_call if pd.notna(row.j_call) else '-999.0'
            })

        print(f"    Got {len(result)} sequences")
        return pd.DataFrame(result)


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print("AIRR-ML-25 GPU XGBoost Championship Pipeline")
    print("=" * 60)

    # Check GPU
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                               capture_output=True, text=True)
        print(f"GPU: {result.stdout.strip()}")
    except:
        print("GPU: Not detected, will use CPU fallback")

    # Paths
    train_root = Path("./data/train_datasets/train_datasets")
    test_root = Path("./data/test_datasets/test_datasets")
    out_dir = Path("./results_gpu_xgb")
    out_dir.mkdir(exist_ok=True)

    # Get datasets
    train_datasets = sorted([d for d in train_root.iterdir() if d.is_dir()])
    test_datasets = sorted([d for d in test_root.iterdir() if d.is_dir()])

    print(f"\nFound {len(train_datasets)} training datasets")
    print(f"Found {len(test_datasets)} test datasets")

    all_predictions = []
    all_sequences = []

    # Process each training dataset
    for train_dir in train_datasets:
        dataset_name = train_dir.name
        print(f"\n{'='*60}")
        print(f"Processing: {dataset_name}")
        print(f"{'='*60}")

        # Initialize and train
        predictor = GPUXGBoostPredictor(n_jobs=8)
        predictor.fit(str(train_dir))

        # Find corresponding test datasets
        base_num = dataset_name.split('_')[-1]
        matching_tests = [
            t for t in test_datasets
            if f"dataset_{base_num}" in t.name or t.name == f"test_dataset_{base_num}"
        ]

        for test_dir in matching_tests:
            preds = predictor.predict_proba(str(test_dir))
            all_predictions.append(preds)

        # Identify sequences
        seqs = predictor.identify_sequences(top_k=50000)
        all_sequences.append(seqs)

        # Clean up
        del predictor
        gc.collect()

    # Combine results
    if all_predictions:
        final_predictions = pd.concat(all_predictions, ignore_index=True)
        print(f"\nTotal predictions: {len(final_predictions)}")
    else:
        final_predictions = pd.DataFrame()

    if all_sequences:
        final_sequences = pd.concat(all_sequences, ignore_index=True)
        print(f"Total sequences: {len(final_sequences)}")
    else:
        final_sequences = pd.DataFrame()

    # Create submission
    if len(final_predictions) > 0 or len(final_sequences) > 0:
        print("\nCreating submission file...")
        submission = pd.concat([final_predictions, final_sequences], ignore_index=True)
        submission_path = out_dir / "submission.csv"
        submission.to_csv(submission_path, index=False)
        print(f"Submission saved to: {submission_path}")
        print(f"Submission shape: {submission.shape}")

    print("\n" + "=" * 60)
    print("GPU XGBoost Championship Pipeline Complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
