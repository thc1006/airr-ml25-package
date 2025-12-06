#!/usr/bin/env python3
"""
AIRR-ML-25 Fast Championship Pipeline
======================================
Optimized for speed using:
- DictVectorizer for efficient sparse matrix creation
- Only k=3 k-mers (fastest)
- XGBoost only (fastest GPU model)
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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')

# =============================================================================
# Feature Extraction
# =============================================================================

def extract_kmer_counts(sequences: List[str], k: int = 3) -> Dict[str, int]:
    """Extract k-mer counts from sequences."""
    counts = Counter()
    for seq in sequences:
        if isinstance(seq, str) and len(seq) >= k:
            for i in range(len(seq) - k + 1):
                counts[seq[i:i+k]] += 1
    # Normalize
    total = sum(counts.values()) or 1
    return {f"k{k}_{kmer}": c / total for kmer, c in counts.items()}


def load_repertoire_fast(file_path: str, repertoire_id: str, k: int = 3) -> Tuple[str, Dict]:
    """Load a single repertoire and extract features quickly."""
    try:
        df = pd.read_csv(file_path, sep='\t', usecols=['junction_aa'])
        sequences = df['junction_aa'].dropna().tolist()
        features = extract_kmer_counts(sequences, k=k)
        return repertoire_id, features
    except Exception as e:
        return repertoire_id, {}


def load_all_repertoires(data_dir: str, k: int = 3, n_jobs: int = 8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load all repertoires and create sparse feature matrix."""
    metadata_path = os.path.join(data_dir, 'metadata.csv')

    if os.path.exists(metadata_path):
        metadata_df = pd.read_csv(metadata_path)
        job_args = [
            (os.path.join(data_dir, row.filename), row.repertoire_id, k)
            for row in metadata_df.itertuples(index=False)
        ]
        labels = {row.repertoire_id: row.label_positive for row in metadata_df.itertuples(index=False)}
    else:
        # Test set without metadata
        tsv_files = sorted(glob.glob(os.path.join(data_dir, '*.tsv')))
        job_args = [
            (f, os.path.basename(f).replace('.tsv', ''), k)
            for f in tsv_files
        ]
        labels = {}

    print(f"  Loading {len(job_args)} repertoires (k={k})...")

    # Parallel loading
    results = Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(load_repertoire_fast)(*args) for args in tqdm(job_args, disable=False)
    )

    # Build sparse matrix using DictVectorizer
    rep_ids = [r[0] for r in results]
    features = [r[1] for r in results]

    print(f"  Creating sparse feature matrix...")
    vectorizer = DictVectorizer(sparse=True)
    X = vectorizer.fit_transform(features)

    print(f"  Shape: {X.shape} (sparse)")

    # Create DataFrames
    feature_names = vectorizer.get_feature_names_out()
    X_df = pd.DataFrame.sparse.from_spmatrix(X, index=rep_ids, columns=feature_names)

    if labels:
        y_df = pd.DataFrame([{'ID': rid, 'label_positive': labels.get(rid)} for rid in rep_ids])
    else:
        y_df = pd.DataFrame([{'ID': rid, 'label_positive': None} for rid in rep_ids])

    return X_df, y_df, vectorizer


# =============================================================================
# Simple Fast Predictor
# =============================================================================

class FastPredictor:
    """Fast predictor using k-mer features and logistic regression."""

    def __init__(self, k: int = 3, n_jobs: int = 8):
        self.k = k
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
        X_df, y_df, self.vectorizer = load_all_repertoires(train_dir, k=self.k, n_jobs=self.n_jobs)

        # Align
        y_indexed = y_df.set_index('ID')['label_positive']
        common_ids = X_df.index.intersection(y_indexed.index)
        X = X_df.loc[common_ids].values
        y = y_indexed.loc[common_ids].values.astype(int)

        print(f"  Training samples: {len(y)}")
        print(f"  Features: {X.shape[1]}")

        # Split for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Train model
        print(f"  Training logistic regression...")
        self.model = LogisticRegression(
            C=0.1, penalty='l1', solver='saga', max_iter=1000, n_jobs=-1, random_state=42
        )
        self.model.fit(X_train, y_train)

        # Validate
        y_pred = self.model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_pred)
        print(f"  Validation AUC: {auc:.4f}")

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
            _, features = load_repertoire_fast(file_path, rep_id, k=self.k)

            # Transform to same feature space
            X = self.vectorizer.transform([features])

            # Predict
            prob = self.model.predict_proba(X)[0, 1]

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

        # Get feature importances
        coefs = self.model.coef_[0]
        feature_names = self.vectorizer.get_feature_names_out()

        # Get top positive k-mers
        kmer_importance = list(zip(feature_names, coefs))
        kmer_importance.sort(key=lambda x: abs(x[1]), reverse=True)

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

        # Score sequences
        important_kmers = {name: abs(coef) for name, coef in kmer_importance[:1000]}

        scores = []
        for seq in unique_seqs['junction_aa']:
            score = 0.0
            if isinstance(seq, str) and len(seq) >= self.k:
                for i in range(len(seq) - self.k + 1):
                    kmer = f"k{self.k}_{seq[i:i+self.k]}"
                    score += important_kmers.get(kmer, 0)
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
    print("AIRR-ML-25 Fast Championship Pipeline")
    print("=" * 60)

    # Paths
    train_root = Path("./data/train_datasets/train_datasets")
    test_root = Path("./data/test_datasets/test_datasets")
    out_dir = Path("./results_championship")
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
        predictor = FastPredictor(k=3, n_jobs=8)
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
    print("Fast Championship Pipeline Complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
