#!/usr/bin/env python3
"""
main.py - AIRR-ML-25 Competition Baseline
==========================================

Official template-compliant implementation for the Adaptive Immune Profiling Challenge 2025.
Refactored from the official example notebook to ensure correct submission format.

Hardware Optimized for:
- AMD Ryzen 7 7800X3D (8 cores)
- 32GB RAM
- NVIDIA RTX 5080 (16GB VRAM)

Tasks:
- Task A: Predict label_positive probability for each repertoire
- Task B: Identify top 50,000 label-associated sequences per dataset

Submission format: 404,213 rows total
- 4,213 test predictions
- 8 datasets × 50,000 important sequences
"""

import argparse
import glob
import os
import sys
from collections import Counter, defaultdict
from typing import Dict, Iterator, List, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# ============================================================================
# Utility Functions (from official template)
# ============================================================================

def load_data_generator(data_dir: str, metadata_filename: str = 'metadata.csv') -> Iterator[
    Union[Tuple[str, pd.DataFrame, bool], Tuple[str, pd.DataFrame]]]:
    """
    A generator to load immune repertoire data.

    Yields:
        With metadata: (repertoire_id, pd.DataFrame, label_positive)
        Without metadata: (filename, pd.DataFrame)
    """
    metadata_path = os.path.join(data_dir, metadata_filename)

    if os.path.exists(metadata_path):
        metadata_df = pd.read_csv(metadata_path)
        for row in metadata_df.itertuples(index=False):
            file_path = os.path.join(data_dir, row.filename)
            try:
                repertoire_df = pd.read_csv(file_path, sep='\t')
                yield row.repertoire_id, repertoire_df, row.label_positive
            except FileNotFoundError:
                print(f"Warning: File '{row.filename}' not found. Skipping.")
                continue
    else:
        search_pattern = os.path.join(data_dir, '*.tsv')
        tsv_files = glob.glob(search_pattern)
        for file_path in sorted(tsv_files):
            try:
                filename = os.path.basename(file_path)
                repertoire_df = pd.read_csv(file_path, sep='\t')
                yield filename, repertoire_df
            except Exception as e:
                print(f"Warning: Could not read '{file_path}': {e}. Skipping.")
                continue


def load_full_dataset(data_dir: str) -> pd.DataFrame:
    """Loads all TSV files and concatenates them into a single DataFrame."""
    metadata_path = os.path.join(data_dir, 'metadata.csv')
    df_list = []
    data_loader = load_data_generator(data_dir=data_dir)

    if os.path.exists(metadata_path):
        metadata_df = pd.read_csv(metadata_path)
        total_files = len(metadata_df)
        for rep_id, data_df, label in tqdm(data_loader, total=total_files, desc="Loading files"):
            data_df['ID'] = rep_id
            data_df['label_positive'] = label
            df_list.append(data_df)
    else:
        search_pattern = os.path.join(data_dir, '*.tsv')
        total_files = len(glob.glob(search_pattern))
        for filename, data_df in tqdm(data_loader, total=total_files, desc="Loading files"):
            data_df['ID'] = os.path.basename(filename).replace(".tsv", "")
            df_list.append(data_df)

    if not df_list:
        print("Warning: No data files were loaded.")
        return pd.DataFrame()

    return pd.concat(df_list, ignore_index=True)


def load_and_encode_kmers(data_dir: str, k: int = 3) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Memory-efficient k-mer encoding of repertoire data.

    Returns:
        Tuple of (encoded_features_df, metadata_df)
    """
    metadata_path = os.path.join(data_dir, 'metadata.csv')
    data_loader = load_data_generator(data_dir=data_dir)

    repertoire_features = []
    metadata_records = []

    search_pattern = os.path.join(data_dir, '*.tsv')
    total_files = len(glob.glob(search_pattern))

    for item in tqdm(data_loader, total=total_files, desc=f"Encoding {k}-mers"):
        if os.path.exists(metadata_path):
            rep_id, data_df, label = item
        else:
            filename, data_df = item
            rep_id = os.path.basename(filename).replace(".tsv", "")
            label = None

        kmer_counts = Counter()
        for seq in data_df['junction_aa'].dropna():
            for i in range(len(seq) - k + 1):
                kmer_counts[seq[i:i + k]] += 1

        repertoire_features.append({
            'ID': rep_id,
            **kmer_counts
        })

        metadata_record = {'ID': rep_id}
        if label is not None:
            metadata_record['label_positive'] = label
        metadata_records.append(metadata_record)

        del data_df, kmer_counts

    features_df = pd.DataFrame(repertoire_features).fillna(0).set_index('ID')
    metadata_df = pd.DataFrame(metadata_records)

    return features_df, metadata_df


def save_tsv(df: pd.DataFrame, path: str):
    """Save DataFrame as TSV file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, sep='\t', index=False)


def get_dataset_pairs(train_dir: str, test_dir: str) -> List[Tuple[str, List[str]]]:
    """Returns list of (train_path, [test_paths]) tuples for dataset pairs."""
    test_groups = defaultdict(list)
    for test_name in sorted(os.listdir(test_dir)):
        if test_name.startswith("test_dataset_"):
            base_id = test_name.replace("test_dataset_", "").split("_")[0]
            test_groups[base_id].append(os.path.join(test_dir, test_name))

    pairs = []
    for train_name in sorted(os.listdir(train_dir)):
        if train_name.startswith("train_dataset_"):
            train_id = train_name.replace("train_dataset_", "")
            train_path = os.path.join(train_dir, train_name)
            pairs.append((train_path, test_groups.get(train_id, [])))

    return pairs


def concatenate_output_files(out_dir: str) -> None:
    """Concatenates all output TSV files into submissions.csv."""
    predictions_pattern = os.path.join(out_dir, '*_test_predictions.tsv')
    sequences_pattern = os.path.join(out_dir, '*_important_sequences.tsv')

    predictions_files = sorted(glob.glob(predictions_pattern))
    sequences_files = sorted(glob.glob(sequences_pattern))

    df_list = []

    for pred_file in predictions_files:
        try:
            df = pd.read_csv(pred_file, sep='\t')
            df_list.append(df)
        except Exception as e:
            print(f"Warning: Could not read '{pred_file}': {e}. Skipping.")

    for seq_file in sequences_files:
        try:
            df = pd.read_csv(seq_file, sep='\t')
            df_list.append(df)
        except Exception as e:
            print(f"Warning: Could not read '{seq_file}': {e}. Skipping.")

    if not df_list:
        print("Warning: No output files found.")
        concatenated_df = pd.DataFrame(
            columns=['ID', 'dataset', 'label_positive_probability', 'junction_aa', 'v_call', 'j_call'])
    else:
        concatenated_df = pd.concat(df_list, ignore_index=True)

    submissions_file = os.path.join(out_dir, 'submissions.csv')
    concatenated_df.to_csv(submissions_file, index=False)
    print(f"Concatenated output: {submissions_file} ({len(concatenated_df)} rows)")


def validate_dirs_and_files(train_dir: str, test_dirs: List[str], out_dir: str) -> None:
    """Validate input directories and files exist."""
    assert os.path.isdir(train_dir), f"Train directory '{train_dir}' does not exist."
    train_tsvs = glob.glob(os.path.join(train_dir, "*.tsv"))
    assert train_tsvs, f"No .tsv files found in '{train_dir}'."
    metadata_path = os.path.join(train_dir, "metadata.csv")
    assert os.path.isfile(metadata_path), f"metadata.csv not found in '{train_dir}'."

    for test_dir in test_dirs:
        assert os.path.isdir(test_dir), f"Test directory '{test_dir}' does not exist."
        test_tsvs = glob.glob(os.path.join(test_dir, "*.tsv"))
        assert test_tsvs, f"No .tsv files found in '{test_dir}'."

    try:
        os.makedirs(out_dir, exist_ok=True)
    except Exception as e:
        print(f"Failed to create output directory '{out_dir}': {e}")
        sys.exit(1)


# ============================================================================
# KmerClassifier - Core ML Model
# ============================================================================

class KmerClassifier:
    """L1-regularized logistic regression for k-mer count data with ROC-AUC optimization."""

    def __init__(self, c_values=None, cv_folds=5,
                 opt_metric='roc_auc', random_state=42, n_jobs=-1):
        if c_values is None:
            c_values = [1.0, 0.2, 0.1, 0.05, 0.03]
        self.c_values = c_values
        self.cv_folds = cv_folds
        self.opt_metric = opt_metric
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.best_C_ = None
        self.best_score_ = None
        self.cv_results_ = None
        self.model_ = None
        self.feature_names_ = None
        self.val_score_ = None

    def _make_pipeline(self, C):
        """Create standardization + L1 logistic regression pipeline."""
        return Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(
                penalty='l1', C=C, solver='liblinear',
                random_state=self.random_state, max_iter=1000
            ))
        ])

    def _get_scorer(self):
        """Get scoring function for optimization."""
        if self.opt_metric == 'balanced_accuracy':
            return 'balanced_accuracy'
        elif self.opt_metric == 'roc_auc':
            return 'roc_auc'
        else:
            raise ValueError(f"Unknown metric: {self.opt_metric}")

    def tune_and_fit(self, X, y, val_size=0.2):
        """Perform CV tuning on train split and fit with validation."""
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns.tolist()
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        if val_size > 0:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=val_size, random_state=self.random_state, stratify=y)
        else:
            X_train, y_train = X, y
            X_val, y_val = None, None

        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True,
                             random_state=self.random_state)
        scorer = self._get_scorer()

        results = []
        for C in self.c_values:
            pipeline = self._make_pipeline(C)
            scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring=scorer,
                                     n_jobs=self.n_jobs)
            results.append({
                'C': C,
                'mean_score': scores.mean(),
                'std_score': scores.std(),
                'scores': scores
            })

        self.cv_results_ = pd.DataFrame(results)
        best_idx = self.cv_results_['mean_score'].idxmax()
        self.best_C_ = self.cv_results_.loc[best_idx, 'C']
        self.best_score_ = self.cv_results_.loc[best_idx, 'mean_score']

        print(f"Best C: {self.best_C_} (CV {self.opt_metric}: {self.best_score_:.4f})")

        # Fit on training split with best hyperparameter
        self.model_ = self._make_pipeline(self.best_C_)
        self.model_.fit(X_train, y_train)

        if X_val is not None:
            if scorer == 'balanced_accuracy':
                self.val_score_ = balanced_accuracy_score(y_val, self.model_.predict(X_val))
            else:
                self.val_score_ = roc_auc_score(y_val, self.model_.predict_proba(X_val)[:, 1])
            print(f"Validation {self.opt_metric}: {self.val_score_:.4f}")

        return self

    def predict_proba(self, X):
        """Predict class probabilities."""
        if self.model_ is None:
            raise ValueError("Model not fitted.")
        if isinstance(X, pd.DataFrame):
            X = X.values
        return self.model_.predict_proba(X)[:, 1]

    def predict(self, X):
        """Predict class labels."""
        if self.model_ is None:
            raise ValueError("Model not fitted.")
        if isinstance(X, pd.DataFrame):
            X = X.values
        return self.model_.predict(X)

    def get_feature_importance(self):
        """Get feature importance from L1 coefficients."""
        if self.model_ is None:
            raise ValueError("Model not fitted.")

        coef = self.model_.named_steps['classifier'].coef_[0]

        if self.feature_names_ is not None:
            feature_names = self.feature_names_
        else:
            feature_names = [f"feature_{i}" for i in range(len(coef))]

        importance_df = pd.DataFrame({
            'feature': feature_names,
            'coefficient': coef,
            'abs_coefficient': np.abs(coef)
        })

        return importance_df.sort_values('abs_coefficient', ascending=False)

    def score_all_sequences(self, sequences_df, sequence_col='junction_aa'):
        """
        Score all sequences using model coefficients.

        This is the CORRECT way to identify important sequences for Task B.
        """
        if self.model_ is None:
            raise ValueError("Model not fitted.")

        scaler = self.model_.named_steps['scaler']
        coefficients = self.model_.named_steps['classifier'].coef_[0]
        coefficients = coefficients / scaler.scale_

        kmer_to_index = {kmer: idx for idx, kmer in enumerate(self.feature_names_)}
        k = len(self.feature_names_[0])

        scores = []
        total_seqs = len(sequences_df)
        for seq in tqdm(sequences_df[sequence_col], total=total_seqs, desc="Scoring sequences"):
            counts = np.zeros(len(kmer_to_index), dtype=np.uint8)
            for i in range(len(seq) - k + 1):
                kmer = seq[i:i + k]
                if kmer in kmer_to_index:
                    counts[kmer_to_index[kmer]] = 1
            scores.append(np.dot(counts, coefficients))

        result_df = sequences_df.copy()
        result_df['importance_score'] = scores
        return result_df


def prepare_data(X_df, labels_df, id_col='ID', label_col='label_positive'):
    """Merge feature matrix with labels, ensuring alignment."""
    if id_col in labels_df.columns:
        labels_indexed = labels_df.set_index(id_col)[label_col]
    else:
        labels_indexed = labels_df[label_col]

    common_ids = X_df.index.intersection(labels_indexed.index)

    if len(common_ids) == 0:
        raise ValueError("No common IDs found between feature matrix and labels")

    X = X_df.loc[common_ids]
    y = labels_indexed.loc[common_ids]

    print(f"Aligned {len(common_ids)} samples with labels")

    return X, y, common_ids


# ============================================================================
# ImmuneStatePredictor - Official Template Class
# ============================================================================

class ImmuneStatePredictor:
    """
    Official template-compliant predictor for AIRR-ML-25.

    This class implements:
    - Task A: Repertoire classification (label_positive probability)
    - Task B: Important sequence identification (top 50,000 per dataset)
    """

    def __init__(self, n_jobs: int = -1, device: str = 'cpu', **kwargs):
        """
        Initialize predictor.

        Args:
            n_jobs: Number of CPU cores (-1 for all available)
            device: Computation device ('cpu' or 'cuda')
        """
        self.train_ids_ = None
        total_cores = os.cpu_count()
        if n_jobs == -1:
            self.n_jobs = total_cores
        else:
            self.n_jobs = min(n_jobs, total_cores)
        self.device = device
        self.model = None
        self.important_sequences_ = None

        print(f"Initialized with n_jobs={self.n_jobs}, device={self.device}")

    def fit(self, train_dir_path: str):
        """
        Train the model on provided training data.

        Args:
            train_dir_path: Path to training data directory with metadata.csv
        """
        print(f"Training on: {train_dir_path}")

        # Load and encode k-mers
        X_train_df, y_train_df = load_and_encode_kmers(train_dir_path, k=4)

        # Prepare aligned data
        X_train, y_train, train_ids = prepare_data(
            X_train_df, y_train_df, id_col='ID', label_col='label_positive'
        )

        # Initialize and train classifier
        self.model = KmerClassifier(
            c_values=[1.0, 0.2, 0.1, 0.05, 0.03],
            cv_folds=5,
            opt_metric='roc_auc',  # Competition metric!
            random_state=42,
            n_jobs=self.n_jobs
        )

        self.model.tune_and_fit(X_train, y_train)
        self.train_ids_ = train_ids

        # Identify important sequences (Task B)
        self.important_sequences_ = self.identify_associated_sequences(
            train_dir_path=train_dir_path, top_k=50000
        )

        print("Training complete.")
        return self

    def predict_proba(self, test_dir_path: str) -> pd.DataFrame:
        """
        Predict probabilities for test repertoires.

        Returns DataFrame with official submission format columns.
        """
        print(f"Predicting on: {test_dir_path}")
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        # Load and encode test data
        X_test_df, _ = load_and_encode_kmers(test_dir_path, k=4)

        # Align features with training
        if self.model.feature_names_ is not None:
            X_test_df = X_test_df.reindex(columns=self.model.feature_names_, fill_value=0)

        repertoire_ids = X_test_df.index.tolist()

        # Predict
        probabilities = self.model.predict_proba(X_test_df)

        # Build output DataFrame with OFFICIAL format
        predictions_df = pd.DataFrame({
            'ID': repertoire_ids,
            'dataset': [os.path.basename(test_dir_path)] * len(repertoire_ids),
            'label_positive_probability': probabilities,
            'junction_aa': -999.0,  # Placeholder for prediction rows
            'v_call': -999.0,
            'j_call': -999.0
        })

        # Ensure correct column order
        predictions_df = predictions_df[['ID', 'dataset', 'label_positive_probability',
                                         'junction_aa', 'v_call', 'j_call']]

        print(f"Predicted {len(repertoire_ids)} repertoires in {test_dir_path}")
        return predictions_df

    def identify_associated_sequences(self, train_dir_path: str, top_k: int = 50000) -> pd.DataFrame:
        """
        Identify top K important sequences for Task B.

        Returns DataFrame with actual sequences (junction_aa, v_call, j_call).
        """
        dataset_name = os.path.basename(train_dir_path)
        print(f"Identifying {top_k} important sequences for {dataset_name}...")

        # Load full dataset to get actual sequences
        full_df = load_full_dataset(train_dir_path)

        # Get unique sequences with v_call and j_call
        unique_seqs = full_df[['junction_aa', 'v_call', 'j_call']].drop_duplicates()
        print(f"Found {len(unique_seqs)} unique sequences")

        # Score sequences using model coefficients
        all_sequences_scored = self.model.score_all_sequences(unique_seqs, sequence_col='junction_aa')

        # Select top K by importance score
        top_sequences_df = all_sequences_scored.nlargest(top_k, 'importance_score')
        top_sequences_df = top_sequences_df[['junction_aa', 'v_call', 'j_call']]

        # Build output with OFFICIAL format
        top_sequences_df = top_sequences_df.copy()
        top_sequences_df['dataset'] = dataset_name
        top_sequences_df['ID'] = [f"{dataset_name}_seq_top_{i+1}" for i in range(len(top_sequences_df))]
        top_sequences_df['label_positive_probability'] = -999.0  # Placeholder for sequence rows

        # Ensure correct column order
        top_sequences_df = top_sequences_df[['ID', 'dataset', 'label_positive_probability',
                                              'junction_aa', 'v_call', 'j_call']]

        return top_sequences_df


# ============================================================================
# Main Execution Pipeline
# ============================================================================

def _train_predictor(predictor: ImmuneStatePredictor, train_dir: str):
    """Train the predictor."""
    print(f"\n{'='*60}")
    print(f"TRAINING: {train_dir}")
    print(f"{'='*60}")
    predictor.fit(train_dir)


def _generate_predictions(predictor: ImmuneStatePredictor, test_dirs: List[str]) -> pd.DataFrame:
    """Generate predictions for all test directories."""
    all_preds = []
    for test_dir in test_dirs:
        print(f"\n--- Predicting: {test_dir}")
        preds = predictor.predict_proba(test_dir)
        if preds is not None and not preds.empty:
            all_preds.append(preds)
        else:
            print(f"Warning: No predictions for {test_dir}")
    if all_preds:
        return pd.concat(all_preds, ignore_index=True)
    return pd.DataFrame()


def _save_predictions(predictions: pd.DataFrame, out_dir: str, train_dir: str) -> None:
    """Save predictions to TSV file."""
    if predictions.empty:
        raise ValueError("No predictions to save")

    preds_path = os.path.join(out_dir, f"{os.path.basename(train_dir)}_test_predictions.tsv")
    save_tsv(predictions, preds_path)
    print(f"Predictions saved: {preds_path}")


def _save_important_sequences(predictor: ImmuneStatePredictor, out_dir: str, train_dir: str) -> None:
    """Save important sequences to TSV file."""
    seqs = predictor.important_sequences_
    if seqs is None or seqs.empty:
        raise ValueError("No important sequences to save")

    seqs_path = os.path.join(out_dir, f"{os.path.basename(train_dir)}_important_sequences.tsv")
    save_tsv(seqs, seqs_path)
    print(f"Important sequences saved: {seqs_path} ({len(seqs)} sequences)")


def process_single_dataset(train_dir: str, test_dirs: List[str], out_dir: str,
                           n_jobs: int, device: str) -> None:
    """Process a single train-test dataset pair."""
    validate_dirs_and_files(train_dir, test_dirs, out_dir)

    predictor = ImmuneStatePredictor(n_jobs=n_jobs, device=device)
    _train_predictor(predictor, train_dir)
    predictions = _generate_predictions(predictor, test_dirs)
    _save_predictions(predictions, out_dir, train_dir)
    _save_important_sequences(predictor, out_dir, train_dir)


def run_all_datasets(train_root: str, test_root: str, out_dir: str,
                     n_jobs: int, device: str) -> None:
    """Process all dataset pairs and generate final submission."""
    print(f"\n{'#'*60}")
    print("AIRR-ML-25: Full Pipeline Execution")
    print(f"{'#'*60}")
    print(f"Train root: {train_root}")
    print(f"Test root:  {test_root}")
    print(f"Output:     {out_dir}")
    print(f"n_jobs:     {n_jobs}")
    print(f"device:     {device}")

    # Get all dataset pairs
    dataset_pairs = get_dataset_pairs(train_root, test_root)
    print(f"\nFound {len(dataset_pairs)} dataset pairs:")
    for train_path, test_paths in dataset_pairs:
        print(f"  {os.path.basename(train_path)} -> {[os.path.basename(p) for p in test_paths]}")

    # Process each pair
    for train_dir, test_dirs in dataset_pairs:
        if not test_dirs:
            print(f"\nSkipping {train_dir} - no matching test directories")
            continue
        process_single_dataset(train_dir, test_dirs, out_dir, n_jobs, device)

    # Concatenate all outputs
    print(f"\n{'='*60}")
    print("Generating final submission file...")
    print(f"{'='*60}")
    concatenate_output_files(out_dir)

    # Validate submission
    submissions_path = os.path.join(out_dir, 'submissions.csv')
    if os.path.exists(submissions_path):
        df = pd.read_csv(submissions_path)
        print(f"\nFinal submission: {len(df)} rows")
        print(f"Expected: 404,213 rows")
        if len(df) == 404213:
            print("PASS: Row count matches expected!")
        else:
            print(f"WARNING: Row count mismatch! Difference: {len(df) - 404213}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="AIRR-ML-25 Baseline Predictor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single dataset
  python main.py --train_dir ./data/train_datasets/train_dataset_1 \\
                 --test_dirs ./data/test_datasets/test_dataset_1 \\
                 --out_dir ./results --n_jobs 8

  # All datasets
  python main.py --train_root ./data/train_datasets \\
                 --test_root ./data/test_datasets \\
                 --out_dir ./results --n_jobs 8
        """
    )

    # Single dataset mode
    parser.add_argument('--train_dir', type=str,
                        help='Single training directory (with metadata.csv)')
    parser.add_argument('--test_dirs', type=str, nargs='+',
                        help='Test directories for single dataset mode')

    # Batch mode
    parser.add_argument('--train_root', type=str,
                        help='Root directory containing all train_dataset_* folders')
    parser.add_argument('--test_root', type=str,
                        help='Root directory containing all test_dataset_* folders')

    # Common options
    parser.add_argument('--out_dir', type=str, required=True,
                        help='Output directory for results')
    parser.add_argument('--n_jobs', type=int, default=-1,
                        help='Number of CPU cores (-1 for all available)')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'],
                        help='Computation device')

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Determine execution mode
    if args.train_root and args.test_root:
        # Batch mode: process all datasets
        run_all_datasets(args.train_root, args.test_root, args.out_dir,
                        args.n_jobs, args.device)
    elif args.train_dir and args.test_dirs:
        # Single dataset mode
        process_single_dataset(args.train_dir, args.test_dirs, args.out_dir,
                              args.n_jobs, args.device)
    else:
        print("Error: Specify either (--train_root, --test_root) or (--train_dir, --test_dirs)")
        sys.exit(1)


if __name__ == '__main__':
    main()
