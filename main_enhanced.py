#!/usr/bin/env python3
"""
Enhanced AIRR-ML-25 Predictor with Priority 1 Features:
- Multi-scale k-mers (k=3,4,5)
- V/J gene usage patterns
- Clonality metrics
- CDR3 length statistics

Based on main.py with enhancements from FINAL_ACTION_PLAN.md
"""

import os
import sys
import argparse
import glob
import pandas as pd
import numpy as np
from collections import Counter
from typing import List, Tuple
from tqdm import tqdm
from scipy.stats import entropy, skew, kurtosis

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.pipeline import Pipeline


# ============================================================================
# Feature Extraction Functions
# ============================================================================

def extract_multiscale_kmers(seq: str, k_values: List[int] = [3, 4, 5]) -> Counter:
    """Extract k-mers at multiple scales."""
    kmer_counts = Counter()
    for k in k_values:
        for i in range(len(seq) - k + 1):
            kmer = seq[i:i + k]
            kmer_counts[f"k{k}_{kmer}"] += 1
    return kmer_counts


def extract_vj_features(df: pd.DataFrame) -> dict:
    """
    Extract V/J gene usage patterns.

    Returns dict with:
    - v_usage_top20: frequency of top 20 V genes
    - j_usage_top20: frequency of top 20 J genes
    - vj_pair_top50: frequency of top 50 VJ combinations
    """
    features = {}
    total = len(df)

    if total == 0:
        return features

    # V gene usage
    if 'v_call' in df.columns:
        v_counts = df['v_call'].value_counts().head(20)
        for gene, count in v_counts.items():
            if pd.notna(gene) and gene != 'unknown':
                features[f"v_usage_{gene}"] = count / total

    # J gene usage
    if 'j_call' in df.columns:
        j_counts = df['j_call'].value_counts().head(20)
        for gene, count in j_counts.items():
            if pd.notna(gene) and gene != 'unknown':
                features[f"j_usage_{gene}"] = count / total

    # VJ pair usage
    if 'v_call' in df.columns and 'j_call' in df.columns:
        vj_pairs = df.groupby(['v_call', 'j_call']).size().nlargest(50)
        for (v, j), count in vj_pairs.items():
            if pd.notna(v) and pd.notna(j) and v != 'unknown' and j != 'unknown':
                features[f"vj_pair_{v}_{j}"] = count / total

    return features


def extract_clonality_features(df: pd.DataFrame) -> dict:
    """
    Extract clonality and diversity metrics.

    Returns:
    - shannon_entropy
    - gini_simpson
    - d50 (diversity index)
    - clonality score
    """
    features = {}

    if 'junction_aa' not in df.columns or len(df) == 0:
        return features

    # Count sequence frequencies
    seq_counts = df['junction_aa'].value_counts()
    frequencies = seq_counts.values / seq_counts.sum()

    # Shannon entropy
    features['shannon_entropy'] = entropy(frequencies)

    # Gini-Simpson index
    features['gini_simpson'] = 1 - np.sum(frequencies ** 2)

    # D50 - number of clones accounting for 50% of repertoire
    cumsum = np.cumsum(np.sort(frequencies)[::-1])
    features['d50'] = np.sum(cumsum <= 0.5)

    # Clonality score (1 - normalized entropy)
    max_entropy = np.log(len(seq_counts))
    if max_entropy > 0:
        features['clonality'] = 1 - (features['shannon_entropy'] / max_entropy)
    else:
        features['clonality'] = 0

    return features


def extract_cdr3_length_features(df: pd.DataFrame) -> dict:
    """
    Extract CDR3 length distribution statistics.

    Returns:
    - length_mean, length_std, length_median
    - length_q25, length_q75 (quartiles)
    - length_skew, length_kurt (shape metrics)
    """
    features = {}

    if 'junction_aa' not in df.columns or len(df) == 0:
        return features

    lengths = df['junction_aa'].dropna().apply(len)

    if len(lengths) > 0:
        features['length_mean'] = lengths.mean()
        features['length_std'] = lengths.std()
        features['length_median'] = lengths.median()
        features['length_q25'] = lengths.quantile(0.25)
        features['length_q75'] = lengths.quantile(0.75)

        if len(lengths) > 1:
            features['length_skew'] = skew(lengths)
            features['length_kurt'] = kurtosis(lengths)
        else:
            features['length_skew'] = 0
            features['length_kurt'] = 0

    return features


def load_and_encode_enhanced_features(
    data_dir: str,
    k_values: List[int] = [3, 4, 5]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Enhanced feature extraction with multi-scale k-mers and biological features.

    Returns:
        Tuple of (encoded_features_df, metadata_df)
    """
    from main import load_data_generator  # Import from original main.py

    metadata_path = os.path.join(data_dir, 'metadata.csv')
    data_loader = load_data_generator(data_dir=data_dir)

    repertoire_features = []
    metadata_records = []

    search_pattern = os.path.join(data_dir, '*.tsv')
    total_files = len(glob.glob(search_pattern))

    for item in tqdm(data_loader, total=total_files, desc=f"Enhanced features (k={k_values})"):
        if os.path.exists(metadata_path):
            rep_id, data_df, label = item
        else:
            filename, data_df = item
            rep_id = os.path.basename(filename).replace(".tsv", "")
            label = None

        # Initialize feature dict
        features = {'ID': rep_id}

        # 1. Multi-scale k-mers
        kmer_counts = Counter()
        for seq in data_df['junction_aa'].dropna():
            kmer_counts.update(extract_multiscale_kmers(seq, k_values))
        features.update(kmer_counts)

        # 2. V/J gene usage
        vj_features = extract_vj_features(data_df)
        features.update(vj_features)

        # 3. Clonality metrics
        clonality_features = extract_clonality_features(data_df)
        features.update(clonality_features)

        # 4. CDR3 length features
        length_features = extract_cdr3_length_features(data_df)
        features.update(length_features)

        repertoire_features.append(features)

        # Metadata
        metadata_record = {'ID': rep_id}
        if label is not None:
            metadata_record['label_positive'] = label
        metadata_records.append(metadata_record)

        del data_df

    features_df = pd.DataFrame(repertoire_features).fillna(0).set_index('ID')
    metadata_df = pd.DataFrame(metadata_records)

    print(f"\n=== Enhanced Feature Summary ===")
    print(f"Total features: {len(features_df.columns)}")
    print(f"  K-mer features: {sum('k3_' in c or 'k4_' in c or 'k5_' in c for c in features_df.columns)}")
    print(f"  V/J features: {sum('v_usage' in c or 'j_usage' in c or 'vj_pair' in c for c in features_df.columns)}")
    print(f"  Clonality features: {sum(c in ['shannon_entropy', 'gini_simpson', 'd50', 'clonality'] for c in features_df.columns)}")
    print(f"  CDR3 length features: {sum('length_' in c for c in features_df.columns)}")

    return features_df, metadata_df


# ============================================================================
# Main Execution
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Enhanced AIRR-ML-25 Predictor')
    parser.add_argument('--train_root', required=True, help='Root directory containing train_dataset_*')
    parser.add_argument('--test_root', required=True, help='Root directory containing test_dataset_*')
    parser.add_argument('--out_dir', required=True, help='Output directory for results')
    parser.add_argument('--n_jobs', type=int, default=-1, help='Number of parallel jobs')
    parser.add_argument('--k_values', type=str, default='3,4,5', help='Comma-separated k values')
    args = parser.parse_args()

    k_values = [int(k) for k in args.k_values.split(',')]
    print(f"\n{'='*60}")
    print(f"ENHANCED AIRR-ML-25 PREDICTOR")
    print(f"K-mer scales: {k_values}")
    print(f"Features: Multi-scale k-mers + V/J + Clonality + CDR3 length")
    print(f"{'='*60}\n")

    # Import original functions we need
    from main import (
        get_dataset_pairs, ImmuneStatePredictor,
        _train_predictor, _generate_predictions,
        _save_predictions, _save_important_sequences,
        concatenate_output_files
    )

    # Get all dataset pairs
    dataset_pairs = get_dataset_pairs(args.train_root, args.test_root)

    if not dataset_pairs:
        print("No dataset pairs found!")
        sys.exit(1)

    print(f"Found {len(dataset_pairs)} dataset pairs")

    # Process each dataset
    for train_dir, test_dirs in dataset_pairs:
        print(f"\n{'='*60}")
        print(f"Processing: {os.path.basename(train_dir)}")
        print(f"{'='*60}")

        # TODO: Create enhanced predictor class
        # For now, use original with note that we need to integrate enhanced features
        print("⚠️  Enhanced feature extraction prepared - integration pending")
        print("   Next step: Integrate into ImmuneStatePredictor class")

    print("\n✅ Enhanced predictor framework ready!")
    print("   Next: Integrate enhanced features into training pipeline")


if __name__ == '__main__':
    main()
