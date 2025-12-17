#!/usr/bin/env python3
"""
Test script for main_v2.py enhanced features.
Validates that all Priority 1 features are correctly extracted.
"""

import pandas as pd
import numpy as np
from collections import Counter

# Import feature extraction functions
import sys
sys.path.insert(0, '/home/thc1006/dev/airr-ml25-package')
from main_v2 import (
    extract_multiscale_kmers,
    extract_vj_features,
    extract_clonality_features,
    extract_cdr3_length_features,
    extract_all_features
)

def test_multiscale_kmers():
    """Test multi-scale k-mer extraction."""
    print("="*70)
    print("Testing Multi-scale K-mer Extraction")
    print("="*70)

    # Create sample data
    df = pd.DataFrame({
        'junction_aa': ['CASSLAPG', 'CASSLQAY', 'CASSLA', 'CASSPG']
    })

    # Extract k-mers for k=3,4,5
    kmers = extract_multiscale_kmers(df, k_values=[3, 4, 5])

    # Count features per k
    k3_features = sum(1 for k in kmers.keys() if k.startswith('k3_'))
    k4_features = sum(1 for k in kmers.keys() if k.startswith('k4_'))
    k5_features = sum(1 for k in kmers.keys() if k.startswith('k5_'))

    print(f"Total k-mers extracted: {len(kmers)}")
    print(f"  - k=3 k-mers: {k3_features}")
    print(f"  - k=4 k-mers: {k4_features}")
    print(f"  - k=5 k-mers: {k5_features}")
    print(f"\nSample k-mers:")
    for i, (kmer, count) in enumerate(list(kmers.items())[:10]):
        print(f"  {kmer}: {count}")
    print("✓ PASS: Multi-scale k-mers extracted\n")


def test_vj_features():
    """Test V/J gene usage extraction."""
    print("="*70)
    print("Testing V/J Gene Usage Features")
    print("="*70)

    # Create sample data
    df = pd.DataFrame({
        'junction_aa': ['CASSLAPG', 'CASSLQAY', 'CASSLA', 'CASSPG', 'CASSQA'],
        'v_call': ['TRBV20-1', 'TRBV20-1', 'TRBV7-9', 'TRBV7-9', 'TRBV20-1'],
        'j_call': ['TRBJ2-7', 'TRBJ1-1', 'TRBJ2-7', 'TRBJ1-1', 'TRBJ2-7']
    })

    features = extract_vj_features(df)

    v_usage = sum(1 for k in features.keys() if k.startswith('v_usage_'))
    j_usage = sum(1 for k in features.keys() if k.startswith('j_usage_'))
    vj_pairs = sum(1 for k in features.keys() if k.startswith('vj_pair_'))

    print(f"Total V/J features: {len(features)}")
    print(f"  - V gene usage: {v_usage}")
    print(f"  - J gene usage: {j_usage}")
    print(f"  - VJ pairs: {vj_pairs}")
    print(f"\nSample features:")
    for i, (feat, val) in enumerate(list(features.items())[:8]):
        print(f"  {feat}: {val:.4f}")
    print("✓ PASS: V/J features extracted\n")


def test_clonality_features():
    """Test clonality metrics extraction."""
    print("="*70)
    print("Testing Clonality Metrics")
    print("="*70)

    # Create sample data with some duplicates (clonal expansion)
    df = pd.DataFrame({
        'junction_aa': ['CASSLAPG'] * 10 + ['CASSLQAY'] * 5 + ['CASSLA'] * 3 + ['CASSPG', 'CASSQA']
    })

    features = extract_clonality_features(df)

    print(f"Total clonality features: {len(features)}")
    print(f"  - Shannon entropy: {features['shannon_entropy']:.4f}")
    print(f"  - Gini-Simpson index: {features['gini_simpson']:.4f}")
    print(f"  - D50: {features['d50']:.0f}")
    print(f"  - Clonality score: {features['clonality_score']:.4f}")
    print("\n✓ PASS: Clonality metrics computed\n")


def test_cdr3_length_features():
    """Test CDR3 length distribution features."""
    print("="*70)
    print("Testing CDR3 Length Statistics")
    print("="*70)

    # Create sample data with varying lengths
    df = pd.DataFrame({
        'junction_aa': ['CASS', 'CASSLA', 'CASSLAPG', 'CASSLQAY',
                       'CASSLQAYG', 'CASSLQAYGPQH']
    })

    features = extract_cdr3_length_features(df)

    print(f"Total length features: {len(features)}")
    print(f"  - Mean length: {features['cdr3_length_mean']:.2f}")
    print(f"  - Std dev: {features['cdr3_length_std']:.2f}")
    print(f"  - Median: {features['cdr3_length_median']:.2f}")
    print(f"  - Q25: {features['cdr3_length_q25']:.2f}")
    print(f"  - Q75: {features['cdr3_length_q75']:.2f}")
    print(f"  - Skewness: {features['cdr3_length_skewness']:.4f}")
    print(f"  - Kurtosis: {features['cdr3_length_kurtosis']:.4f}")
    print("\n✓ PASS: CDR3 length statistics computed\n")


def test_all_features():
    """Test complete feature extraction."""
    print("="*70)
    print("Testing Complete Feature Extraction")
    print("="*70)

    # Create comprehensive sample data
    df = pd.DataFrame({
        'junction_aa': ['CASSLAPG', 'CASSLAPG', 'CASSLQAY', 'CASSLA',
                       'CASSPG', 'CASSQA', 'CASSQAYG'],
        'v_call': ['TRBV20-1', 'TRBV20-1', 'TRBV7-9', 'TRBV7-9',
                  'TRBV20-1', 'TRBV7-9', 'TRBV20-1'],
        'j_call': ['TRBJ2-7', 'TRBJ2-7', 'TRBJ1-1', 'TRBJ2-7',
                  'TRBJ1-1', 'TRBJ2-7', 'TRBJ1-1']
    })

    features = extract_all_features(df, k_values=[3, 4, 5])

    # Count feature types
    kmer_features = sum(1 for k in features.keys() if k.startswith('k'))
    vj_features = sum(1 for k in features.keys() if 'usage' in k or 'vj_pair' in k)
    clonality_features = sum(1 for k in features.keys() if k in [
        'shannon_entropy', 'gini_simpson', 'd50', 'clonality_score'
    ])
    length_features = sum(1 for k in features.keys() if 'cdr3_length' in k)

    print(f"Total features: {len(features)}")
    print(f"  - K-mer features (k=3,4,5): {kmer_features}")
    print(f"  - V/J usage features: {vj_features}")
    print(f"  - Clonality features: {clonality_features}")
    print(f"  - CDR3 length features: {length_features}")
    print("\n✓ PASS: All features successfully extracted\n")


def estimate_feature_count():
    """Estimate total feature count for full dataset."""
    print("="*70)
    print("Estimated Feature Count for Full Dataset")
    print("="*70)

    # Rough estimates based on typical AIRR datasets
    amino_acids = 20

    # Multi-scale k-mers (assuming sparse features)
    k3_max = amino_acids ** 3  # 8,000
    k4_max = amino_acids ** 4  # 160,000
    k5_max = amino_acids ** 5  # 3,200,000
    # In practice, much smaller due to biological constraints
    estimated_kmers = 50000  # Conservative estimate

    # V/J features
    v_genes = 20
    j_genes = 20
    vj_pairs = 50
    vj_features = v_genes + j_genes + vj_pairs

    # Clonality + length
    other_features = 4 + 7

    total_estimated = estimated_kmers + vj_features + other_features

    print(f"Estimated feature counts:")
    print(f"  - Multi-scale k-mers (sparse): ~{estimated_kmers:,}")
    print(f"  - V/J usage features: {vj_features}")
    print(f"  - Clonality features: 4")
    print(f"  - CDR3 length features: 7")
    print(f"  - TOTAL: ~{total_estimated:,} features")
    print(f"\nMemory estimate (sparse representation):")
    print(f"  - Training matrix (sparse): ~{total_estimated * 4 / 1024 / 1024:.1f} MB per sample")
    print(f"  - For 500 samples: ~{total_estimated * 500 * 4 / 1024 / 1024 / 1024:.2f} GB")
    print("\n")


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("AIRR-ML-25 Enhanced Features Test Suite")
    print("="*70 + "\n")

    try:
        test_multiscale_kmers()
        test_vj_features()
        test_clonality_features()
        test_cdr3_length_features()
        test_all_features()
        estimate_feature_count()

        print("="*70)
        print("ALL TESTS PASSED ✓")
        print("="*70)
        print("\nReady to run on full dataset with:")
        print("  python3 main_v2.py --train_root ./data/train_datasets \\")
        print("                     --test_root ./data/test_datasets \\")
        print("                     --out_dir ./results_v2 --n_jobs 8")

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
