"""Quick test script for gene_usage module."""

import pandas as pd
import numpy as np
from src.features.gene_usage import (
    normalize_gene_name,
    extract_v_gene_usage,
    extract_j_gene_usage,
    extract_vj_pair_usage,
    GeneUsageFeaturizer,
    get_gene_diversity
)


def test_normalize_gene_name():
    """Test gene name normalization."""
    assert normalize_gene_name('TRBV20-1*01') == 'TRBV20-1'
    assert normalize_gene_name('TRBV20-1*02') == 'TRBV20-1'
    assert normalize_gene_name('TRBJ2-7') == 'TRBJ2-7'
    assert normalize_gene_name(None) == 'UNKNOWN'
    assert normalize_gene_name(np.nan) == 'UNKNOWN'
    print("✓ normalize_gene_name tests passed")


def test_extract_v_gene_usage():
    """Test V gene usage extraction."""
    df = pd.DataFrame({
        'v_call': ['TRBV20-1*01', 'TRBV20-1*02', 'TRBV7-3']
    })
    usage = extract_v_gene_usage(df)

    assert 'TRBV20-1' in usage
    assert 'TRBV7-3' in usage
    assert abs(usage['TRBV20-1'] - 2/3) < 0.01
    assert abs(usage['TRBV7-3'] - 1/3) < 0.01
    assert abs(sum(usage.values()) - 1.0) < 0.01
    print("✓ extract_v_gene_usage tests passed")


def test_extract_j_gene_usage():
    """Test J gene usage extraction."""
    df = pd.DataFrame({
        'j_call': ['TRBJ2-7', 'TRBJ2-7', 'TRBJ1-1']
    })
    usage = extract_j_gene_usage(df)

    assert 'TRBJ2-7' in usage
    assert 'TRBJ1-1' in usage
    assert abs(usage['TRBJ2-7'] - 2/3) < 0.01
    assert abs(usage['TRBJ1-1'] - 1/3) < 0.01
    print("✓ extract_j_gene_usage tests passed")


def test_extract_vj_pair_usage():
    """Test VJ pair usage extraction."""
    df = pd.DataFrame({
        'v_call': ['TRBV20-1', 'TRBV20-1', 'TRBV7-3'],
        'j_call': ['TRBJ2-7', 'TRBJ2-7', 'TRBJ1-1']
    })
    usage = extract_vj_pair_usage(df, top_k=2)

    assert 'TRBV20-1_TRBJ2-7' in usage
    assert 'TRBV7-3_TRBJ1-1' in usage
    assert abs(usage['TRBV20-1_TRBJ2-7'] - 2/3) < 0.01
    print("✓ extract_vj_pair_usage tests passed")


def test_gene_usage_featurizer():
    """Test GeneUsageFeaturizer class."""
    # Create sample training data
    train_rep1 = pd.DataFrame({
        'v_call': ['TRBV20-1'] * 50 + ['TRBV7-3'] * 30 + ['TRBV12-1'] * 20,
        'j_call': ['TRBJ2-7'] * 60 + ['TRBJ1-1'] * 40
    })

    train_rep2 = pd.DataFrame({
        'v_call': ['TRBV20-1'] * 40 + ['TRBV7-3'] * 40 + ['TRBV5-1'] * 20,
        'j_call': ['TRBJ2-7'] * 50 + ['TRBJ1-1'] * 50
    })

    # Initialize and fit
    featurizer = GeneUsageFeaturizer(top_v_genes=3, top_j_genes=2, top_vj_pairs=4)
    featurizer.fit([train_rep1, train_rep2])

    # Check fitted attributes
    assert featurizer.v_genes_ is not None
    assert featurizer.j_genes_ is not None
    assert featurizer.vj_pairs_ is not None
    assert len(featurizer.v_genes_) == 3
    assert len(featurizer.j_genes_) == 2
    assert len(featurizer.vj_pairs_) == 4

    # Transform
    test_rep = pd.DataFrame({
        'v_call': ['TRBV20-1'] * 60 + ['TRBV7-3'] * 40,
        'j_call': ['TRBJ2-7'] * 50 + ['TRBJ1-1'] * 50
    })

    features = featurizer.transform(test_rep)
    assert features.shape == (9,)  # 3 V + 2 J + 4 VJ = 9
    assert all(f >= 0 for f in features)
    assert any(f > 0 for f in features)

    # Test transform_many
    features_many = featurizer.transform_many([test_rep, test_rep])
    assert features_many.shape == (2, 9)

    # Test feature names
    names = featurizer.feature_names
    assert len(names) == 9
    assert all(name.startswith(('v_', 'j_', 'vj_')) for name in names)

    # Test n_features
    assert featurizer.n_features == 9

    print("✓ GeneUsageFeaturizer tests passed")


def test_gene_diversity():
    """Test gene diversity metrics."""
    # Uniform distribution (high diversity)
    df_uniform = pd.DataFrame({
        'v_call': ['TRBV1'] * 25 + ['TRBV2'] * 25 + ['TRBV3'] * 25 + ['TRBV4'] * 25
    })

    # Skewed distribution (low diversity)
    df_skewed = pd.DataFrame({
        'v_call': ['TRBV1'] * 90 + ['TRBV2'] * 10
    })

    shannon_uniform = get_gene_diversity(df_uniform, 'v_call', 'shannon')
    shannon_skewed = get_gene_diversity(df_skewed, 'v_call', 'shannon')

    # Uniform should have higher diversity
    assert shannon_uniform > shannon_skewed

    simpson_uniform = get_gene_diversity(df_uniform, 'v_call', 'simpson')
    simpson_skewed = get_gene_diversity(df_skewed, 'v_call', 'simpson')

    assert simpson_uniform > simpson_skewed

    print("✓ get_gene_diversity tests passed")


def test_missing_data_handling():
    """Test handling of missing data."""
    df = pd.DataFrame({
        'v_call': ['TRBV20-1', None, 'TRBV7-3', np.nan],
        'j_call': ['TRBJ2-7', 'TRBJ1-1', None, np.nan]
    })

    # Should ignore missing values
    v_usage = extract_v_gene_usage(df)
    assert 'UNKNOWN' not in v_usage
    assert len(v_usage) == 2

    j_usage = extract_j_gene_usage(df)
    assert 'UNKNOWN' not in j_usage
    assert len(j_usage) == 2

    vj_usage = extract_vj_pair_usage(df)
    assert all('UNKNOWN' not in pair for pair in vj_usage.keys())

    print("✓ Missing data handling tests passed")


def run_all_tests():
    """Run all tests."""
    print("\nRunning gene_usage module tests...")
    print("=" * 50)

    test_normalize_gene_name()
    test_extract_v_gene_usage()
    test_extract_j_gene_usage()
    test_extract_vj_pair_usage()
    test_gene_usage_featurizer()
    test_gene_diversity()
    test_missing_data_handling()

    print("=" * 50)
    print("✓ All tests passed!")


if __name__ == "__main__":
    run_all_tests()
