#!/usr/bin/env python3
"""
Unit tests for VJ Features Module (champion_v13).

Tests VJ pair extraction, gene family parsing, diversity metrics,
and edge case handling.
"""

import pytest
import numpy as np
import pandas as pd
from collections import Counter


# ============================================================================
# Test VJ Pair Extraction
# ============================================================================

@pytest.mark.unit
def test_vj_pair_extraction(sample_repertoire):
    """Test VJ pair feature extraction."""
    from champion_v13_vj import VJFeatureExtractor

    extractor = VJFeatureExtractor()
    features = extractor.extract_vj_pairs(sample_repertoire)

    # Should return frequency dict
    assert isinstance(features, dict)
    assert len(features) > 0

    # Check specific pair
    expected_pair = ('TRBV20-1', 'TRBJ2-7')
    assert f'vj_pair_{expected_pair[0]}_{expected_pair[1]}' in features

    # Frequencies should sum to ~1.0
    total_freq = sum(features.values())
    assert abs(total_freq - 1.0) < 0.01


@pytest.mark.unit
def test_vj_pair_frequency_calculation(sample_vj_pairs):
    """Test VJ pair frequency calculation is correct."""
    from champion_v13_vj import VJFeatureExtractor

    extractor = VJFeatureExtractor()

    # Manually count pairs
    pair_counts = Counter(sample_vj_pairs)
    total = len(sample_vj_pairs)

    # Extract features
    df = pd.DataFrame({
        'v_call': [p[0] for p in sample_vj_pairs],
        'j_call': [p[1] for p in sample_vj_pairs]
    })
    features = extractor.extract_vj_pairs(df)

    # Verify frequencies
    for pair, count in pair_counts.items():
        expected_freq = count / total
        feature_name = f'vj_pair_{pair[0]}_{pair[1]}'
        assert feature_name in features
        assert abs(features[feature_name] - expected_freq) < 1e-6


@pytest.mark.unit
def test_vj_pair_with_missing_values():
    """Test VJ pair extraction with missing values."""
    from champion_v13_vj import VJFeatureExtractor

    extractor = VJFeatureExtractor()

    df = pd.DataFrame({
        'v_call': ['TRBV20-1', None, 'TRBV5-1', np.nan],
        'j_call': ['TRBJ2-7', 'TRBJ2-1', None, 'TRBJ1-3']
    })

    features = extractor.extract_vj_pairs(df)

    # Should handle missing values gracefully
    assert isinstance(features, dict)
    assert len(features) > 0


# ============================================================================
# Test Gene Family Parsing
# ============================================================================

@pytest.mark.unit
def test_parse_v_gene_family():
    """Test V gene family parsing."""
    from champion_v13_vj import parse_gene_family

    # Test standard format
    assert parse_gene_family('TRBV20-1') == 'TRBV20'
    assert parse_gene_family('TRBV5-1') == 'TRBV5'

    # Test with asterisk
    assert parse_gene_family('TRBV20-1*01') == 'TRBV20'

    # Test edge cases
    assert parse_gene_family('TRBV1') == 'TRBV1'
    assert parse_gene_family('UNK') == 'UNK'


@pytest.mark.unit
def test_parse_j_gene_family():
    """Test J gene family parsing."""
    from champion_v13_vj import parse_gene_family

    # Test standard format
    assert parse_gene_family('TRBJ2-7') == 'TRBJ2'
    assert parse_gene_family('TRBJ1-3') == 'TRBJ1'

    # Test with asterisk
    assert parse_gene_family('TRBJ2-7*01') == 'TRBJ2'


@pytest.mark.unit
def test_gene_family_extraction():
    """Test gene family feature extraction."""
    from champion_v13_vj import VJFeatureExtractor

    extractor = VJFeatureExtractor()

    df = pd.DataFrame({
        'v_call': ['TRBV20-1', 'TRBV20-2', 'TRBV5-1'],
        'j_call': ['TRBJ2-7', 'TRBJ2-1', 'TRBJ1-3']
    })

    features = extractor.extract_gene_families(df)

    # Should have V and J family frequencies
    assert 'v_family_TRBV20' in features
    assert 'v_family_TRBV5' in features
    assert 'j_family_TRBJ2' in features
    assert 'j_family_TRBJ1' in features

    # Check frequencies
    assert abs(features['v_family_TRBV20'] - 2/3) < 1e-6
    assert abs(features['v_family_TRBV5'] - 1/3) < 1e-6


# ============================================================================
# Test VJ Diversity Metrics
# ============================================================================

@pytest.mark.unit
def test_vj_diversity_shannon():
    """Test Shannon entropy for VJ pairs."""
    from champion_v13_vj import VJFeatureExtractor

    extractor = VJFeatureExtractor()

    # Uniform distribution (high diversity)
    df_uniform = pd.DataFrame({
        'v_call': ['TRBV1', 'TRBV2', 'TRBV3', 'TRBV4'],
        'j_call': ['TRBJ1', 'TRBJ2', 'TRBJ3', 'TRBJ4']
    })

    # Skewed distribution (low diversity)
    df_skewed = pd.DataFrame({
        'v_call': ['TRBV1', 'TRBV1', 'TRBV1', 'TRBV2'],
        'j_call': ['TRBJ1', 'TRBJ1', 'TRBJ1', 'TRBJ2']
    })

    diversity_uniform = extractor.extract_diversity_metrics(df_uniform)
    diversity_skewed = extractor.extract_diversity_metrics(df_skewed)

    # Uniform should have higher entropy
    assert 'vj_shannon_entropy' in diversity_uniform
    assert 'vj_shannon_entropy' in diversity_skewed
    assert diversity_uniform['vj_shannon_entropy'] > diversity_skewed['vj_shannon_entropy']


@pytest.mark.unit
def test_vj_diversity_gini():
    """Test Gini-Simpson index for VJ pairs."""
    from champion_v13_vj import VJFeatureExtractor

    extractor = VJFeatureExtractor()

    df = pd.DataFrame({
        'v_call': ['TRBV1', 'TRBV1', 'TRBV2'],
        'j_call': ['TRBJ1', 'TRBJ1', 'TRBJ2']
    })

    diversity = extractor.extract_diversity_metrics(df)

    assert 'vj_gini_simpson' in diversity
    assert 0 <= diversity['vj_gini_simpson'] <= 1


@pytest.mark.unit
def test_vj_diversity_unique_pairs():
    """Test unique VJ pair count."""
    from champion_v13_vj import VJFeatureExtractor

    extractor = VJFeatureExtractor()

    df = pd.DataFrame({
        'v_call': ['TRBV1', 'TRBV1', 'TRBV2', 'TRBV3'],
        'j_call': ['TRBJ1', 'TRBJ1', 'TRBJ2', 'TRBJ3']
    })

    diversity = extractor.extract_diversity_metrics(df)

    assert 'vj_n_unique_pairs' in diversity
    assert diversity['vj_n_unique_pairs'] == 3  # 3 unique pairs


# ============================================================================
# Test VJ Usage Patterns
# ============================================================================

@pytest.mark.unit
def test_vj_usage_statistics():
    """Test V/J usage statistics."""
    from champion_v13_vj import VJFeatureExtractor

    extractor = VJFeatureExtractor()

    df = pd.DataFrame({
        'v_call': ['TRBV20-1'] * 5 + ['TRBV5-1'] * 3 + ['TRBV6-2'] * 2,
        'j_call': ['TRBJ2-7'] * 6 + ['TRBJ1-3'] * 4
    })

    features = extractor.extract_usage_statistics(df)

    # Should include top usage stats
    assert 'v_top1_freq' in features
    assert 'j_top1_freq' in features
    assert 'v_n_unique' in features
    assert 'j_n_unique' in features

    # Check values
    assert abs(features['v_top1_freq'] - 0.5) < 1e-6  # TRBV20-1: 5/10
    assert abs(features['j_top1_freq'] - 0.6) < 1e-6  # TRBJ2-7: 6/10
    assert features['v_n_unique'] == 3
    assert features['j_n_unique'] == 2


# ============================================================================
# Test Empty Repertoire Handling
# ============================================================================

@pytest.mark.unit
def test_empty_repertoire():
    """Test handling of empty repertoire."""
    from champion_v13_vj import VJFeatureExtractor

    extractor = VJFeatureExtractor()

    empty_df = pd.DataFrame({'v_call': [], 'j_call': []})

    # Should return default features or raise informative error
    with pytest.raises(ValueError, match="empty|no data"):
        extractor.extract_vj_pairs(empty_df)


@pytest.mark.unit
def test_all_missing_values():
    """Test repertoire with all missing values."""
    from champion_v13_vj import VJFeatureExtractor

    extractor = VJFeatureExtractor()

    df = pd.DataFrame({
        'v_call': [None, np.nan, None],
        'j_call': [None, np.nan, None]
    })

    # Should handle gracefully
    with pytest.raises(ValueError, match="no valid"):
        extractor.extract_vj_pairs(df)


# ============================================================================
# Test VJ Combination Features
# ============================================================================

@pytest.mark.unit
def test_vj_combination_ratio():
    """Test V/J combination ratio features."""
    from champion_v13_vj import VJFeatureExtractor

    extractor = VJFeatureExtractor()

    df = pd.DataFrame({
        'v_call': ['TRBV20-1', 'TRBV20-1', 'TRBV5-1'],
        'j_call': ['TRBJ2-7', 'TRBJ1-3', 'TRBJ2-7']
    })

    features = extractor.extract_combination_features(df)

    # Should include combination metrics
    assert 'vj_combinations_per_v' in features
    assert 'vj_combinations_per_j' in features

    # TRBV20-1 combines with 2 J genes out of 3 uses
    assert features['vj_combinations_per_v'] > 0


@pytest.mark.unit
def test_vj_promiscuity():
    """Test V/J gene promiscuity metrics."""
    from champion_v13_vj import VJFeatureExtractor

    extractor = VJFeatureExtractor()

    # TRBV20-1 is promiscuous (pairs with multiple J)
    df_promiscuous = pd.DataFrame({
        'v_call': ['TRBV20-1', 'TRBV20-1', 'TRBV20-1'],
        'j_call': ['TRBJ2-7', 'TRBJ1-3', 'TRBJ2-1']
    })

    # TRBV5-1 is specific (pairs with only one J)
    df_specific = pd.DataFrame({
        'v_call': ['TRBV5-1', 'TRBV5-1', 'TRBV5-1'],
        'j_call': ['TRBJ2-7', 'TRBJ2-7', 'TRBJ2-7']
    })

    features_prom = extractor.extract_promiscuity(df_promiscuous)
    features_spec = extractor.extract_promiscuity(df_specific)

    # Promiscuous should have higher score
    assert features_prom['v_promiscuity'] > features_spec['v_promiscuity']


# ============================================================================
# Test Integration with Other Features
# ============================================================================

@pytest.mark.unit
def test_extract_all_vj_features(sample_repertoire):
    """Test extracting all VJ features at once."""
    from champion_v13_vj import VJFeatureExtractor

    extractor = VJFeatureExtractor()
    all_features = extractor.extract_all_features(sample_repertoire)

    # Should include all feature types
    feature_prefixes = ['vj_pair_', 'v_family_', 'j_family_', 'vj_shannon', 'vj_gini', 'v_top1']

    for prefix in feature_prefixes:
        assert any(k.startswith(prefix) for k in all_features.keys()), \
            f"Missing features with prefix: {prefix}"

    # All values should be numeric
    for v in all_features.values():
        assert isinstance(v, (int, float, np.number))
        assert not np.isnan(v)
        assert not np.isinf(v)


# ============================================================================
# Test Deterministic Behavior
# ============================================================================

@pytest.mark.unit
def test_deterministic_extraction(sample_repertoire):
    """Test that VJ feature extraction is deterministic."""
    from champion_v13_vj import VJFeatureExtractor

    extractor1 = VJFeatureExtractor()
    features1 = extractor1.extract_all_features(sample_repertoire)

    extractor2 = VJFeatureExtractor()
    features2 = extractor2.extract_all_features(sample_repertoire)

    # Should produce identical results
    assert features1.keys() == features2.keys()
    for key in features1.keys():
        assert features1[key] == features2[key]


# ============================================================================
# Test Feature Vocabulary Management
# ============================================================================

@pytest.mark.unit
def test_feature_vocabulary_consistency():
    """Test that feature vocabulary is consistent across repertoires."""
    from champion_v13_vj import VJFeatureExtractor

    extractor = VJFeatureExtractor()

    # Build vocabulary from first repertoire
    df1 = pd.DataFrame({
        'v_call': ['TRBV20-1', 'TRBV5-1'],
        'j_call': ['TRBJ2-7', 'TRBJ1-3']
    })
    extractor.fit_vocabulary(df1)
    vocab_size_1 = len(extractor.vocabulary)

    # Second repertoire with new genes
    df2 = pd.DataFrame({
        'v_call': ['TRBV20-1', 'TRBV7-2'],  # TRBV7-2 is new
        'j_call': ['TRBJ2-7', 'TRBJ2-1']    # TRBJ2-1 is new
    })

    # Using same vocabulary should give zero for unseen pairs
    features = extractor.transform(df2)

    # Should have same number of features
    assert len(features) == vocab_size_1


@pytest.mark.unit
def test_unknown_genes_handling():
    """Test handling of unknown gene names."""
    from champion_v13_vj import VJFeatureExtractor

    extractor = VJFeatureExtractor()

    df = pd.DataFrame({
        'v_call': ['TRBV20-1', 'UNK', 'TRBV5-1'],
        'j_call': ['TRBJ2-7', 'TRBJ1-3', 'UNK']
    })

    features = extractor.extract_all_features(df)

    # Should handle gracefully
    assert isinstance(features, dict)
    assert len(features) > 0
