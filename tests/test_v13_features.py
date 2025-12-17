#!/usr/bin/env python3
"""
Quick test for V13 VJ Pairs feature extraction.
Tests the new feature extraction functions without running full pipeline.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from champion_v13 import (
    extract_v_family,
    extract_j_family,
    extract_vj_pair_features,
    V13Config
)


def test_v_family_extraction():
    """Test V gene family extraction."""
    print("Testing V family extraction...")

    test_cases = [
        ("TRBV20-1", "TRBV20"),
        ("TRBV20*01", "TRBV20"),
        ("TRBV7-9*01", "TRBV7"),
        ("UNK", "UNK"),
        (None, "UNK"),
    ]

    for v_call, expected in test_cases:
        result = extract_v_family(v_call)
        assert result == expected, f"Failed: {v_call} -> {result} (expected {expected})"
        print(f"  ✓ {v_call} -> {result}")

    print("  All V family tests passed!\n")


def test_j_family_extraction():
    """Test J gene family extraction."""
    print("Testing J family extraction...")

    test_cases = [
        ("TRBJ2-7", "TRBJ2"),
        ("TRBJ2*01", "TRBJ2"),
        ("TRBJ1-1*01", "TRBJ1"),
        ("UNK", "UNK"),
        (None, "UNK"),
    ]

    for j_call, expected in test_cases:
        result = extract_j_family(j_call)
        assert result == expected, f"Failed: {j_call} -> {result} (expected {expected})"
        print(f"  ✓ {j_call} -> {result}")

    print("  All J family tests passed!\n")


def test_vj_pair_features():
    """Test VJ pair feature extraction."""
    print("Testing VJ pair feature extraction...")

    # Create sample data
    df = pd.DataFrame({
        'v_call': ['TRBV20-1', 'TRBV20-1', 'TRBV7-9', 'TRBV20-1', 'TRBV7-9'],
        'j_call': ['TRBJ2-7', 'TRBJ2-7', 'TRBJ1-1', 'TRBJ2-7', 'TRBJ1-1']
    })

    # Define vocabularies
    vj_vocab = [
        ('TRBV20-1', 'TRBJ2-7'),
        ('TRBV7-9', 'TRBJ1-1'),
        ('TRBV1-1', 'TRBJ1-1')  # Not in data
    ]

    v_vocab = ['TRBV20', 'TRBV7', 'TRBV1']
    j_vocab = ['TRBJ2', 'TRBJ1']

    # Extract features
    features = extract_vj_pair_features(df, vj_vocab, v_vocab, j_vocab)

    # Validate feature structure
    expected_features = [
        'vj_pair_TRBV20-1_TRBJ2-7',
        'vj_pair_TRBV7-9_TRBJ1-1',
        'vj_entropy',
        'vj_unique',
        'vj_max_freq',
        'v_family_TRBV20',
        'v_family_TRBV7',
        'j_family_TRBJ2',
        'j_family_TRBJ1',
    ]

    for feat in expected_features:
        assert feat in features, f"Missing feature: {feat}"
        print(f"  ✓ {feat}: {features[feat]:.4f}")

    # Validate values
    assert abs(features['vj_pair_TRBV20-1_TRBJ2-7'] - 0.6) < 1e-6, "Wrong VJ pair freq"
    assert abs(features['vj_pair_TRBV7-9_TRBJ1-1'] - 0.4) < 1e-6, "Wrong VJ pair freq"
    assert features['vj_unique'] == 2, "Wrong unique count"
    assert abs(features['v_family_TRBV20'] - 0.6) < 1e-6, "Wrong V family freq"
    assert abs(features['j_family_TRBJ2'] - 0.6) < 1e-6, "Wrong J family freq"

    print("  All VJ pair feature tests passed!\n")


def test_config():
    """Test V13Config dataclass."""
    print("Testing V13Config...")

    config = V13Config()

    # Check defaults
    assert config.n_threads == 8
    assert config.use_gpu == True
    assert config.n_top_vj_pairs == 500
    assert config.n_top_v_families == 20
    assert config.n_top_j_families == 10

    # Check ensemble weights sum to 1
    total_weight = config.weight_xgb + config.weight_lgb + config.weight_cb
    assert abs(total_weight - 1.0) < 1e-6, f"Weights sum to {total_weight}, not 1.0"

    print(f"  ✓ Config loaded: {config.n_threads} threads, GPU={config.use_gpu}")
    print(f"  ✓ Ensemble weights: XGB={config.weight_xgb}, LGB={config.weight_lgb}, CB={config.weight_cb}")
    print(f"  ✓ VJ vocab sizes: pairs={config.n_top_vj_pairs}, V={config.n_top_v_families}, J={config.n_top_j_families}")
    print("  All config tests passed!\n")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Champion V13 Feature Extraction Tests")
    print("=" * 60)
    print()

    try:
        test_v_family_extraction()
        test_j_family_extraction()
        test_vj_pair_features()
        test_config()

        print("=" * 60)
        print("✓ ALL TESTS PASSED!")
        print("=" * 60)
        return 0

    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
