#!/usr/bin/env python3
"""
Test script for champion pipeline.

Validates the pipeline with a small subset of data before running on full dataset.
"""

import os
import sys
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from pipeline.champion_pipeline import ImmuneStatePredictor, PipelineConfig


def test_basic_interface():
    """Test that the predictor implements required interface."""
    print("="*60)
    print("Test 1: Interface Validation")
    print("="*60)

    predictor = ImmuneStatePredictor(n_jobs=2, device='cpu')

    # Check required methods
    assert hasattr(predictor, 'fit'), "Missing fit method"
    assert hasattr(predictor, 'predict_proba'), "Missing predict_proba method"
    assert hasattr(predictor, 'identify_associated_sequences'), "Missing identify_associated_sequences method"

    print("✓ All required methods present")
    print()


def test_config():
    """Test configuration object."""
    print("="*60)
    print("Test 2: Configuration")
    print("="*60)

    config = PipelineConfig(
        n_jobs=4,
        device='cuda',
        k_values=[3, 4, 5],
        val_size=0.2,
        random_state=42
    )

    assert config.n_jobs == 4
    assert config.device == 'cuda'
    assert config.k_values == [3, 4, 5]
    assert config.val_size == 0.2

    print(f"✓ Config created: {config}")
    print()


def test_feature_extraction():
    """Test feature extraction functions."""
    print("="*60)
    print("Test 3: Feature Extraction")
    print("="*60)

    from pipeline.champion_pipeline import (
        extract_kmers_multiscale,
        extract_vj_features,
        extract_diversity_metrics,
        extract_length_features
    )

    # Test k-mer extraction
    sequences = ['CASSLGQAY', 'CASSYNLTK', 'CASSLGQAY']  # One duplicate
    kmers = extract_kmers_multiscale(sequences, k_values=[3])
    assert len(kmers) > 0, "No k-mers extracted"
    print(f"✓ Extracted {len(kmers)} k-mers")

    # Test V/J features
    v_calls = ['TRBV20-1', 'TRBV6-2', 'TRBV20-1']
    j_calls = ['TRBJ2-7', 'TRBJ2-7', 'TRBJ1-3']
    vj_feats = extract_vj_features(v_calls, j_calls)
    assert len(vj_feats) > 0, "No V/J features extracted"
    print(f"✓ Extracted {len(vj_feats)} V/J features")

    # Test diversity metrics
    div_feats = extract_diversity_metrics(sequences)
    assert 'shannon' in div_feats
    assert 'gini_simpson' in div_feats
    assert 'clonality' in div_feats
    print(f"✓ Diversity metrics: Shannon={div_feats['shannon']:.3f}, Clonality={div_feats['clonality']:.3f}")

    # Test length features
    len_feats = extract_length_features(sequences)
    assert 'len_mean' in len_feats
    assert 'len_std' in len_feats
    print(f"✓ Length metrics: Mean={len_feats['len_mean']:.1f}, Std={len_feats['len_std']:.1f}")

    print()


def test_small_dataset():
    """Test on smallest available dataset."""
    print("="*60)
    print("Test 4: Small Dataset Training")
    print("="*60)

    # Find smallest training dataset
    train_root = './data/train_datasets/train_datasets'
    if not os.path.exists(train_root):
        print("⚠ Skipping: Training data not found")
        print()
        return

    datasets = [d for d in os.listdir(train_root) if d.startswith('train_dataset_')]
    if not datasets:
        print("⚠ Skipping: No datasets found")
        print()
        return

    # Use dataset 1 (usually smallest)
    train_dir = os.path.join(train_root, 'train_dataset_1')
    if not os.path.exists(train_dir):
        print("⚠ Skipping: train_dataset_1 not found")
        print()
        return

    print(f"Training on: {train_dir}")

    # Create predictor with minimal config
    predictor = ImmuneStatePredictor(
        n_jobs=2,
        device='cpu',  # Use CPU for testing
        k_values=[3],  # Only 3-mers for speed
        val_size=0.3,
        verbose=True
    )

    # Train
    predictor.fit(train_dir)

    assert predictor.model is not None, "Model not trained"
    print("✓ Model trained successfully")

    # Check sequences identified
    if predictor.important_sequences_ is not None:
        print(f"✓ Identified {len(predictor.important_sequences_)} sequences")
    else:
        print("⚠ No sequences identified")

    print()


def test_prediction_format():
    """Test prediction output format."""
    print("="*60)
    print("Test 5: Prediction Format")
    print("="*60)

    # Create mock predictions
    preds = pd.DataFrame({
        'ID': ['rep_001', 'rep_002'],
        'dataset': ['test_dataset_1', 'test_dataset_1'],
        'label_positive_probability': [0.75, 0.23],
        'junction_aa': ['-999.0', '-999.0'],
        'v_call': ['-999.0', '-999.0'],
        'j_call': ['-999.0', '-999.0']
    })

    # Check columns
    required_cols = ['ID', 'dataset', 'label_positive_probability', 'junction_aa', 'v_call', 'j_call']
    for col in required_cols:
        assert col in preds.columns, f"Missing column: {col}"

    # Check probabilities in range
    probs = preds['label_positive_probability']
    assert (probs >= 0).all() and (probs <= 1).all(), "Probabilities out of range"

    print("✓ Prediction format valid")
    print()


def test_sequence_format():
    """Test sequence output format."""
    print("="*60)
    print("Test 6: Sequence Format")
    print("="*60)

    # Create mock sequences
    seqs = pd.DataFrame({
        'ID': ['train_dataset_1_seq_top_1', 'train_dataset_1_seq_top_2'],
        'dataset': ['train_dataset_1', 'train_dataset_1'],
        'label_positive_probability': [-999.0, -999.0],
        'junction_aa': ['CASSLGQAY', 'CASSYNLTK'],
        'v_call': ['TRBV20-1', 'TRBV6-2'],
        'j_call': ['TRBJ2-7', 'TRBJ2-7']
    })

    # Check format
    assert (seqs['label_positive_probability'] == -999.0).all(), "Sequences should have -999.0 probability"
    assert seqs['junction_aa'].notna().all(), "Sequences should have junction_aa"

    print("✓ Sequence format valid")
    print()


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*60)
    print("CHAMPION PIPELINE TEST SUITE")
    print("="*60 + "\n")

    tests = [
        test_basic_interface,
        test_config,
        test_feature_extraction,
        test_small_dataset,
        test_prediction_format,
        test_sequence_format
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ Test failed: {test.__name__}")
            print(f"  Error: {e}")
            print()
            failed += 1

    print("="*60)
    print(f"TEST SUMMARY: {passed} passed, {failed} failed")
    print("="*60)

    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
