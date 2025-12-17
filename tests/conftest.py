#!/usr/bin/env python3
"""
Pytest configuration and fixtures for champion_v13 tests.

Provides reusable fixtures for ESM2, VJ features, CatBoost, and mock data.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil
import torch


# ============================================================================
# Pytest Hooks
# ============================================================================

def pytest_configure(config):
    """Configure pytest."""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "gpu: marks tests requiring GPU")
    config.addinivalue_line("markers", "integration: marks integration tests")
    config.addinivalue_line("markers", "unit: marks unit tests")
    config.addinivalue_line("markers", "performance: marks performance benchmarks")
    config.addinivalue_line("markers", "requires_data: marks tests requiring full dataset")


def pytest_collection_modifyitems(config, items):
    """Auto-mark GPU tests if they use GPU fixtures."""
    for item in items:
        if "esm2_extractor" in item.fixturenames or "gpu_available" in item.fixturenames:
            item.add_marker(pytest.mark.gpu)


# ============================================================================
# Session-level Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def gpu_available():
    """Check if GPU is available."""
    return torch.cuda.is_available()


@pytest.fixture(scope="session")
def device(gpu_available):
    """Get device for testing."""
    return 'cuda' if gpu_available else 'cpu'


@pytest.fixture(scope="session")
def test_data_root():
    """Root directory for test data."""
    return Path(__file__).parent / "fixtures"


# ============================================================================
# Mock Data Fixtures
# ============================================================================

@pytest.fixture
def sample_sequences():
    """Sample CDR3 sequences for testing."""
    return [
        'CASSLAPGATNEKLFF',
        'CASSLGQAYEQYF',
        'CASSYNLTK',
        'CASSLAPGATNEKLFF',  # Duplicate for testing
        'CASSLGQAYEQYF',     # Duplicate for testing
        'CASSIRSSYEQYF',
        'CASSQETQYF',
        'CASSPGQGADEQFF',
    ]


@pytest.fixture
def sample_repertoire():
    """Sample repertoire DataFrame."""
    return pd.DataFrame({
        'junction_aa': [
            'CASSLAPGATNEKLFF',
            'CASSLGQAYEQYF',
            'CASSYNLTK',
            'CASSLAPGATNEKLFF',
            'CASSIRSSYEQYF',
            'CASSQETQYF',
        ],
        'v_call': ['TRBV20-1', 'TRBV5-1', 'TRBV20-1', 'TRBV20-1', 'TRBV6-2', 'TRBV7-2'],
        'j_call': ['TRBJ2-7', 'TRBJ2-1', 'TRBJ2-7', 'TRBJ2-7', 'TRBJ1-3', 'TRBJ2-7'],
        'templates': [100, 50, 30, 100, 20, 10]
    })


@pytest.fixture
def sample_metadata():
    """Sample metadata DataFrame."""
    return pd.DataFrame({
        'repertoire_id': ['rep_001', 'rep_002', 'rep_003', 'rep_004', 'rep_005'],
        'filename': ['rep_001.tsv', 'rep_002.tsv', 'rep_003.tsv', 'rep_004.tsv', 'rep_005.tsv'],
        'label_positive': [1, 0, 1, 0, 1]
    })


@pytest.fixture
def sample_vj_pairs():
    """Sample VJ pair data."""
    return [
        ('TRBV20-1', 'TRBJ2-7'),
        ('TRBV5-1', 'TRBJ2-1'),
        ('TRBV20-1', 'TRBJ2-7'),  # Duplicate
        ('TRBV6-2', 'TRBJ1-3'),
    ]


# ============================================================================
# Temporary Directory Fixtures
# ============================================================================

@pytest.fixture
def temp_dataset_dir(tmp_path, sample_metadata, sample_repertoire):
    """Create temporary dataset directory with mock data."""
    dataset_dir = tmp_path / "train_dataset_1"
    dataset_dir.mkdir()

    # Write metadata
    metadata_path = dataset_dir / "metadata.csv"
    sample_metadata.to_csv(metadata_path, index=False)

    # Write sample TSV files
    for _, row in sample_metadata.iterrows():
        tsv_path = dataset_dir / row['filename']
        sample_repertoire.to_csv(tsv_path, sep='\t', index=False)

    yield dataset_dir

    # Cleanup
    shutil.rmtree(dataset_dir)


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create temporary output directory."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    yield output_dir
    shutil.rmtree(output_dir)


# ============================================================================
# ESM2 Fixtures (will be implemented in champion_v13)
# ============================================================================

@pytest.fixture
def esm2_mock_embeddings():
    """Mock ESM2 embeddings for testing without loading model."""
    # Mock embedding: 5 sequences x 1280 dimensions
    np.random.seed(42)
    return np.random.randn(5, 1280).astype(np.float32)


@pytest.fixture
def esm2_expected_stats_shape():
    """Expected shape of aggregated ESM2 statistics."""
    # Mean, std, min, max for 1280 dims = 5120 features
    return (1, 5120)


# ============================================================================
# VJ Features Fixtures
# ============================================================================

@pytest.fixture
def vj_gene_families():
    """Common VJ gene families for testing."""
    return {
        'v_families': ['TRBV20', 'TRBV5', 'TRBV6', 'TRBV7'],
        'j_families': ['TRBJ2', 'TRBJ1']
    }


# ============================================================================
# CatBoost Fixtures
# ============================================================================

@pytest.fixture
def catboost_params():
    """Default CatBoost parameters for testing."""
    return {
        'iterations': 10,  # Few iterations for fast testing
        'learning_rate': 0.1,
        'depth': 4,
        'task_type': 'CPU',  # Use CPU for testing
        'verbose': False,
        'random_seed': 42
    }


@pytest.fixture
def sample_features_labels():
    """Sample features and labels for model testing."""
    np.random.seed(42)
    n_samples = 20
    n_features = 100

    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, 2, n_samples)

    return X, y


# ============================================================================
# Expected Values for Validation
# ============================================================================

@pytest.fixture
def expected_submission_columns():
    """Expected columns in submission file."""
    return ['ID', 'dataset', 'label_positive_probability', 'junction_aa', 'v_call', 'j_call']


@pytest.fixture
def expected_submission_row_counts():
    """Expected row counts in submission file."""
    # 4,213 predictions + 8 datasets * 50,000 sequences = 404,213
    return {
        'predictions': 4213,
        'sequences_per_dataset': 50000,
        'total_datasets': 8,
        'total_rows': 404213
    }


# ============================================================================
# Performance Benchmarks
# ============================================================================

@pytest.fixture
def performance_thresholds():
    """Performance thresholds for testing."""
    return {
        'esm2_extraction_time_per_repertoire': 60,  # seconds
        'gpu_memory_max': 10,  # GB
        'total_training_time': 4 * 3600,  # 4 hours in seconds
        'feature_extraction_time': 300,  # 5 minutes
    }


# ============================================================================
# Utility Functions
# ============================================================================

def assert_valid_probabilities(probs):
    """Assert probabilities are in valid range [0, 1]."""
    assert np.all((probs >= 0) & (probs <= 1)), "Probabilities must be in [0, 1]"


def assert_valid_submission_format(df, expected_columns):
    """Assert submission DataFrame has valid format."""
    # Check columns
    assert list(df.columns) == expected_columns, f"Invalid columns: {df.columns}"

    # Check no NaN in required columns
    assert not df['ID'].isna().any(), "ID column contains NaN"
    assert not df['dataset'].isna().any(), "dataset column contains NaN"

    # Check probabilities
    prob_col = 'label_positive_probability'
    pred_rows = df[df[prob_col] != -999.0]
    if len(pred_rows) > 0:
        assert_valid_probabilities(pred_rows[prob_col].values)
