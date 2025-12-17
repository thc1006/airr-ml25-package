#!/usr/bin/env python3
"""
Unit tests for ESM2 Feature Extractor (champion_v13).

Tests ESM2 model loading, embedding extraction, aggregation,
GPU usage, and checkpoint persistence.
"""

import pytest
import numpy as np
import torch
import tempfile
from pathlib import Path


# ============================================================================
# Test ESM2 Model Loading
# ============================================================================

@pytest.mark.unit
@pytest.mark.gpu
def test_esm2_model_loading_gpu(device, gpu_available):
    """Test ESM2 model loads correctly on GPU."""
    if not gpu_available:
        pytest.skip("GPU not available")

    # This test will fail until we implement ESM2FeatureExtractor
    from champion_v13_esm2 import ESM2FeatureExtractor

    extractor = ESM2FeatureExtractor(device='cuda')

    # Assertions
    assert extractor.device == 'cuda'
    assert extractor.model is not None
    assert next(extractor.model.parameters()).is_cuda


@pytest.mark.unit
def test_esm2_model_loading_cpu():
    """Test ESM2 model loads correctly on CPU."""
    from champion_v13_esm2 import ESM2FeatureExtractor

    extractor = ESM2FeatureExtractor(device='cpu')

    assert extractor.device == 'cpu'
    assert extractor.model is not None
    assert not next(extractor.model.parameters()).is_cuda


@pytest.mark.unit
def test_esm2_model_loading_auto_device(gpu_available):
    """Test ESM2 model automatically selects best device."""
    from champion_v13_esm2 import ESM2FeatureExtractor

    extractor = ESM2FeatureExtractor(device='auto')

    expected_device = 'cuda' if gpu_available else 'cpu'
    assert extractor.device == expected_device


# ============================================================================
# Test Single Sequence Embedding
# ============================================================================

@pytest.mark.unit
def test_extract_single_sequence(device):
    """Test single sequence embedding extraction."""
    from champion_v13_esm2 import ESM2FeatureExtractor

    extractor = ESM2FeatureExtractor(device=device)
    sequence = 'CASSLAPGATNEKLFF'

    embedding = extractor.extract_sequence_embedding(sequence)

    # ESM2 650M produces 1280-dim embeddings
    assert embedding.shape == (1280,)
    assert embedding.dtype == np.float32
    assert not np.isnan(embedding).any()
    assert not np.isinf(embedding).any()


@pytest.mark.unit
def test_extract_multiple_sequences(device, sample_sequences):
    """Test batch sequence embedding extraction."""
    from champion_v13_esm2 import ESM2FeatureExtractor

    extractor = ESM2FeatureExtractor(device=device)

    embeddings = extractor.extract_batch_embeddings(sample_sequences)

    # Check shape
    assert embeddings.shape == (len(sample_sequences), 1280)
    assert embeddings.dtype == np.float32

    # Check no invalid values
    assert not np.isnan(embeddings).any()
    assert not np.isinf(embeddings).any()


@pytest.mark.unit
def test_extract_sequence_with_invalid_characters(device):
    """Test handling of sequences with invalid amino acids."""
    from champion_v13_esm2 import ESM2FeatureExtractor

    extractor = ESM2FeatureExtractor(device=device)

    # Sequence with invalid character 'X'
    invalid_sequence = 'CASSXLAPGATNEKLFF'

    # Should handle gracefully (replace X or skip)
    embedding = extractor.extract_sequence_embedding(invalid_sequence)

    assert embedding.shape == (1280,)
    assert not np.isnan(embedding).any()


# ============================================================================
# Test Repertoire-Level Aggregation
# ============================================================================

@pytest.mark.unit
def test_extract_repertoire_features(device, sample_repertoire):
    """Test repertoire-level feature aggregation."""
    from champion_v13_esm2 import ESM2FeatureExtractor

    extractor = ESM2FeatureExtractor(device=device)

    features = extractor.extract_repertoire_features(sample_repertoire)

    # Expected shape: mean(1280) + std(1280) + min(1280) + max(1280) = 5120
    assert features.shape == (5120,)
    assert features.dtype == np.float32
    assert not np.isnan(features).any()
    assert not np.isinf(features).any()


@pytest.mark.unit
def test_repertoire_aggregation_statistics(device, sample_sequences):
    """Test that aggregation statistics are computed correctly."""
    from champion_v13_esm2 import ESM2FeatureExtractor

    extractor = ESM2FeatureExtractor(device=device)

    # Extract embeddings
    embeddings = extractor.extract_batch_embeddings(sample_sequences)

    # Compute statistics manually
    expected_mean = embeddings.mean(axis=0)
    expected_std = embeddings.std(axis=0)
    expected_min = embeddings.min(axis=0)
    expected_max = embeddings.max(axis=0)

    # Extract repertoire features
    features = extractor.aggregate_statistics(embeddings)

    # Verify statistics match
    np.testing.assert_allclose(features[:1280], expected_mean, rtol=1e-5)
    np.testing.assert_allclose(features[1280:2560], expected_std, rtol=1e-5)
    np.testing.assert_allclose(features[2560:3840], expected_min, rtol=1e-5)
    np.testing.assert_allclose(features[3840:5120], expected_max, rtol=1e-5)


@pytest.mark.unit
def test_empty_repertoire_handling(device):
    """Test handling of empty repertoire."""
    from champion_v13_esm2 import ESM2FeatureExtractor
    import pandas as pd

    extractor = ESM2FeatureExtractor(device=device)

    empty_df = pd.DataFrame({'junction_aa': []})

    # Should return zeros or raise informative error
    with pytest.raises(ValueError, match="empty"):
        extractor.extract_repertoire_features(empty_df)


# ============================================================================
# Test Batch Processing
# ============================================================================

@pytest.mark.unit
def test_batch_processing_efficiency(device, sample_sequences):
    """Test batch processing is more efficient than sequential."""
    from champion_v13_esm2 import ESM2FeatureExtractor
    import time

    extractor = ESM2FeatureExtractor(device=device, batch_size=4)

    # Batch processing
    start_batch = time.time()
    batch_result = extractor.extract_batch_embeddings(sample_sequences)
    time_batch = time.time() - start_batch

    # Sequential processing
    start_seq = time.time()
    seq_results = [extractor.extract_sequence_embedding(seq) for seq in sample_sequences]
    time_seq = time.time() - start_seq

    # Batch should be faster (or similar for small batches)
    assert time_batch <= time_seq * 1.5  # Allow 50% margin

    # Results should match
    seq_result = np.stack(seq_results)
    np.testing.assert_allclose(batch_result, seq_result, rtol=1e-4)


@pytest.mark.unit
def test_batch_size_configuration(device):
    """Test different batch sizes work correctly."""
    from champion_v13_esm2 import ESM2FeatureExtractor

    for batch_size in [1, 4, 8, 16]:
        extractor = ESM2FeatureExtractor(device=device, batch_size=batch_size)
        assert extractor.batch_size == batch_size

        # Test extraction works
        sequences = ['CASSF'] * 10
        embeddings = extractor.extract_batch_embeddings(sequences)
        assert embeddings.shape == (10, 1280)


# ============================================================================
# Test GPU Memory Management
# ============================================================================

@pytest.mark.unit
@pytest.mark.gpu
@pytest.mark.slow
def test_gpu_memory_usage(gpu_available, performance_thresholds):
    """Test GPU memory stays under limit."""
    if not gpu_available:
        pytest.skip("GPU not available")

    from champion_v13_esm2 import ESM2FeatureExtractor

    # Monitor GPU memory
    torch.cuda.reset_peak_memory_stats()

    extractor = ESM2FeatureExtractor(device='cuda')

    # Process large batch
    large_batch = ['CASSLAPGATNEKLFF'] * 1000
    embeddings = extractor.extract_batch_embeddings(large_batch)

    # Check memory usage
    max_memory_gb = torch.cuda.max_memory_allocated() / 1e9

    assert max_memory_gb < performance_thresholds['gpu_memory_max'], \
        f"GPU memory usage {max_memory_gb:.2f} GB exceeds threshold"

    # Verify output
    assert embeddings.shape == (1000, 1280)


@pytest.mark.unit
@pytest.mark.gpu
def test_gpu_memory_cleanup(gpu_available):
    """Test GPU memory is properly cleaned up."""
    if not gpu_available:
        pytest.skip("GPU not available")

    from champion_v13_esm2 import ESM2FeatureExtractor

    torch.cuda.reset_peak_memory_stats()
    initial_memory = torch.cuda.memory_allocated()

    # Create and delete extractor
    extractor = ESM2FeatureExtractor(device='cuda')
    sequences = ['CASSF'] * 100
    _ = extractor.extract_batch_embeddings(sequences)

    # Force cleanup
    del extractor
    torch.cuda.empty_cache()

    final_memory = torch.cuda.memory_allocated()

    # Memory should be close to initial (within 10%)
    memory_increase = (final_memory - initial_memory) / initial_memory
    assert memory_increase < 0.1, f"Memory leak detected: {memory_increase:.2%} increase"


# ============================================================================
# Test Checkpoint Save/Load
# ============================================================================

@pytest.mark.unit
def test_checkpoint_save_load(device, sample_sequences):
    """Test checkpoint persistence."""
    from champion_v13_esm2 import ESM2FeatureExtractor

    extractor = ESM2FeatureExtractor(device=device)

    # Extract features
    original_embeddings = extractor.extract_batch_embeddings(sample_sequences)

    # Save checkpoint
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = Path(tmpdir) / "esm2_checkpoint.pkl"
        extractor.save_checkpoint(checkpoint_path)

        # Load checkpoint
        loaded_extractor = ESM2FeatureExtractor(device=device)
        loaded_extractor.load_checkpoint(checkpoint_path)

        # Extract features again
        loaded_embeddings = loaded_extractor.extract_batch_embeddings(sample_sequences)

        # Results should match
        np.testing.assert_allclose(original_embeddings, loaded_embeddings, rtol=1e-5)


@pytest.mark.unit
def test_checkpoint_includes_config(device):
    """Test checkpoint includes configuration."""
    from champion_v13_esm2 import ESM2FeatureExtractor

    extractor = ESM2FeatureExtractor(device=device, batch_size=8)

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = Path(tmpdir) / "config_checkpoint.pkl"
        extractor.save_checkpoint(checkpoint_path)

        # Load and verify config
        loaded_extractor = ESM2FeatureExtractor.load_from_checkpoint(checkpoint_path)

        assert loaded_extractor.batch_size == 8
        assert loaded_extractor.device == device


# ============================================================================
# Test Error Handling
# ============================================================================

@pytest.mark.unit
def test_invalid_device_raises_error():
    """Test invalid device raises informative error."""
    from champion_v13_esm2 import ESM2FeatureExtractor

    with pytest.raises(ValueError, match="device"):
        ESM2FeatureExtractor(device='invalid_device')


@pytest.mark.unit
def test_extraction_with_none_sequence(device):
    """Test handling of None sequences."""
    from champion_v13_esm2 import ESM2FeatureExtractor

    extractor = ESM2FeatureExtractor(device=device)

    sequences = ['CASSF', None, 'CASSG']

    # Should skip None values or raise informative error
    with pytest.raises(ValueError, match="None|invalid"):
        extractor.extract_batch_embeddings(sequences)


@pytest.mark.unit
def test_extraction_with_empty_string(device):
    """Test handling of empty string sequences."""
    from champion_v13_esm2 import ESM2FeatureExtractor

    extractor = ESM2FeatureExtractor(device=device)

    sequences = ['CASSF', '', 'CASSG']

    # Should skip empty strings or raise informative error
    with pytest.raises(ValueError, match="empty"):
        extractor.extract_batch_embeddings(sequences)


# ============================================================================
# Test Deterministic Behavior
# ============================================================================

@pytest.mark.unit
def test_deterministic_extraction(device):
    """Test that extraction is deterministic."""
    from champion_v13_esm2 import ESM2FeatureExtractor

    sequence = 'CASSLAPGATNEKLFF'

    # Extract twice
    extractor1 = ESM2FeatureExtractor(device=device)
    embedding1 = extractor1.extract_sequence_embedding(sequence)

    extractor2 = ESM2FeatureExtractor(device=device)
    embedding2 = extractor2.extract_sequence_embedding(sequence)

    # Should be identical
    np.testing.assert_array_equal(embedding1, embedding2)


@pytest.mark.unit
def test_reproducibility_with_seed(device):
    """Test reproducibility when using random seed (if applicable)."""
    from champion_v13_esm2 import ESM2FeatureExtractor

    sequence = 'CASSLAPGATNEKLFF'

    # Set seed and extract
    np.random.seed(42)
    torch.manual_seed(42)
    extractor1 = ESM2FeatureExtractor(device=device)
    embedding1 = extractor1.extract_sequence_embedding(sequence)

    # Reset seed and extract again
    np.random.seed(42)
    torch.manual_seed(42)
    extractor2 = ESM2FeatureExtractor(device=device)
    embedding2 = extractor2.extract_sequence_embedding(sequence)

    # Should be identical
    np.testing.assert_array_equal(embedding1, embedding2)
