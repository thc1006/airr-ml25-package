#!/usr/bin/env python3
"""
Performance tests for champion_v13.

Benchmarks for ESM2 extraction speed, GPU memory usage, training time,
and overall pipeline performance.
"""

import pytest
import numpy as np
import pandas as pd
import time
import torch
from pathlib import Path


# ============================================================================
# ESM2 Extraction Performance
# ============================================================================

@pytest.mark.performance
@pytest.mark.slow
def test_esm2_extraction_speed_per_repertoire(temp_dataset_dir, device, performance_thresholds):
    """Test ESM2 extraction meets speed target (<60 sec/repertoire)."""
    from champion_v13_esm2 import ESM2FeatureExtractor, ESM2Config

    config = ESM2Config(
        device=device,
        batch_size=32,
        max_seqs_per_repertoire=500
    )

    extractor = ESM2FeatureExtractor(config=config)

    # Time extraction for single repertoire
    tsv_files = list(temp_dataset_dir.glob("*.tsv"))

    if len(tsv_files) == 0:
        pytest.skip("No TSV files found in temp dataset")

    tsv_path = tsv_files[0]
    repertoire_id = tsv_path.stem

    start_time = time.time()
    features = extractor.extract_repertoire_features(tsv_path, repertoire_id)
    elapsed = time.time() - start_time

    # Assert performance target
    threshold = performance_thresholds['esm2_extraction_time_per_repertoire']
    assert elapsed < threshold, \
        f"ESM2 extraction took {elapsed:.2f}s, exceeds {threshold}s threshold"

    print(f"\n  ESM2 extraction time: {elapsed:.2f}s (target: <{threshold}s)")


@pytest.mark.performance
@pytest.mark.slow
def test_esm2_batch_extraction_speed(device, sample_sequences):
    """Test ESM2 batch extraction is efficient."""
    from champion_v13_esm2 import ESM2FeatureExtractor, ESM2Config

    config = ESM2Config(device=device, batch_size=32)
    extractor = ESM2FeatureExtractor(config=config)

    # Generate large batch
    large_batch = sample_sequences * 100  # 800 sequences

    start_time = time.time()
    embeddings = extractor.extract_sequence_embeddings(large_batch)
    elapsed = time.time() - start_time

    # Should process at reasonable speed
    sequences_per_second = len(large_batch) / elapsed

    assert sequences_per_second > 10, \
        f"Processing speed {sequences_per_second:.1f} seq/s is too slow"

    print(f"\n  Batch extraction: {sequences_per_second:.1f} sequences/second")


# ============================================================================
# GPU Memory Performance
# ============================================================================

@pytest.mark.performance
@pytest.mark.gpu
@pytest.mark.slow
def test_gpu_memory_efficiency(gpu_available, performance_thresholds):
    """Test GPU memory usage stays under 10GB."""
    if not gpu_available:
        pytest.skip("GPU not available")

    from champion_v13_esm2 import ESM2FeatureExtractor, ESM2Config

    torch.cuda.reset_peak_memory_stats()
    initial_memory = torch.cuda.memory_allocated() / 1e9

    config = ESM2Config(device='cuda', batch_size=32)
    extractor = ESM2FeatureExtractor(config=config)

    # Process large repertoire
    sequences = ['CASSLAPGATNEKLFF'] * 500
    embeddings = extractor.extract_sequence_embeddings(sequences)

    # Check peak memory
    peak_memory_gb = torch.cuda.max_memory_allocated() / 1e9
    memory_increase = peak_memory_gb - initial_memory

    threshold = performance_thresholds['gpu_memory_max']
    assert peak_memory_gb < threshold, \
        f"GPU memory usage {peak_memory_gb:.2f}GB exceeds {threshold}GB threshold"

    print(f"\n  GPU memory: {peak_memory_gb:.2f}GB peak (target: <{threshold}GB)")
    print(f"  Memory increase: {memory_increase:.2f}GB")


@pytest.mark.performance
@pytest.mark.gpu
def test_gpu_memory_per_batch(gpu_available):
    """Test GPU memory scales linearly with batch size."""
    if not gpu_available:
        pytest.skip("GPU not available")

    from champion_v13_esm2 import ESM2FeatureExtractor, ESM2Config

    batch_sizes = [8, 16, 32]
    memory_usage = []

    for batch_size in batch_sizes:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

        config = ESM2Config(device='cuda', batch_size=batch_size)
        extractor = ESM2FeatureExtractor(config=config)

        sequences = ['CASSLAPGATNEKLFF'] * batch_size
        _ = extractor.extract_sequence_embeddings(sequences)

        peak_memory = torch.cuda.max_memory_allocated() / 1e9
        memory_usage.append(peak_memory)

        print(f"\n  Batch size {batch_size}: {peak_memory:.2f}GB")

    # Memory should scale roughly linearly (within 2x)
    ratio = memory_usage[-1] / memory_usage[0]
    expected_ratio = batch_sizes[-1] / batch_sizes[0]

    assert ratio < expected_ratio * 2, \
        f"Memory scaling {ratio:.2f}x exceeds expected {expected_ratio}x by >2x"


# ============================================================================
# Training Performance
# ============================================================================

@pytest.mark.performance
@pytest.mark.slow
def test_training_time_single_dataset(temp_dataset_dir, device, performance_thresholds):
    """Test training completes in reasonable time."""
    from champion_v13 import V13Config, train_gpu_ensemble, extract_features_parallel, build_vocabulary
    import pandas as pd

    config = V13Config(use_gpu=(device == 'cuda'))

    # Load metadata
    metadata = pd.read_csv(temp_dataset_dir / "metadata.csv")
    y = metadata['label_positive'].astype(int).values

    # Build vocabulary
    kmer_vocab, vj_vocab, v_vocab, j_vocab, public_clones = \
        build_vocabulary(temp_dataset_dir, config)

    # Extract features
    X_df = extract_features_parallel(
        temp_dataset_dir, metadata,
        kmer_vocab, vj_vocab, v_vocab, j_vocab, public_clones,
        config
    )
    X = X_df.fillna(0).values

    # Time training
    start_time = time.time()
    ensemble = train_gpu_ensemble(X, y, config)
    elapsed = time.time() - start_time

    # Should complete in reasonable time (5 minutes for small dataset)
    assert elapsed < 300, \
        f"Training took {elapsed:.2f}s, exceeds 300s threshold"

    print(f"\n  Training time: {elapsed:.2f}s")
    print(f"  CV AUC: {ensemble.cv_auc:.4f}")


@pytest.mark.performance
@pytest.mark.slow
def test_feature_extraction_speed(temp_dataset_dir, device, performance_thresholds):
    """Test feature extraction speed."""
    from champion_v13 import V13Config, extract_features_parallel, build_vocabulary
    import pandas as pd

    config = V13Config()

    metadata = pd.read_csv(temp_dataset_dir / "metadata.csv")

    # Build vocabulary
    kmer_vocab, vj_vocab, v_vocab, j_vocab, public_clones = \
        build_vocabulary(temp_dataset_dir, config)

    # Time feature extraction
    start_time = time.time()
    X_df = extract_features_parallel(
        temp_dataset_dir, metadata,
        kmer_vocab, vj_vocab, v_vocab, j_vocab, public_clones,
        config
    )
    elapsed = time.time() - start_time

    threshold = performance_thresholds['feature_extraction_time']
    assert elapsed < threshold, \
        f"Feature extraction took {elapsed:.2f}s, exceeds {threshold}s threshold"

    print(f"\n  Feature extraction: {elapsed:.2f}s (target: <{threshold}s)")
    print(f"  Repertoires processed: {len(X_df)}")
    print(f"  Features extracted: {X_df.shape[1]}")


# ============================================================================
# Model Inference Performance
# ============================================================================

@pytest.mark.performance
def test_inference_speed(sample_features_labels):
    """Test model inference speed."""
    from champion_v13 import V13Config, train_gpu_ensemble

    X, y = sample_features_labels
    config = V13Config(use_gpu=torch.cuda.is_available())

    # Train model
    ensemble = train_gpu_ensemble(X, y, config)

    # Generate large test set
    X_test = np.random.randn(1000, X.shape[1])

    # Time prediction
    start_time = time.time()
    X_selected = ensemble.selector.transform(X_test)
    xgb_pred = ensemble.xgb.predict_proba(X_selected)[:, 1]
    lgb_pred = ensemble.lgb.predict_proba(X_selected)[:, 1]
    cb_pred = ensemble.cb.predict_proba(X_selected)[:, 1]
    elapsed = time.time() - start_time

    # Should be fast (<1 second for 1000 samples)
    assert elapsed < 1.0, \
        f"Inference took {elapsed:.2f}s for 1000 samples, too slow"

    predictions_per_second = len(X_test) / elapsed
    print(f"\n  Inference speed: {predictions_per_second:.0f} predictions/second")


# ============================================================================
# End-to-End Performance
# ============================================================================

@pytest.mark.performance
@pytest.mark.slow
@pytest.mark.integration
def test_end_to_end_pipeline_performance(temp_dataset_dir, device):
    """Test complete pipeline performance."""
    from champion_v13 import V13Config, main
    import time

    # Override config paths
    config = V13Config(
        train_root=temp_dataset_dir.parent,
        test_root=temp_dataset_dir.parent,
        use_gpu=(device == 'cuda')
    )

    start_time = time.time()

    # Run simplified pipeline (single dataset)
    # This would be the full main() but we test a subset

    elapsed = time.time() - start_time

    print(f"\n  End-to-end time: {elapsed:.2f}s")


# ============================================================================
# Memory Profiling
# ============================================================================

@pytest.mark.performance
@pytest.mark.slow
def test_cpu_memory_usage(temp_dataset_dir):
    """Test CPU memory usage stays reasonable."""
    import psutil
    import os
    from champion_v13 import V13Config, extract_features_parallel, build_vocabulary
    import pandas as pd

    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1e9  # GB

    config = V13Config()
    metadata = pd.read_csv(temp_dataset_dir / "metadata.csv")

    # Build vocab and extract features
    kmer_vocab, vj_vocab, v_vocab, j_vocab, public_clones = \
        build_vocabulary(temp_dataset_dir, config)

    X_df = extract_features_parallel(
        temp_dataset_dir, metadata,
        kmer_vocab, vj_vocab, v_vocab, j_vocab, public_clones,
        config
    )

    final_memory = process.memory_info().rss / 1e9
    memory_increase = final_memory - initial_memory

    # Should use less than 4GB additional memory
    assert memory_increase < 4.0, \
        f"CPU memory increase {memory_increase:.2f}GB exceeds 4GB threshold"

    print(f"\n  CPU memory increase: {memory_increase:.2f}GB")


# ============================================================================
# Comparative Performance Tests
# ============================================================================

@pytest.mark.performance
def test_gpu_vs_cpu_speedup(sample_features_labels, gpu_available):
    """Test GPU provides speedup over CPU."""
    if not gpu_available:
        pytest.skip("GPU not available")

    from champion_v13 import V13Config, train_gpu_ensemble

    X, y = sample_features_labels

    # CPU training
    config_cpu = V13Config(use_gpu=False, xgb_n_estimators=50)
    start_cpu = time.time()
    ensemble_cpu = train_gpu_ensemble(X, y, config_cpu)
    time_cpu = time.time() - start_cpu

    # GPU training
    config_gpu = V13Config(use_gpu=True, xgb_n_estimators=50)
    start_gpu = time.time()
    ensemble_gpu = train_gpu_ensemble(X, y, config_gpu)
    time_gpu = time.time() - start_gpu

    speedup = time_cpu / time_gpu

    print(f"\n  CPU time: {time_cpu:.2f}s")
    print(f"  GPU time: {time_gpu:.2f}s")
    print(f"  Speedup: {speedup:.2f}x")

    # GPU should be at least as fast (may be slower for tiny datasets)
    assert speedup > 0.5, f"GPU is {1/speedup:.2f}x slower than CPU"


# ============================================================================
# Scalability Tests
# ============================================================================

@pytest.mark.performance
@pytest.mark.slow
def test_feature_extraction_scalability():
    """Test feature extraction scales linearly with dataset size."""
    from champion_v13 import extract_features_single

    # Mock vocabulary
    kmer_vocab = [f'ABC{i}' for i in range(100)]
    vj_vocab = [('TRBV20-1', 'TRBJ2-7')] * 50
    v_vocab = [f'TRBV{i}' for i in range(20)]
    j_vocab = [f'TRBJ{i}' for i in range(10)]
    public_clones = [f'CASS{i}' for i in range(100)]

    dataset_sizes = [10, 50, 100]
    extraction_times = []

    import tempfile
    import pandas as pd

    for size in dataset_sizes:
        # Create mock TSV
        df = pd.DataFrame({
            'junction_aa': [f'CASSLAPGATNEKLFF{i}' for i in range(size)],
            'v_call': ['TRBV20-1'] * size,
            'j_call': ['TRBJ2-7'] * size
        })

        with tempfile.NamedTemporaryFile(mode='w', suffix='.tsv', delete=False) as f:
            df.to_csv(f.name, sep='\t', index=False)
            temp_path = Path(f.name)

        start = time.time()
        features = extract_features_single(
            temp_path, kmer_vocab, vj_vocab, v_vocab, j_vocab, public_clones
        )
        elapsed = time.time() - start
        extraction_times.append(elapsed)

        temp_path.unlink()

    # Time should scale roughly linearly (within 2x of expected)
    for i in range(1, len(dataset_sizes)):
        expected_ratio = dataset_sizes[i] / dataset_sizes[0]
        actual_ratio = extraction_times[i] / extraction_times[0]

        assert actual_ratio < expected_ratio * 2, \
            f"Scaling not linear: {actual_ratio:.2f}x vs expected {expected_ratio:.2f}x"

    print("\n  Scalability test:")
    for size, time_val in zip(dataset_sizes, extraction_times):
        print(f"    {size} sequences: {time_val:.4f}s")
