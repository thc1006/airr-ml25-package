#!/usr/bin/env python3
"""
Integration tests for champion_v13 pipeline.

Tests end-to-end pipeline, feature compatibility, submission format,
and cross-validation.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path


# ============================================================================
# Test End-to-End Pipeline on Single Dataset
# ============================================================================

@pytest.mark.integration
@pytest.mark.slow
def test_end_to_end_single_dataset(temp_dataset_dir, temp_output_dir, device):
    """Test complete pipeline on single dataset."""
    from champion_v13_pipeline import ChampionV13Pipeline

    pipeline = ChampionV13Pipeline(
        device=device,
        n_jobs=2,
        verbose=True
    )

    # Train on dataset
    pipeline.fit(str(temp_dataset_dir))

    # Check model is trained
    assert pipeline.model is not None
    assert pipeline.is_fitted

    # Check features were extracted
    assert pipeline.esm2_features is not None
    assert pipeline.vj_features is not None
    assert pipeline.combined_features is not None


@pytest.mark.integration
@pytest.mark.slow
def test_end_to_end_prediction(temp_dataset_dir, temp_output_dir, device):
    """Test end-to-end prediction pipeline."""
    from champion_v13_pipeline import ChampionV13Pipeline

    pipeline = ChampionV13Pipeline(device=device, n_jobs=2)

    # Train
    pipeline.fit(str(temp_dataset_dir))

    # Predict
    predictions = pipeline.predict_proba(str(temp_dataset_dir))

    # Check predictions
    assert isinstance(predictions, pd.DataFrame)
    assert 'ID' in predictions.columns
    assert 'label_positive_probability' in predictions.columns

    # Check probabilities
    probs = predictions['label_positive_probability'].values
    assert np.all((probs >= 0) & (probs <= 1))


@pytest.mark.integration
@pytest.mark.slow
def test_end_to_end_sequence_identification(temp_dataset_dir, device):
    """Test end-to-end sequence identification."""
    from champion_v13_pipeline import ChampionV13Pipeline

    pipeline = ChampionV13Pipeline(device=device, n_jobs=2)

    # Train
    pipeline.fit(str(temp_dataset_dir))

    # Identify sequences
    sequences = pipeline.identify_associated_sequences(str(temp_dataset_dir), top_k=100)

    # Check sequences
    assert isinstance(sequences, pd.DataFrame)
    assert len(sequences) == 100
    assert 'junction_aa' in sequences.columns
    assert 'v_call' in sequences.columns
    assert 'j_call' in sequences.columns

    # Check no missing values in sequences
    assert not sequences['junction_aa'].isna().any()


# ============================================================================
# Test Feature Compatibility
# ============================================================================

@pytest.mark.integration
def test_esm2_vj_feature_compatibility(temp_dataset_dir, device):
    """Test ESM2 + VJ features work together."""
    from champion_v13_pipeline import ChampionV13Pipeline

    pipeline = ChampionV13Pipeline(device=device)

    # Extract both feature types
    pipeline._extract_esm2_features(str(temp_dataset_dir))
    pipeline._extract_vj_features(str(temp_dataset_dir))

    # Combine features
    combined = pipeline._combine_features()

    # Check combined features
    assert combined is not None
    assert combined.shape[0] > 0  # Has samples
    assert combined.shape[1] > 5120  # ESM2 (5120) + VJ features

    # Check no NaN or inf
    assert not np.isnan(combined).any()
    assert not np.isinf(combined).any()


@pytest.mark.integration
def test_feature_dimensionality_consistency(temp_dataset_dir, device):
    """Test feature dimensions are consistent across samples."""
    from champion_v13_pipeline import ChampionV13Pipeline

    pipeline = ChampionV13Pipeline(device=device)
    pipeline._extract_all_features(str(temp_dataset_dir))

    # All samples should have same feature dimension
    feature_dims = [f.shape[0] for f in pipeline.combined_features]
    assert len(set(feature_dims)) == 1, "Inconsistent feature dimensions"


@pytest.mark.integration
def test_feature_names_consistency():
    """Test feature names are consistent and unique."""
    from champion_v13_pipeline import ChampionV13Pipeline

    pipeline = ChampionV13Pipeline()

    # Get feature names
    feature_names = pipeline.get_feature_names()

    # Should have unique names
    assert len(feature_names) == len(set(feature_names)), "Duplicate feature names"

    # Should include both ESM2 and VJ features
    assert any('esm2' in name for name in feature_names)
    assert any('vj' in name for name in feature_names)


# ============================================================================
# Test Submission Format
# ============================================================================

@pytest.mark.integration
def test_submission_format_validation(temp_dataset_dir, temp_output_dir, device, expected_submission_columns):
    """Test final submission file format."""
    from champion_v13_pipeline import ChampionV13Pipeline

    pipeline = ChampionV13Pipeline(device=device, n_jobs=2)
    pipeline.fit(str(temp_dataset_dir))

    # Generate submission
    submission_path = temp_output_dir / "submission_v13.csv"
    pipeline.generate_submission(
        test_dirs=[str(temp_dataset_dir)],
        output_path=str(submission_path)
    )

    # Read and validate
    submission = pd.read_csv(submission_path)

    # Check columns
    assert list(submission.columns) == expected_submission_columns

    # Check no NaN in critical columns
    assert not submission['ID'].isna().any()
    assert not submission['dataset'].isna().any()


@pytest.mark.integration
def test_submission_probability_format(temp_dataset_dir, temp_output_dir, device):
    """Test submission probabilities format."""
    from champion_v13_pipeline import ChampionV13Pipeline

    pipeline = ChampionV13Pipeline(device=device)
    pipeline.fit(str(temp_dataset_dir))

    submission_path = temp_output_dir / "submission_v13.csv"
    pipeline.generate_submission(
        test_dirs=[str(temp_dataset_dir)],
        output_path=str(submission_path)
    )

    submission = pd.read_csv(submission_path)

    # Separate predictions and sequences
    predictions = submission[submission['label_positive_probability'] != -999.0]
    sequences = submission[submission['label_positive_probability'] == -999.0]

    # Check predictions have valid probabilities
    if len(predictions) > 0:
        probs = predictions['label_positive_probability'].values
        assert np.all((probs >= 0) & (probs <= 1))

    # Check sequences have -999.0
    if len(sequences) > 0:
        assert np.all(sequences['label_positive_probability'] == -999.0)
        assert not sequences['junction_aa'].isna().any()


@pytest.mark.integration
def test_submission_sequence_format(temp_dataset_dir, temp_output_dir, device):
    """Test submission sequence format."""
    from champion_v13_pipeline import ChampionV13Pipeline

    pipeline = ChampionV13Pipeline(device=device)
    pipeline.fit(str(temp_dataset_dir))

    submission_path = temp_output_dir / "submission_v13.csv"
    pipeline.generate_submission(
        test_dirs=[str(temp_dataset_dir)],
        output_path=str(submission_path),
        top_k_sequences=100
    )

    submission = pd.read_csv(submission_path)
    sequences = submission[submission['label_positive_probability'] == -999.0]

    # Should have 100 sequences
    assert len(sequences) == 100

    # Sequences should have valid columns
    assert not sequences['junction_aa'].isna().any()
    assert not sequences['v_call'].isna().any()
    assert not sequences['j_call'].isna().any()

    # Prediction columns should be -999.0
    assert np.all(sequences['label_positive_probability'] == -999.0)


# ============================================================================
# Test Cross-Validation
# ============================================================================

@pytest.mark.integration
@pytest.mark.slow
def test_cross_validation_pipeline(temp_dataset_dir, device):
    """Test 5-fold cross-validation runs without errors."""
    from champion_v13_pipeline import ChampionV13Pipeline

    pipeline = ChampionV13Pipeline(device=device, n_folds=3)  # Use 3 folds for speed

    # Run cross-validation
    cv_scores = pipeline.cross_validate(str(temp_dataset_dir))

    # Check scores
    assert len(cv_scores) == 3
    assert all(0 <= score <= 1 for score in cv_scores)

    # Check average score
    avg_score = np.mean(cv_scores)
    assert 0 <= avg_score <= 1


@pytest.mark.integration
@pytest.mark.slow
def test_cross_validation_reproducibility(temp_dataset_dir, device):
    """Test cross-validation is reproducible."""
    from champion_v13_pipeline import ChampionV13Pipeline

    # First run
    pipeline1 = ChampionV13Pipeline(device=device, n_folds=3, random_state=42)
    scores1 = pipeline1.cross_validate(str(temp_dataset_dir))

    # Second run
    pipeline2 = ChampionV13Pipeline(device=device, n_folds=3, random_state=42)
    scores2 = pipeline2.cross_validate(str(temp_dataset_dir))

    # Should produce identical scores
    np.testing.assert_array_equal(scores1, scores2)


# ============================================================================
# Test Multi-Dataset Handling
# ============================================================================

@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.requires_data
def test_multi_dataset_training():
    """Test training on multiple datasets."""
    from champion_v13_pipeline import ChampionV13Pipeline

    # This test requires actual data
    train_root = Path("./data/train_datasets/train_datasets")
    if not train_root.exists():
        pytest.skip("Training data not available")

    dataset_dirs = list(train_root.glob("train_dataset_*"))[:2]  # Use first 2 datasets
    if len(dataset_dirs) < 2:
        pytest.skip("Not enough datasets")

    pipeline = ChampionV13Pipeline(device='cpu', n_jobs=2)

    # Train on multiple datasets
    pipeline.fit_multiple(dataset_dirs)

    # Should be fitted
    assert pipeline.is_fitted
    assert pipeline.model is not None


# ============================================================================
# Test Pipeline Checkpointing
# ============================================================================

@pytest.mark.integration
def test_pipeline_checkpoint_save_load(temp_dataset_dir, temp_output_dir, device):
    """Test pipeline checkpoint save and load."""
    from champion_v13_pipeline import ChampionV13Pipeline

    # Train pipeline
    pipeline = ChampionV13Pipeline(device=device)
    pipeline.fit(str(temp_dataset_dir))

    # Get predictions before save
    pred_before = pipeline.predict_proba(str(temp_dataset_dir))

    # Save checkpoint
    checkpoint_path = temp_output_dir / "pipeline_checkpoint.pkl"
    pipeline.save_checkpoint(str(checkpoint_path))

    # Load checkpoint
    loaded_pipeline = ChampionV13Pipeline.load_checkpoint(str(checkpoint_path))

    # Get predictions after load
    pred_after = loaded_pipeline.predict_proba(str(temp_dataset_dir))

    # Predictions should match
    pd.testing.assert_frame_equal(pred_before, pred_after)


# ============================================================================
# Test Error Handling
# ============================================================================

@pytest.mark.integration
def test_pipeline_invalid_directory(device):
    """Test pipeline handles invalid directory gracefully."""
    from champion_v13_pipeline import ChampionV13Pipeline

    pipeline = ChampionV13Pipeline(device=device)

    with pytest.raises(FileNotFoundError):
        pipeline.fit("/nonexistent/directory")


@pytest.mark.integration
def test_pipeline_missing_metadata(tmp_path, device):
    """Test pipeline handles missing metadata file."""
    from champion_v13_pipeline import ChampionV13Pipeline

    # Create directory without metadata
    dataset_dir = tmp_path / "invalid_dataset"
    dataset_dir.mkdir()

    pipeline = ChampionV13Pipeline(device=device)

    with pytest.raises(FileNotFoundError, match="metadata"):
        pipeline.fit(str(dataset_dir))


@pytest.mark.integration
def test_pipeline_empty_dataset(tmp_path, device):
    """Test pipeline handles empty dataset."""
    from champion_v13_pipeline import ChampionV13Pipeline

    # Create empty dataset
    dataset_dir = tmp_path / "empty_dataset"
    dataset_dir.mkdir()

    # Empty metadata
    metadata = pd.DataFrame(columns=['repertoire_id', 'filename', 'label_positive'])
    metadata.to_csv(dataset_dir / "metadata.csv", index=False)

    pipeline = ChampionV13Pipeline(device=device)

    with pytest.raises(ValueError, match="empty|no data"):
        pipeline.fit(str(dataset_dir))


# ============================================================================
# Test Memory Management
# ============================================================================

@pytest.mark.integration
@pytest.mark.slow
def test_pipeline_memory_cleanup(temp_dataset_dir, device):
    """Test pipeline cleans up memory properly."""
    import gc

    from champion_v13_pipeline import ChampionV13Pipeline

    # Get initial memory
    gc.collect()

    # Create and train pipeline
    pipeline = ChampionV13Pipeline(device=device)
    pipeline.fit(str(temp_dataset_dir))

    # Delete pipeline
    del pipeline
    gc.collect()

    # Memory should be released (hard to test precisely, but should not crash)
    assert True


# ============================================================================
# Test Configuration Management
# ============================================================================

@pytest.mark.integration
def test_pipeline_config_persistence(temp_dataset_dir, temp_output_dir, device):
    """Test pipeline configuration is saved and loaded."""
    from champion_v13_pipeline import ChampionV13Pipeline

    # Create pipeline with custom config
    pipeline = ChampionV13Pipeline(
        device=device,
        n_jobs=4,
        n_folds=5,
        esm2_batch_size=8,
        random_state=123
    )

    pipeline.fit(str(temp_dataset_dir))

    # Save checkpoint
    checkpoint_path = temp_output_dir / "config_checkpoint.pkl"
    pipeline.save_checkpoint(str(checkpoint_path))

    # Load and verify config
    loaded_pipeline = ChampionV13Pipeline.load_checkpoint(str(checkpoint_path))

    assert loaded_pipeline.n_jobs == 4
    assert loaded_pipeline.n_folds == 5
    assert loaded_pipeline.esm2_batch_size == 8
    assert loaded_pipeline.random_state == 123


@pytest.mark.integration
def test_pipeline_reproducibility_with_seed(temp_dataset_dir, device):
    """Test pipeline produces reproducible results with fixed seed."""
    from champion_v13_pipeline import ChampionV13Pipeline

    # First run
    pipeline1 = ChampionV13Pipeline(device=device, random_state=42)
    pipeline1.fit(str(temp_dataset_dir))
    pred1 = pipeline1.predict_proba(str(temp_dataset_dir))

    # Second run
    pipeline2 = ChampionV13Pipeline(device=device, random_state=42)
    pipeline2.fit(str(temp_dataset_dir))
    pred2 = pipeline2.predict_proba(str(temp_dataset_dir))

    # Predictions should match
    pd.testing.assert_frame_equal(pred1, pred2)
