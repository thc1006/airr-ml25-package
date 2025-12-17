#!/usr/bin/env python3
"""
Unit tests for CatBoost Integration (champion_v13).

Tests CatBoost training, GPU acceleration, prediction,
ensemble weighting, and cross-validation.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


# ============================================================================
# Test CatBoost Model Training
# ============================================================================

@pytest.mark.unit
def test_catboost_training(sample_features_labels, catboost_params):
    """Test CatBoost model training."""
    from champion_v13_catboost import CatBoostClassifier

    X, y = sample_features_labels

    model = CatBoostClassifier(**catboost_params)
    model.fit(X, y)

    # Model should be trained
    assert model.is_fitted()
    assert model.tree_count_ > 0


@pytest.mark.unit
def test_catboost_with_validation_set(sample_features_labels, catboost_params):
    """Test CatBoost training with validation set."""
    from champion_v13_catboost import CatBoostClassifier

    X, y = sample_features_labels
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = CatBoostClassifier(**catboost_params)
    model.fit(X_train, y_train, eval_set=(X_val, y_val))

    # Should track validation metrics
    assert hasattr(model, 'evals_result_')
    assert len(model.evals_result_) > 0


@pytest.mark.unit
def test_catboost_early_stopping(sample_features_labels):
    """Test CatBoost early stopping."""
    from champion_v13_catboost import CatBoostClassifier

    X, y = sample_features_labels
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    params = {
        'iterations': 1000,
        'learning_rate': 0.1,
        'early_stopping_rounds': 10,
        'verbose': False,
        'random_seed': 42
    }

    model = CatBoostClassifier(**params)
    model.fit(X_train, y_train, eval_set=(X_val, y_val))

    # Should stop before max iterations
    assert model.tree_count_ < 1000


# ============================================================================
# Test GPU Acceleration
# ============================================================================

@pytest.mark.unit
@pytest.mark.gpu
def test_catboost_gpu_usage(sample_features_labels, gpu_available):
    """Test CatBoost GPU acceleration works."""
    if not gpu_available:
        pytest.skip("GPU not available")

    from champion_v13_catboost import CatBoostClassifier

    X, y = sample_features_labels

    params = {
        'iterations': 10,
        'task_type': 'GPU',
        'devices': '0',
        'verbose': False,
        'random_seed': 42
    }

    model = CatBoostClassifier(**params)
    model.fit(X, y)

    # Should train successfully on GPU
    assert model.is_fitted()


@pytest.mark.unit
@pytest.mark.gpu
@pytest.mark.slow
def test_catboost_gpu_vs_cpu(sample_features_labels, gpu_available):
    """Test GPU training produces similar results to CPU."""
    if not gpu_available:
        pytest.skip("GPU not available")

    from champion_v13_catboost import CatBoostClassifier

    X, y = sample_features_labels
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train on CPU
    model_cpu = CatBoostClassifier(
        iterations=50,
        task_type='CPU',
        verbose=False,
        random_seed=42
    )
    model_cpu.fit(X_train, y_train)
    pred_cpu = model_cpu.predict_proba(X_test)[:, 1]

    # Train on GPU
    model_gpu = CatBoostClassifier(
        iterations=50,
        task_type='GPU',
        devices='0',
        verbose=False,
        random_seed=42
    )
    model_gpu.fit(X_train, y_train)
    pred_gpu = model_gpu.predict_proba(X_test)[:, 1]

    # Predictions should be very similar (allow small numerical differences)
    np.testing.assert_allclose(pred_cpu, pred_gpu, rtol=0.01, atol=0.01)


# ============================================================================
# Test Prediction
# ============================================================================

@pytest.mark.unit
def test_catboost_prediction(sample_features_labels, catboost_params):
    """Test CatBoost prediction output shape."""
    from champion_v13_catboost import CatBoostClassifier

    X, y = sample_features_labels
    X_train, X_test = train_test_split(X, test_size=0.3, random_state=42)

    model = CatBoostClassifier(**catboost_params)
    model.fit(X, y)

    predictions = model.predict_proba(X_test)

    # Check shape
    assert predictions.shape == (len(X_test), 2)  # Binary classification

    # Check probabilities sum to 1
    np.testing.assert_allclose(predictions.sum(axis=1), 1.0, rtol=1e-5)


@pytest.mark.unit
def test_catboost_prediction_probabilities(sample_features_labels, catboost_params):
    """Test predicted probabilities are in valid range."""
    from champion_v13_catboost import CatBoostClassifier

    X, y = sample_features_labels

    model = CatBoostClassifier(**catboost_params)
    model.fit(X, y)

    predictions = model.predict_proba(X)[:, 1]  # Positive class

    # All probabilities should be in [0, 1]
    assert np.all((predictions >= 0) & (predictions <= 1))


@pytest.mark.unit
def test_catboost_prediction_consistency(sample_features_labels, catboost_params):
    """Test prediction consistency across multiple calls."""
    from champion_v13_catboost import CatBoostClassifier

    X, y = sample_features_labels

    model = CatBoostClassifier(**catboost_params)
    model.fit(X, y)

    # Predict twice
    pred1 = model.predict_proba(X)
    pred2 = model.predict_proba(X)

    # Should be identical
    np.testing.assert_array_equal(pred1, pred2)


# ============================================================================
# Test Ensemble Weighting
# ============================================================================

@pytest.mark.unit
def test_ensemble_weights():
    """Test ensemble weighting is correct."""
    from champion_v13_catboost import EnsemblePredictor

    # Create mock predictions from 3 models
    np.random.seed(42)
    n_samples = 100

    pred1 = np.random.rand(n_samples)
    pred2 = np.random.rand(n_samples)
    pred3 = np.random.rand(n_samples)

    predictions = np.column_stack([pred1, pred2, pred3])
    weights = np.array([0.5, 0.3, 0.2])

    ensemble = EnsemblePredictor(weights=weights)
    ensemble_pred = ensemble.predict(predictions)

    # Verify weighted average
    expected = pred1 * 0.5 + pred2 * 0.3 + pred3 * 0.2
    np.testing.assert_allclose(ensemble_pred, expected)


@pytest.mark.unit
def test_ensemble_weights_normalization():
    """Test ensemble weights are automatically normalized."""
    from champion_v13_catboost import EnsemblePredictor

    # Weights that don't sum to 1
    weights = np.array([2.0, 1.0, 1.0])

    ensemble = EnsemblePredictor(weights=weights, normalize=True)

    # Should normalize to sum to 1
    np.testing.assert_allclose(ensemble.weights_.sum(), 1.0)
    expected_weights = np.array([0.5, 0.25, 0.25])
    np.testing.assert_allclose(ensemble.weights_, expected_weights)


@pytest.mark.unit
def test_ensemble_uniform_weights():
    """Test ensemble with uniform weights."""
    from champion_v13_catboost import EnsemblePredictor

    np.random.seed(42)
    predictions = np.random.rand(100, 5)  # 5 models

    ensemble = EnsemblePredictor()  # Default: uniform weights
    ensemble_pred = ensemble.predict(predictions)

    # Should be simple average
    expected = predictions.mean(axis=1)
    np.testing.assert_allclose(ensemble_pred, expected)


# ============================================================================
# Test Cross-Validation Integration
# ============================================================================

@pytest.mark.unit
def test_catboost_cross_validation(sample_features_labels):
    """Test CatBoost with cross-validation."""
    from champion_v13_catboost import CatBoostCVClassifier

    X, y = sample_features_labels

    params = {
        'iterations': 10,
        'verbose': False,
        'random_seed': 42
    }

    cv_model = CatBoostCVClassifier(n_folds=3, **params)
    cv_model.fit(X, y)

    # Should train n_folds models
    assert len(cv_model.models_) == 3

    # All models should be fitted
    for model in cv_model.models_:
        assert model.is_fitted()


@pytest.mark.unit
def test_catboost_cv_predictions(sample_features_labels):
    """Test cross-validation out-of-fold predictions."""
    from champion_v13_catboost import CatBoostCVClassifier

    X, y = sample_features_labels

    params = {
        'iterations': 10,
        'verbose': False,
        'random_seed': 42
    }

    cv_model = CatBoostCVClassifier(n_folds=3, **params)
    cv_model.fit(X, y)

    # Get OOF predictions
    oof_preds = cv_model.predict_oof(X, y)

    # Should have predictions for all samples
    assert len(oof_preds) == len(y)
    assert np.all((oof_preds >= 0) & (oof_preds <= 1))


@pytest.mark.unit
def test_catboost_cv_scoring(sample_features_labels):
    """Test cross-validation scoring."""
    from champion_v13_catboost import CatBoostCVClassifier

    X, y = sample_features_labels

    # Ensure balanced classes for ROC-AUC
    y = np.array([0, 1] * (len(y) // 2))[:len(y)]

    params = {
        'iterations': 20,
        'verbose': False,
        'random_seed': 42
    }

    cv_model = CatBoostCVClassifier(n_folds=3, **params)
    cv_model.fit(X, y)

    # Get OOF score
    oof_preds = cv_model.predict_oof(X, y)
    score = roc_auc_score(y, oof_preds)

    # Should be reasonable (random = 0.5)
    assert 0.3 <= score <= 1.0


# ============================================================================
# Test Feature Importance
# ============================================================================

@pytest.mark.unit
def test_catboost_feature_importance(sample_features_labels, catboost_params):
    """Test CatBoost feature importance extraction."""
    from champion_v13_catboost import CatBoostClassifier

    X, y = sample_features_labels

    model = CatBoostClassifier(**catboost_params)
    model.fit(X, y)

    importance = model.get_feature_importance()

    # Should have importance for all features
    assert len(importance) == X.shape[1]

    # All importances should be non-negative
    assert np.all(importance >= 0)

    # Sum should be positive (at least some features used)
    assert importance.sum() > 0


@pytest.mark.unit
def test_catboost_feature_ranking(sample_features_labels, catboost_params):
    """Test feature importance ranking."""
    from champion_v13_catboost import CatBoostClassifier

    X, y = sample_features_labels

    model = CatBoostClassifier(**catboost_params)
    model.fit(X, y)

    # Get top features
    top_features = model.get_top_features(n=10)

    # Should return at most 10 features
    assert len(top_features) <= 10

    # Should be sorted by importance (descending)
    importances = [imp for _, imp in top_features]
    assert importances == sorted(importances, reverse=True)


# ============================================================================
# Test Model Persistence
# ============================================================================

@pytest.mark.unit
def test_catboost_save_load(sample_features_labels, catboost_params, tmp_path):
    """Test CatBoost model save and load."""
    from champion_v13_catboost import CatBoostClassifier

    X, y = sample_features_labels

    # Train model
    model = CatBoostClassifier(**catboost_params)
    model.fit(X, y)

    # Get predictions before save
    pred_before = model.predict_proba(X)

    # Save model
    model_path = tmp_path / "catboost_model.cbm"
    model.save_model(str(model_path))

    # Load model
    loaded_model = CatBoostClassifier(**catboost_params)
    loaded_model.load_model(str(model_path))

    # Get predictions after load
    pred_after = loaded_model.predict_proba(X)

    # Predictions should match
    np.testing.assert_array_equal(pred_before, pred_after)


# ============================================================================
# Test Hyperparameter Validation
# ============================================================================

@pytest.mark.unit
def test_catboost_invalid_iterations():
    """Test invalid iterations parameter."""
    from champion_v13_catboost import CatBoostClassifier

    with pytest.raises(ValueError):
        CatBoostClassifier(iterations=-1)


@pytest.mark.unit
def test_catboost_invalid_learning_rate():
    """Test invalid learning rate parameter."""
    from champion_v13_catboost import CatBoostClassifier

    with pytest.raises(ValueError):
        CatBoostClassifier(learning_rate=0)  # Must be positive


@pytest.mark.unit
def test_catboost_invalid_depth():
    """Test invalid depth parameter."""
    from champion_v13_catboost import CatBoostClassifier

    with pytest.raises(ValueError):
        CatBoostClassifier(depth=0)  # Must be positive


# ============================================================================
# Test Categorical Features
# ============================================================================

@pytest.mark.unit
def test_catboost_with_categorical_features():
    """Test CatBoost with categorical features."""
    from champion_v13_catboost import CatBoostClassifier

    # Create data with categorical features
    X = pd.DataFrame({
        'num1': np.random.rand(50),
        'num2': np.random.rand(50),
        'cat1': np.random.choice(['A', 'B', 'C'], 50),
        'cat2': np.random.choice(['X', 'Y'], 50)
    })
    y = np.random.randint(0, 2, 50)

    model = CatBoostClassifier(
        iterations=10,
        verbose=False,
        random_seed=42,
        cat_features=['cat1', 'cat2']
    )

    model.fit(X, y)

    # Should train successfully
    assert model.is_fitted()


# ============================================================================
# Test Handling of Edge Cases
# ============================================================================

@pytest.mark.unit
def test_catboost_with_missing_values():
    """Test CatBoost handling of missing values."""
    from champion_v13_catboost import CatBoostClassifier

    X = np.random.rand(50, 10)
    X[np.random.rand(50, 10) < 0.1] = np.nan  # 10% missing
    y = np.random.randint(0, 2, 50)

    model = CatBoostClassifier(iterations=10, verbose=False, random_seed=42)

    # CatBoost handles NaN natively
    model.fit(X, y)
    predictions = model.predict_proba(X)

    assert predictions.shape == (50, 2)


@pytest.mark.unit
def test_catboost_single_class_error():
    """Test error when training data has single class."""
    from champion_v13_catboost import CatBoostClassifier

    X = np.random.rand(50, 10)
    y = np.ones(50)  # All same class

    model = CatBoostClassifier(iterations=10, verbose=False)

    # Should raise error or warning
    with pytest.raises(ValueError):
        model.fit(X, y)


@pytest.mark.unit
def test_catboost_imbalanced_classes(sample_features_labels):
    """Test CatBoost with imbalanced classes."""
    from champion_v13_catboost import CatBoostClassifier

    X, y = sample_features_labels

    # Create imbalance (90% class 0, 10% class 1)
    y = np.array([0] * 18 + [1] * 2)
    X = X[:20]

    params = {
        'iterations': 10,
        'verbose': False,
        'random_seed': 42,
        'auto_class_weights': 'Balanced'  # Handle imbalance
    }

    model = CatBoostClassifier(**params)
    model.fit(X, y)

    # Should train successfully
    assert model.is_fitted()
