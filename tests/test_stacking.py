#!/usr/bin/env python3
"""
Test script for ensemble stacking module.

This script validates the stacking implementation with synthetic data
and checks GPU availability.
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report

from ensemble.stacking import (
    StackingClassifier,
    HillClimbingEnsemble,
    get_default_base_learners,
    quick_stack,
    check_gpu_availability,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_gpu_availability():
    """Test GPU availability check."""
    logger.info("="*80)
    logger.info("Test 1: GPU Availability Check")
    logger.info("="*80)

    xgb_gpu, lgb_gpu, cb_gpu = check_gpu_availability()
    logger.info(f"XGBoost GPU: {xgb_gpu}")
    logger.info(f"LightGBM GPU: {lgb_gpu}")
    logger.info(f"CatBoost GPU: {cb_gpu}")

    return xgb_gpu or lgb_gpu or cb_gpu


def test_default_base_learners(use_gpu=False):
    """Test default base learner creation."""
    logger.info("\n" + "="*80)
    logger.info("Test 2: Default Base Learners")
    logger.info("="*80)

    base_learners = get_default_base_learners(use_gpu=use_gpu)
    logger.info(f"Created {len(base_learners)} base learners:")
    for name, model in base_learners:
        logger.info(f"  - {name}: {type(model).__name__}")

    return base_learners


def test_stacking_classifier(base_learners, use_gpu=False):
    """Test StackingClassifier with synthetic data."""
    logger.info("\n" + "="*80)
    logger.info("Test 3: StackingClassifier")
    logger.info("="*80)

    # Generate synthetic data
    logger.info("Generating synthetic classification data...")
    X, y = make_classification(
        n_samples=2000,
        n_features=100,
        n_informative=60,
        n_redundant=20,
        n_classes=2,
        weights=[0.6, 0.4],  # Imbalanced
        random_state=42,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    logger.info(f"Train set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    logger.info(f"Test set: {X_test.shape[0]} samples")
    logger.info(f"Class distribution (train): {np.bincount(y_train)}")
    logger.info(f"Class distribution (test): {np.bincount(y_test)}")

    # Create and train stacking classifier
    logger.info("\nTraining StackingClassifier...")
    stacker = StackingClassifier(
        base_learners=base_learners,
        meta_learner=None,  # Use default LogisticRegression
        cv=3,  # Use 3 folds for faster testing
        use_gpu=use_gpu,
        verbose=1,
    )

    stacker.fit(X_train, y_train)

    # Make predictions
    logger.info("\nMaking predictions on test set...")
    y_pred_proba = stacker.predict_proba(X_test)[:, 1]
    y_pred = stacker.predict(X_test)

    # Evaluate
    test_auc = roc_auc_score(y_test, y_pred_proba)
    logger.info(f"\nTest AUC: {test_auc:.5f}")

    logger.info("\nClassification Report:")
    logger.info(classification_report(y_test, y_pred, target_names=['Class 0', 'Class 1']))

    # Get CV scores
    logger.info("\nCross-Validation Scores:")
    for name, score in stacker.cv_scores_.items():
        logger.info(f"  {name}: {score:.5f}")

    # Get feature importance
    logger.info("\nFeature Importance:")
    importance = stacker.get_feature_importance()
    for name, imp in importance.items():
        top_5_idx = np.argsort(imp)[-5:]
        logger.info(f"  {name} - Top 5 features: {top_5_idx.tolist()}")
        logger.info(f"           - Importances: {imp[top_5_idx]}")

    return stacker, X_train, y_train, X_test, y_test


def test_hill_climbing(X_train, y_train, X_test, y_test):
    """Test HillClimbingEnsemble."""
    logger.info("\n" + "="*80)
    logger.info("Test 4: HillClimbingEnsemble")
    logger.info("="*80)

    # Train simple models
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression

    logger.info("Training base models for hill climbing...")
    models = [
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
        ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42)),
        ('lr', LogisticRegression(C=1.0, max_iter=1000, random_state=42, n_jobs=-1)),
    ]

    for name, model in models:
        logger.info(f"  Training {name}...")
        model.fit(X_train, y_train)

        # Individual model performance
        y_pred = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred)
        logger.info(f"    {name} AUC: {auc:.5f}")

    # Optimize weights
    logger.info("\nOptimizing ensemble weights...")
    optimizer = HillClimbingEnsemble(models, metric='roc_auc', random_state=42)
    weights = optimizer.optimize_weights(
        X_test, y_test,
        n_iterations=100,
        step_size=0.05,
        verbose=1
    )

    logger.info(f"\nOptimal weights: {weights}")
    logger.info(f"Best score: {optimizer.best_score_:.5f}")

    # Test prediction
    y_pred_weighted = optimizer.predict(X_test)
    weighted_auc = roc_auc_score(y_test, y_pred_weighted)
    logger.info(f"Weighted ensemble AUC: {weighted_auc:.5f}")

    return optimizer


def test_quick_stack(X_train, y_train, X_test, y_test, use_gpu=False):
    """Test quick_stack convenience function."""
    logger.info("\n" + "="*80)
    logger.info("Test 5: quick_stack Convenience Function")
    logger.info("="*80)

    logger.info("Running quick_stack...")
    predictions = quick_stack(
        X_train, y_train, X_test,
        use_gpu=use_gpu,
        cv=3,
        verbose=1,
    )

    auc = roc_auc_score(y_test, predictions)
    logger.info(f"\nQuick stack AUC: {auc:.5f}")

    return predictions


def main():
    """Run all tests."""
    logger.info("Starting ensemble stacking module tests...")
    logger.info(f"NumPy version: {np.__version__}")

    # Test 1: GPU availability
    gpu_available = test_gpu_availability()
    use_gpu = gpu_available

    # Test 2: Default base learners
    try:
        base_learners = test_default_base_learners(use_gpu=use_gpu)
    except ImportError as e:
        logger.error(f"Failed to create base learners: {e}")
        logger.info("Please install gradient boosting libraries:")
        logger.info("  pip install xgboost lightgbm catboost")
        return 1

    # Test 3: Stacking classifier
    try:
        stacker, X_train, y_train, X_test, y_test = test_stacking_classifier(
            base_learners, use_gpu=use_gpu
        )
    except Exception as e:
        logger.error(f"StackingClassifier test failed: {e}", exc_info=True)
        return 1

    # Test 4: Hill climbing
    try:
        optimizer = test_hill_climbing(X_train, y_train, X_test, y_test)
    except Exception as e:
        logger.error(f"HillClimbingEnsemble test failed: {e}", exc_info=True)
        return 1

    # Test 5: Quick stack
    try:
        predictions = test_quick_stack(X_train, y_train, X_test, y_test, use_gpu=use_gpu)
    except Exception as e:
        logger.error(f"quick_stack test failed: {e}", exc_info=True)
        return 1

    logger.info("\n" + "="*80)
    logger.info("ALL TESTS PASSED SUCCESSFULLY!")
    logger.info("="*80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
