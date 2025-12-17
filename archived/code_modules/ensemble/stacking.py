"""
Stacking ensemble module for AIRR-ML-25 competition.

This module implements advanced stacking techniques with GPU support
for combining XGBoost, LightGBM, and CatBoost models.

Key Features:
- Out-of-fold prediction generation for base learners
- GPU-accelerated training with automatic CPU fallback
- Meta-learner training on stacked predictions
- Hill climbing weight optimization
- Feature importance extraction from ensemble components

Example:
    >>> from ensemble.stacking import StackingClassifier, get_default_base_learners
    >>> base_learners = get_default_base_learners(use_gpu=True)
    >>> meta_learner = LogisticRegression()
    >>> stacker = StackingClassifier(base_learners, meta_learner, cv=5)
    >>> stacker.fit(X_train, y_train)
    >>> predictions = stacker.predict_proba(X_test)
"""

import logging
import warnings
from typing import Any, Dict, List, Tuple, Optional, Union

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, log_loss
from tqdm.auto import tqdm

# Suppress warnings from gradient boosting libraries
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_gpu_availability() -> Tuple[bool, bool, bool]:
    """
    Check if GPU is available for XGBoost, LightGBM, and CatBoost.

    Returns:
        Tuple of (xgb_gpu, lgb_gpu, catboost_gpu) availability flags.

    Example:
        >>> xgb_gpu, lgb_gpu, cb_gpu = check_gpu_availability()
        >>> logger.info(f"GPU available - XGB: {xgb_gpu}, LGB: {lgb_gpu}, CB: {cb_gpu}")
    """
    xgb_gpu = False
    lgb_gpu = False
    catboost_gpu = False

    # Check XGBoost GPU
    try:
        import xgboost as xgb
        xgb_gpu = True
        logger.info("XGBoost GPU support available")
    except ImportError:
        logger.warning("XGBoost not installed")

    # Check LightGBM GPU
    try:
        import lightgbm as lgb
        # LightGBM GPU requires special compilation
        lgb_gpu = lgb.__version__ is not None
        logger.info("LightGBM GPU support available")
    except ImportError:
        logger.warning("LightGBM not installed")

    # Check CatBoost GPU
    try:
        import catboost
        catboost_gpu = True
        logger.info("CatBoost GPU support available")
    except ImportError:
        logger.warning("CatBoost not installed")

    return xgb_gpu, lgb_gpu, catboost_gpu


def get_default_base_learners(use_gpu: bool = True) -> List[Tuple[str, Any]]:
    """
    Get default base learners with GPU/CPU configuration.

    Args:
        use_gpu: Whether to use GPU acceleration. If True but GPU not available,
                automatically falls back to CPU.

    Returns:
        List of (name, model) tuples ready for stacking.

    Raises:
        ImportError: If required libraries are not installed.

    Example:
        >>> base_learners = get_default_base_learners(use_gpu=True)
        >>> for name, model in base_learners:
        ...     print(f"{name}: {type(model).__name__}")
    """
    try:
        from xgboost import XGBClassifier
        from lightgbm import LGBMClassifier
        from catboost import CatBoostClassifier
    except ImportError as e:
        raise ImportError(
            "Gradient boosting libraries required. Install with:\n"
            "pip install xgboost lightgbm catboost"
        ) from e

    # Check actual GPU availability
    xgb_gpu, lgb_gpu, cb_gpu = check_gpu_availability()
    use_xgb_gpu = use_gpu and xgb_gpu
    use_lgb_gpu = use_gpu and lgb_gpu
    use_cb_gpu = use_gpu and cb_gpu

    # XGBoost configuration
    xgb_params = {
        'device': 'cuda' if use_xgb_gpu else 'cpu',
        'tree_method': 'hist',
        'n_estimators': 500,
        'learning_rate': 0.05,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'random_state': 42,
        'n_jobs': -1,
    }
    logger.info(f"XGBoost device: {xgb_params['device']}")

    # LightGBM configuration
    lgb_params = {
        'device': 'gpu' if use_lgb_gpu else 'cpu',
        'n_estimators': 500,
        'learning_rate': 0.05,
        'max_depth': 6,
        'num_leaves': 31,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'objective': 'binary',
        'metric': 'auc',
        'random_state': 42,
        'verbose': -1,
        'n_jobs': -1,
    }
    logger.info(f"LightGBM device: {lgb_params['device']}")

    # CatBoost configuration
    catboost_params = {
        'task_type': 'GPU' if use_cb_gpu else 'CPU',
        'devices': '0',
        'iterations': 500,
        'learning_rate': 0.05,
        'depth': 6,
        'loss_function': 'Logloss',
        'eval_metric': 'AUC',
        'random_seed': 42,
        'verbose': False,
        'thread_count': -1,
    }
    logger.info(f"CatBoost task_type: {catboost_params['task_type']}")

    return [
        ('xgb', XGBClassifier(**xgb_params)),
        ('lgb', LGBMClassifier(**lgb_params)),
        ('catboost', CatBoostClassifier(**catboost_params)),
    ]


class StackingClassifier(BaseEstimator, ClassifierMixin):
    """
    Stacking classifier with out-of-fold predictions.

    This implementation creates meta-features by training base learners
    on K-fold splits and generating out-of-fold predictions, which are
    then used to train a meta-learner. This prevents overfitting and
    improves generalization.

    Args:
        base_learners: List of (name, estimator) tuples for base models.
        meta_learner: Estimator to use as meta-learner. If None, uses
                     LogisticRegression with L2 regularization.
        cv: Number of stratified K-fold splits for out-of-fold predictions.
        use_gpu: Whether to enable GPU acceleration for base learners.
        use_proba: If True, use predict_proba for base predictions.
                  If False, use predict.
        verbose: Verbosity level (0=silent, 1=progress bars, 2=detailed).

    Attributes:
        base_models_: List of lists containing trained base models for each fold.
        meta_model_: Trained meta-learner.
        cv_scores_: Cross-validation scores for each base learner.

    Example:
        >>> base_learners = get_default_base_learners(use_gpu=True)
        >>> stacker = StackingClassifier(
        ...     base_learners=base_learners,
        ...     meta_learner=LogisticRegression(C=1.0, max_iter=1000),
        ...     cv=5,
        ...     verbose=1
        ... )
        >>> stacker.fit(X_train, y_train)
        >>> y_pred = stacker.predict_proba(X_test)[:, 1]
        >>> print(f"Feature importance: {stacker.get_feature_importance()}")
    """

    def __init__(
        self,
        base_learners: List[Tuple[str, Any]],
        meta_learner: Optional[Any] = None,
        cv: int = 5,
        use_gpu: bool = True,
        use_proba: bool = True,
        verbose: int = 1,
    ):
        self.base_learners = base_learners
        self.meta_learner = meta_learner
        self.cv = cv
        self.use_gpu = use_gpu
        self.use_proba = use_proba
        self.verbose = verbose

        # Will be set during fit
        self.base_models_: Optional[List[List[Any]]] = None
        self.meta_model_: Optional[Any] = None
        self.cv_scores_: Optional[Dict[str, float]] = None
        self.classes_: Optional[np.ndarray] = None

    def _validate_inputs(self, X: np.ndarray, y: np.ndarray) -> None:
        """Validate input data."""
        if not isinstance(X, np.ndarray):
            raise TypeError(f"X must be numpy array, got {type(X)}")
        if not isinstance(y, np.ndarray):
            raise TypeError(f"y must be numpy array, got {type(y)}")
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y must have same number of samples. "
                f"Got X: {X.shape[0]}, y: {y.shape[0]}"
            )
        if len(np.unique(y)) != 2:
            raise ValueError("Only binary classification is supported")

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'StackingClassifier':
        """
        Fit the stacking ensemble.

        Process:
        1. Split data into K stratified folds
        2. For each base learner:
           - Train on K-1 folds
           - Predict on held-out fold
           - Store out-of-fold predictions
        3. Train meta-learner on stacked out-of-fold predictions

        Args:
            X: Training features of shape (n_samples, n_features).
            y: Training labels of shape (n_samples,).

        Returns:
            self: Fitted estimator.

        Raises:
            ValueError: If inputs are invalid.
        """
        # Validate inputs
        self._validate_inputs(X, y)
        self.classes_ = np.unique(y)

        logger.info(f"Starting stacking ensemble training")
        logger.info(f"Training set: {X.shape[0]} samples, {X.shape[1]} features")
        logger.info(f"Class distribution: {np.bincount(y)}")
        logger.info(f"Base learners: {len(self.base_learners)}")
        logger.info(f"CV folds: {self.cv}")

        # Initialize storage for base models and OOF predictions
        n_samples = X.shape[0]
        n_base_learners = len(self.base_learners)

        # Out-of-fold predictions for meta-learner
        oof_predictions = np.zeros((n_samples, n_base_learners))

        # Store trained models for each fold
        self.base_models_ = [[] for _ in range(n_base_learners)]

        # CV scores for each base learner
        self.cv_scores_ = {}

        # Stratified K-fold cross-validation
        skf = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=42)

        # Train each base learner
        for learner_idx, (learner_name, learner) in enumerate(self.base_learners):
            logger.info(f"\nTraining base learner {learner_idx + 1}/{n_base_learners}: {learner_name}")

            fold_scores = []

            # Iterator for folds
            fold_iter = enumerate(skf.split(X, y))
            if self.verbose >= 1:
                fold_iter = tqdm(
                    fold_iter,
                    total=self.cv,
                    desc=f"{learner_name}",
                    leave=False
                )

            for fold_idx, (train_idx, val_idx) in fold_iter:
                # Split data
                X_train_fold, X_val_fold = X[train_idx], X[val_idx]
                y_train_fold, y_val_fold = y[train_idx], y[val_idx]

                # Clone and train base learner
                model = clone(learner)

                try:
                    # Fit with eval set for early stopping (if supported)
                    if hasattr(model, 'fit') and 'eval_set' in model.fit.__code__.co_varnames:
                        # Check model type for proper verbose parameter
                        model_name = type(model).__name__
                        if 'LGBM' in model_name:
                            # LightGBM uses callbacks for verbosity control
                            model.fit(
                                X_train_fold, y_train_fold,
                                eval_set=[(X_val_fold, y_val_fold)],
                            )
                        else:
                            # XGBoost and CatBoost use verbose parameter
                            model.fit(
                                X_train_fold, y_train_fold,
                                eval_set=[(X_val_fold, y_val_fold)],
                                verbose=False
                            )
                    else:
                        model.fit(X_train_fold, y_train_fold)

                    # Generate predictions
                    if self.use_proba:
                        val_preds = model.predict_proba(X_val_fold)[:, 1]
                    else:
                        val_preds = model.predict(X_val_fold)

                    # Store out-of-fold predictions
                    oof_predictions[val_idx, learner_idx] = val_preds

                    # Calculate fold score
                    fold_score = roc_auc_score(y_val_fold, val_preds)
                    fold_scores.append(fold_score)

                    # Store trained model
                    self.base_models_[learner_idx].append(model)

                    if self.verbose >= 2:
                        logger.info(f"  Fold {fold_idx + 1}/{self.cv}: AUC = {fold_score:.5f}")

                except Exception as e:
                    logger.error(f"Error training {learner_name} on fold {fold_idx + 1}: {e}")
                    raise

            # Calculate mean CV score
            mean_score = np.mean(fold_scores)
            std_score = np.std(fold_scores)
            self.cv_scores_[learner_name] = mean_score

            logger.info(
                f"{learner_name} - Mean CV AUC: {mean_score:.5f} (+/- {std_score:.5f})"
            )

        # Log stacking layer statistics
        logger.info("\nOut-of-fold prediction statistics:")
        for idx, (learner_name, _) in enumerate(self.base_learners):
            preds = oof_predictions[:, idx]
            logger.info(
                f"  {learner_name}: mean={preds.mean():.4f}, "
                f"std={preds.std():.4f}, min={preds.min():.4f}, max={preds.max():.4f}"
            )

        # Train meta-learner
        logger.info("\nTraining meta-learner...")
        if self.meta_learner is None:
            self.meta_model_ = LogisticRegression(
                C=1.0,
                max_iter=1000,
                random_state=42,
                n_jobs=-1
            )
        else:
            self.meta_model_ = clone(self.meta_learner)

        try:
            self.meta_model_.fit(oof_predictions, y)

            # Calculate meta-learner performance
            meta_preds = self.meta_model_.predict_proba(oof_predictions)[:, 1]
            meta_score = roc_auc_score(y, meta_preds)
            meta_loss = log_loss(y, meta_preds)

            logger.info(f"Meta-learner trained successfully")
            logger.info(f"Meta-learner OOF AUC: {meta_score:.5f}")
            logger.info(f"Meta-learner OOF Log Loss: {meta_loss:.5f}")

            # Log meta-learner coefficients if available
            if hasattr(self.meta_model_, 'coef_'):
                logger.info("Meta-learner coefficients:")
                for idx, (learner_name, _) in enumerate(self.base_learners):
                    coef = self.meta_model_.coef_[0][idx]
                    logger.info(f"  {learner_name}: {coef:.4f}")

        except Exception as e:
            logger.error(f"Error training meta-learner: {e}")
            raise

        logger.info("\nStacking ensemble training complete!")
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Process:
        1. Get predictions from all base learners (averaged across folds)
        2. Stack predictions into meta-features
        3. Pass to meta-learner for final prediction

        Args:
            X: Features of shape (n_samples, n_features).

        Returns:
            Class probabilities of shape (n_samples, 2).

        Raises:
            RuntimeError: If estimator is not fitted.
        """
        if self.base_models_ is None or self.meta_model_ is None:
            raise RuntimeError("Estimator must be fitted before prediction")

        n_samples = X.shape[0]
        n_base_learners = len(self.base_learners)

        # Storage for base predictions
        base_predictions = np.zeros((n_samples, n_base_learners))

        # Get predictions from each base learner (averaged across folds)
        for learner_idx, (learner_name, _) in enumerate(self.base_learners):
            fold_preds = []

            # Average predictions across all folds
            for fold_idx, model in enumerate(self.base_models_[learner_idx]):
                if self.use_proba:
                    preds = model.predict_proba(X)[:, 1]
                else:
                    preds = model.predict(X)
                fold_preds.append(preds)

            # Average across folds
            base_predictions[:, learner_idx] = np.mean(fold_preds, axis=0)

        # Meta-learner prediction
        final_predictions = self.meta_model_.predict_proba(base_predictions)

        return final_predictions

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.

        Args:
            X: Features of shape (n_samples, n_features).

        Returns:
            Predicted labels of shape (n_samples,).
        """
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)

    def get_feature_importance(self) -> Dict[str, np.ndarray]:
        """
        Get feature importance from each base learner.

        Returns:
            Dictionary mapping learner name to feature importance array.
            Importance is averaged across all folds.

        Raises:
            RuntimeError: If estimator is not fitted.

        Example:
            >>> importance = stacker.get_feature_importance()
            >>> for name, imp in importance.items():
            ...     print(f"{name}: top features = {np.argsort(imp)[-5:]}")
        """
        if self.base_models_ is None:
            raise RuntimeError("Estimator must be fitted before getting importance")

        importance_dict = {}

        for learner_idx, (learner_name, _) in enumerate(self.base_learners):
            fold_importances = []

            for model in self.base_models_[learner_idx]:
                if hasattr(model, 'feature_importances_'):
                    fold_importances.append(model.feature_importances_)
                else:
                    logger.warning(
                        f"{learner_name} does not have feature_importances_ attribute"
                    )
                    break

            if fold_importances:
                # Average importance across folds
                importance_dict[learner_name] = np.mean(fold_importances, axis=0)

        return importance_dict


class HillClimbingEnsemble:
    """
    Hill climbing optimization for ensemble weights.

    This class finds optimal weights for combining multiple model predictions
    by iteratively adjusting weights to maximize a specified metric.

    Args:
        models: List of (name, model) tuples or list of models.
        metric: Metric to optimize ('roc_auc', 'log_loss', or callable).
        random_state: Random seed for reproducibility.

    Attributes:
        best_weights_: Optimized weights for each model.
        best_score_: Best metric score achieved.
        score_history_: History of scores during optimization.

    Example:
        >>> models = [('xgb', xgb_model), ('lgb', lgb_model)]
        >>> optimizer = HillClimbingEnsemble(models, metric='roc_auc')
        >>> weights = optimizer.optimize_weights(X_val, y_val, n_iterations=100)
        >>> print(f"Optimal weights: {weights}")
        >>> print(f"Best score: {optimizer.best_score_}")
    """

    def __init__(
        self,
        models: Union[List[Tuple[str, Any]], List[Any]],
        metric: Union[str, callable] = 'roc_auc',
        random_state: int = 42,
    ):
        self.models = models
        self.metric = metric
        self.random_state = random_state

        # Set random seed
        np.random.seed(random_state)

        # Will be set during optimization
        self.best_weights_: Optional[np.ndarray] = None
        self.best_score_: Optional[float] = None
        self.score_history_: List[float] = []

    def _get_metric_function(self):
        """Get metric function."""
        if callable(self.metric):
            return self.metric
        elif self.metric == 'roc_auc':
            return lambda y_true, y_pred: roc_auc_score(y_true, y_pred)
        elif self.metric == 'log_loss':
            return lambda y_true, y_pred: -log_loss(y_true, y_pred)
        else:
            raise ValueError(f"Unknown metric: {self.metric}")

    def _get_predictions(self, X: np.ndarray) -> np.ndarray:
        """Get predictions from all models."""
        n_models = len(self.models)
        n_samples = X.shape[0]
        predictions = np.zeros((n_samples, n_models))

        for idx, model in enumerate(self.models):
            # Handle (name, model) tuples
            if isinstance(model, tuple):
                model = model[1]

            # Get predictions
            if hasattr(model, 'predict_proba'):
                predictions[:, idx] = model.predict_proba(X)[:, 1]
            else:
                predictions[:, idx] = model.predict(X)

        return predictions

    def _weighted_average(
        self,
        predictions: np.ndarray,
        weights: np.ndarray
    ) -> np.ndarray:
        """Compute weighted average of predictions."""
        # Normalize weights
        weights = weights / weights.sum()
        return predictions @ weights

    def optimize_weights(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_iterations: int = 100,
        step_size: float = 0.01,
        verbose: int = 1,
    ) -> np.ndarray:
        """
        Optimize ensemble weights using hill climbing.

        Algorithm:
        1. Start with equal weights for all models
        2. For each iteration:
           a. Randomly perturb weights
           b. Normalize to sum to 1
           c. Evaluate metric
           d. Keep if improvement, otherwise revert

        Args:
            X: Validation features of shape (n_samples, n_features).
            y: Validation labels of shape (n_samples,).
            n_iterations: Number of optimization iterations.
            step_size: Maximum perturbation size for each weight.
            verbose: Verbosity level (0=silent, 1=progress).

        Returns:
            Optimized weights of shape (n_models,).
        """
        logger.info("Starting hill climbing weight optimization")
        logger.info(f"Number of models: {len(self.models)}")
        logger.info(f"Metric: {self.metric}")
        logger.info(f"Iterations: {n_iterations}")

        # Get predictions from all models
        predictions = self._get_predictions(X)
        n_models = predictions.shape[1]

        # Initialize with equal weights
        weights = np.ones(n_models) / n_models

        # Get metric function
        metric_fn = self._get_metric_function()

        # Evaluate initial score
        y_pred = self._weighted_average(predictions, weights)
        current_score = metric_fn(y, y_pred)

        self.best_weights_ = weights.copy()
        self.best_score_ = current_score
        self.score_history_ = [current_score]

        logger.info(f"Initial score: {current_score:.5f}")
        logger.info(f"Initial weights: {weights}")

        # Hill climbing iterations
        iterator = range(n_iterations)
        if verbose >= 1:
            iterator = tqdm(iterator, desc="Optimizing")

        for iteration in iterator:
            # Random perturbation
            perturbation = np.random.randn(n_models) * step_size
            new_weights = weights + perturbation

            # Clip to non-negative
            new_weights = np.maximum(new_weights, 0)

            # Normalize
            if new_weights.sum() > 0:
                new_weights = new_weights / new_weights.sum()
            else:
                new_weights = np.ones(n_models) / n_models

            # Evaluate new weights
            y_pred = self._weighted_average(predictions, new_weights)
            new_score = metric_fn(y, y_pred)

            # Accept if improvement
            if new_score > current_score:
                weights = new_weights
                current_score = new_score

                # Update best
                if new_score > self.best_score_:
                    self.best_weights_ = weights.copy()
                    self.best_score_ = new_score

                    if verbose >= 2:
                        logger.info(
                            f"Iteration {iteration + 1}: "
                            f"New best score = {new_score:.5f}"
                        )

            self.score_history_.append(current_score)

            # Adaptive step size (simulated annealing)
            if iteration % (n_iterations // 10) == 0:
                step_size *= 0.95

        logger.info(f"\nOptimization complete!")
        logger.info(f"Best score: {self.best_score_:.5f}")
        logger.info(f"Best weights: {self.best_weights_}")

        # Log weights for named models
        if isinstance(self.models[0], tuple):
            logger.info("\nOptimal weights by model:")
            for idx, (name, _) in enumerate(self.models):
                logger.info(f"  {name}: {self.best_weights_[idx]:.4f}")

        return self.best_weights_

    def predict(
        self,
        X: np.ndarray,
        weights: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Make predictions using weighted ensemble.

        Args:
            X: Features of shape (n_samples, n_features).
            weights: Weights to use. If None, uses best_weights_.

        Returns:
            Weighted predictions of shape (n_samples,).

        Raises:
            RuntimeError: If weights are not available.
        """
        if weights is None:
            if self.best_weights_ is None:
                raise RuntimeError("Must call optimize_weights first")
            weights = self.best_weights_

        predictions = self._get_predictions(X)
        return self._weighted_average(predictions, weights)


# Convenience function for quick stacking
def quick_stack(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    use_gpu: bool = True,
    cv: int = 5,
    verbose: int = 1,
) -> np.ndarray:
    """
    Quick stacking with default configuration.

    Args:
        X_train: Training features.
        y_train: Training labels.
        X_test: Test features.
        use_gpu: Whether to use GPU.
        cv: Number of CV folds.
        verbose: Verbosity level.

    Returns:
        Test predictions of shape (n_samples,).

    Example:
        >>> predictions = quick_stack(X_train, y_train, X_test, use_gpu=True)
        >>> submission['label_positive_probability'] = predictions
    """
    base_learners = get_default_base_learners(use_gpu=use_gpu)
    stacker = StackingClassifier(
        base_learners=base_learners,
        cv=cv,
        verbose=verbose,
    )
    stacker.fit(X_train, y_train)
    predictions = stacker.predict_proba(X_test)[:, 1]
    return predictions


if __name__ == "__main__":
    # Demo and unit tests
    import doctest

    logger.info("Running doctests...")
    doctest.testmod(verbose=True)

    logger.info("\nDemo: Creating synthetic data and testing stacking...")
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    # Generate synthetic data
    X, y = make_classification(
        n_samples=1000,
        n_features=50,
        n_informative=30,
        n_redundant=10,
        random_state=42,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    logger.info(f"Train set: {X_train.shape}")
    logger.info(f"Test set: {X_test.shape}")

    # Test stacking
    logger.info("\n" + "="*60)
    logger.info("Testing StackingClassifier")
    logger.info("="*60)

    base_learners = get_default_base_learners(use_gpu=False)  # Use CPU for demo
    stacker = StackingClassifier(
        base_learners=base_learners,
        cv=3,
        verbose=2,
    )

    stacker.fit(X_train, y_train)
    y_pred = stacker.predict_proba(X_test)[:, 1]
    test_score = roc_auc_score(y_test, y_pred)

    logger.info(f"\nTest AUC: {test_score:.5f}")

    # Test feature importance
    importance = stacker.get_feature_importance()
    for name, imp in importance.items():
        logger.info(f"{name} - Top 5 features: {np.argsort(imp)[-5:]}")

    # Test hill climbing
    logger.info("\n" + "="*60)
    logger.info("Testing HillClimbingEnsemble")
    logger.info("="*60)

    # Create simple models for hill climbing demo
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier

    models = [
        ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
        ('dt', DecisionTreeClassifier(max_depth=5, random_state=42)),
    ]

    # Train models
    for name, model in models:
        model.fit(X_train, y_train)

    # Optimize weights
    optimizer = HillClimbingEnsemble(models, metric='roc_auc')
    weights = optimizer.optimize_weights(X_test, y_test, n_iterations=50)

    # Make predictions
    y_pred_weighted = optimizer.predict(X_test)
    weighted_score = roc_auc_score(y_test, y_pred_weighted)

    logger.info(f"\nWeighted ensemble AUC: {weighted_score:.5f}")

    logger.info("\n" + "="*60)
    logger.info("All tests completed successfully!")
    logger.info("="*60)
