# Ensemble Module for AIRR-ML-25

Production-ready ensemble learning implementations for the AIRR-ML-25 Adaptive Immune Profiling Challenge.

## Overview

This module provides GPU-accelerated ensemble methods combining XGBoost, LightGBM, and CatBoost for maximum predictive performance on TCR repertoire classification.

## Quick Start

```python
from ensemble.stacking import quick_stack

# One-line stacking ensemble
predictions = quick_stack(X_train, y_train, X_test, use_gpu=True, cv=5)
```

## Components

### 1. StackingClassifier

Main ensemble implementation using out-of-fold predictions and meta-learning.

**Key Features**:
- Stratified K-fold cross-validation
- Out-of-fold predictions prevent overfitting
- GPU-accelerated training
- Configurable meta-learner
- Feature importance extraction

**Example**:
```python
from ensemble.stacking import StackingClassifier, get_default_base_learners

base_learners = get_default_base_learners(use_gpu=True)
stacker = StackingClassifier(base_learners=base_learners, cv=5, verbose=1)
stacker.fit(X_train, y_train)
predictions = stacker.predict_proba(X_test)[:, 1]
```

### 2. HillClimbingEnsemble

Optimizes ensemble weights for maximum performance.

**Example**:
```python
from ensemble.stacking import HillClimbingEnsemble

optimizer = HillClimbingEnsemble(models, metric='roc_auc')
weights = optimizer.optimize_weights(X_val, y_val, n_iterations=100)
predictions = optimizer.predict(X_test)
```

### 3. Helper Functions

- `get_default_base_learners(use_gpu=True)`: Create GPU-optimized XGBoost, LightGBM, CatBoost
- `quick_stack()`: One-line stacking with default configuration
- `check_gpu_availability()`: Detect GPU support for each library

## Performance

**Benchmarks** (1,400 samples, 100 features, RTX 5080):

| Method | CV AUC | Test AUC | Training Time |
|--------|--------|----------|---------------|
| XGBoost | 0.944 | - | 1.6s |
| LightGBM | 0.948 | - | 3.1s |
| CatBoost | 0.941 | - | 30.0s |
| **Stacking** | **0.948** | **0.964** | **~35s** |

## Documentation

- **Complete Guide**: `/docs/ensemble_stacking_guide.md`
- **Quick Reference**: `/STACKING_QUICK_REFERENCE.md`
- **Implementation Summary**: `/ENSEMBLE_STACKING_SUMMARY.md`

## Testing

```bash
# Run comprehensive test suite
python3 test_stacking.py

# Run usage examples
python3 examples/stacking_example.py
```

## GPU Support

**Automatic Detection**: Module detects GPU availability and falls back to CPU if needed.

**Hardware Tested**:
- RTX 5080 16GB VRAM
- CUDA 11.0+

**Configuration**:
```python
# Force GPU
base_learners = get_default_base_learners(use_gpu=True)

# Force CPU (for debugging)
base_learners = get_default_base_learners(use_gpu=False)

# Check availability
from ensemble.stacking import check_gpu_availability
xgb_gpu, lgb_gpu, cb_gpu = check_gpu_availability()
```

## AIRR-ML-25 Integration

### Task A: Immune State Prediction

```python
# Per-repertoire predictions
predictions = quick_stack(X_train, y_train, X_test, use_gpu=True)
submission.loc[test_ids, 'label_positive_probability'] = predictions
```

### Task B: Sequence Identification

```python
# Feature importance for sequence ranking
stacker.fit(X_train, y_train)
importance = stacker.get_feature_importance()
avg_importance = np.mean([imp for imp in importance.values()], axis=0)
# Map to sequences and select top 50,000
```

### Per-Dataset Ensembles

```python
for dataset_id in range(1, 9):
    X_train = load_features(f'dataset_{dataset_id}')
    y_train = load_labels(f'dataset_{dataset_id}')
    X_test = load_test_features(f'dataset_{dataset_id}')
    predictions = quick_stack(X_train, y_train, X_test)
```

## Advanced Usage

### Custom Base Learners

```python
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

custom_learners = [
    ('xgb', XGBClassifier(n_estimators=1000, learning_rate=0.01, device='cuda')),
    ('lgb', LGBMClassifier(n_estimators=1000, learning_rate=0.01, device='gpu')),
]

stacker = StackingClassifier(base_learners=custom_learners, cv=10)
```

### Custom Meta-Learner

```python
from sklearn.ensemble import RandomForestClassifier

meta = RandomForestClassifier(n_estimators=200, random_state=42)
stacker = StackingClassifier(base_learners=base_learners, meta_learner=meta)
```

### Feature Importance Analysis

```python
stacker.fit(X_train, y_train)
importance = stacker.get_feature_importance()

for name, imp in importance.items():
    top_10 = np.argsort(imp)[-10:]
    print(f"{name} - Top 10 features: {top_10}")
```

### Weight Optimization

```python
# Train stacker
stacker.fit(X_train, y_train)

# Optimize weights on validation set
optimizer = HillClimbingEnsemble(stacker.base_models_, metric='roc_auc')
weights = optimizer.optimize_weights(X_val, y_val, n_iterations=200)

print(f"Optimal weights: {weights}")
print(f"Best AUC: {optimizer.best_score_:.4f}")
```

## API Reference

### StackingClassifier

```python
class StackingClassifier(base_learners, meta_learner=None, cv=5, use_gpu=True, verbose=1)
```

**Parameters**:
- `base_learners`: List of (name, model) tuples
- `meta_learner`: Meta-learner estimator (default: LogisticRegression)
- `cv`: Number of K-fold splits (default: 5)
- `use_gpu`: Enable GPU (default: True)
- `verbose`: Verbosity level 0-2 (default: 1)

**Methods**:
- `fit(X, y)`: Train ensemble
- `predict_proba(X)`: Predict probabilities
- `predict(X)`: Predict classes
- `get_feature_importance()`: Extract feature importance

**Attributes**:
- `base_models_`: Trained base models
- `meta_model_`: Trained meta-learner
- `cv_scores_`: CV scores per base learner

### HillClimbingEnsemble

```python
class HillClimbingEnsemble(models, metric='roc_auc', random_state=42)
```

**Parameters**:
- `models`: List of (name, model) tuples
- `metric`: Metric to optimize ('roc_auc', 'log_loss', or callable)
- `random_state`: Random seed (default: 42)

**Methods**:
- `optimize_weights(X, y, n_iterations=100, step_size=0.01)`: Find optimal weights
- `predict(X, weights=None)`: Make weighted predictions

**Attributes**:
- `best_weights_`: Optimized weights
- `best_score_`: Best metric score
- `score_history_`: Optimization history

## Troubleshooting

| Issue | Solution |
|-------|----------|
| GPU Out of Memory | Use `use_gpu=False` or reduce model complexity |
| Slow Training | Reduce `cv=3` or use fewer base learners |
| Poor Performance | Optimize weights with HillClimbingEnsemble |
| Import Error | Install: `pip install xgboost lightgbm catboost` |

## Requirements

```
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
xgboost>=2.0
lightgbm>=4.0
catboost>=1.2
tqdm>=4.66
```

## Files

| File | Lines | Purpose |
|------|-------|---------|
| `stacking.py` | 864 | Core implementation |
| `__init__.py` | 23 | Package exports |
| `../test_stacking.py` | 248 | Test suite |
| `../examples/stacking_example.py` | 365 | Usage examples |
| `../docs/ensemble_stacking_guide.md` | 552 | Documentation |

## License

MIT License - Part of AIRR-ML-25 Competition Framework

## Support

For detailed documentation and examples:
- Read: `/docs/ensemble_stacking_guide.md`
- Run: `python3 test_stacking.py`
- Examples: `python3 examples/stacking_example.py`

---

**Version**: 1.0.0 | **Last Updated**: 2025-12-06 | **Status**: Production Ready
