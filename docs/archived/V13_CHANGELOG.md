# Champion V13 Changelog

## 🚀 New Features

### 1. Enhanced VJ Gene Pair Features

V13 introduces comprehensive VJ gene pair feature extraction:

#### Feature Categories

| Category | Count | Description |
|----------|-------|-------------|
| **VJ Pairs** | 500 | Top VJ gene pair frequencies (normalized) |
| **V Families** | 20 | Top V gene family frequencies (e.g., TRBV20) |
| **J Families** | 10 | Top J gene family frequencies (e.g., TRBJ2) |
| **VJ Diversity** | 3 | Entropy, unique count, max frequency |

**Total: ~533 new VJ-related features** (compared to V12's basic 500 VJ pairs)

#### Gene Family Extraction

```python
# V gene family extraction
TRBV20-1   -> TRBV20
TRBV20*01  -> TRBV20
TRBV7-9*01 -> TRBV7

# J gene family extraction
TRBJ2-7    -> TRBJ2
TRBJ2*01   -> TRBJ2
TRBJ1-1*01 -> TRBJ1
```

#### VJ Diversity Metrics

- **vj_entropy**: Shannon entropy of VJ pair distribution
- **vj_unique**: Number of unique VJ pairs
- **vj_max_freq**: Maximum frequency of any VJ pair

### 2. CatBoost Integration

V13 adds CatBoost as the third model in the ensemble:

#### CatBoost Configuration

```python
CatBoostClassifier(
    iterations=1000,
    learning_rate=0.03,
    depth=8,
    loss_function='Logloss',
    eval_metric='AUC',
    task_type='GPU',
    devices='0',
    random_seed=42,
    verbose=False
)
```

#### Ensemble Strategy

**V12 Ensemble** (2 models):
```
prediction = 0.5 * xgb + 0.5 * lgb
```

**V13 Ensemble** (3 models):
```
prediction = 0.4 * xgb + 0.4 * lgb + 0.2 * catboost
```

### 3. Modern Python 3.10+ Patterns

#### Type Hints
```python
def extract_features_single(
    tsv_path: Path,
    kmer_vocab: List[str],
    vj_vocab: List[Tuple[str, str]],
    v_vocab: List[str],
    j_vocab: List[str],
    public_clones: List[str]
) -> Optional[Dict[str, float]]:
    ...
```

#### Dataclasses
```python
@dataclass
class V13Config:
    """V13 configuration with all hyperparameters."""
    train_root: Path = Path("...")
    n_threads: int = 8
    use_gpu: bool = True
    weight_xgb: float = 0.4
    weight_lgb: float = 0.4
    weight_cb: float = 0.2
```

#### Context Managers
```python
@contextmanager
def timer(name: str):
    start = time.time()
    logger.info(f"Starting: {name}")
    yield
    elapsed = time.time() - start
    logger.info(f"Completed: {name} ({elapsed:.2f}s)")
```

#### Structured Logging
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)
```

## 📊 Performance Improvements

### Feature Count Comparison

| Version | K-mers | VJ Pairs | V Families | J Families | Public Clones | Diversity | Total |
|---------|--------|----------|------------|------------|---------------|-----------|-------|
| **V12** | 5,000 | 500 | - | - | 2,500 | 6 | ~8,006 |
| **V13** | 5,000 | 500 | 20 | 10 | 2,500 | 9 | ~8,039 |

### Model Comparison

| Version | Models | Ensemble | GPU Support |
|---------|--------|----------|-------------|
| **V12** | XGBoost + LightGBM | 2-model equal weight | XGB + LGB |
| **V13** | XGBoost + LightGBM + CatBoost | 3-model weighted | XGB + LGB + CB |

### Expected Improvements

1. **Better generalization**: VJ family features capture broader patterns
2. **Increased diversity**: CatBoost adds different inductive bias
3. **Robustness**: 3-model ensemble reduces overfitting risk

## 🛠️ Code Quality Improvements

### Type Safety
- Full type hints on all functions
- Proper use of `Optional`, `Tuple`, `Dict`, `List`
- Type-safe configuration with dataclasses

### Error Handling
- Structured logging throughout
- Proper exception handling with logging
- Graceful degradation for failed samples

### Performance
- Efficient gene family extraction (O(1) string operations)
- Vectorized diversity calculations
- Parallel feature extraction with ThreadPoolExecutor

### Maintainability
- Clear separation of concerns
- Dataclass for configuration management
- Context managers for resource handling
- Comprehensive docstrings

## 🧪 Testing

### Unit Tests
```bash
python3 test_v13_features.py
```

Tests cover:
- V gene family extraction
- J gene family extraction
- VJ pair feature extraction
- Configuration validation
- Ensemble weight validation

### Integration Test
```bash
./auto_train_v13.sh
```

Full pipeline test:
1. GPU availability check
2. Dependency verification
3. Training on all 8 datasets
4. Test predictions on 11 datasets
5. Submission file generation
6. Format validation

## 📝 Usage

### Quick Start
```bash
# Run with auto-training script
./auto_train_v13.sh

# Or run directly
python3 champion_v13.py
```

### Output
```
submissions/submission_v13.csv
```

### Expected Format
- **404,213 rows** total
- **4,213 predictions** (Task A)
- **400,000 sequences** (Task B: 8 datasets × 50,000)

## 🔍 Key Differences from V12

| Aspect | V12 | V13 |
|--------|-----|-----|
| **VJ Features** | Basic 500 VJ pairs | 500 pairs + 20 V families + 10 J families + 3 diversity |
| **Ensemble** | XGB + LGB (50/50) | XGB + LGB + CB (40/40/20) |
| **Type Hints** | Partial | Full coverage |
| **Configuration** | Hard-coded | Dataclass-based |
| **Logging** | Print statements | Structured logging |
| **Context Managers** | None | Timer, resource management |

## 🎯 Expected Results

### Cross-Validation
- **V12 typical**: 0.75-0.80 AUC per dataset
- **V13 expected**: 0.76-0.81 AUC per dataset (+1-2% improvement)

### Leaderboard
- **Target**: Improve upon V12 submission
- **Key advantage**: Better capture of VJ gene usage patterns
- **Risk mitigation**: CatBoost adds ensemble diversity

## 🚦 Next Steps

If V13 performance is satisfactory:
1. Submit to Kaggle
2. Compare with V12 leaderboard score
3. Analyze feature importance differences
4. Consider VJ interaction terms (V14)

If further improvement needed:
1. Tune CatBoost hyperparameters
2. Adjust ensemble weights (e.g., 0.33/0.33/0.34)
3. Add VJ interaction features
4. Consider ESM2 embeddings integration (V14)

## 📚 References

### CatBoost Documentation
- [CatBoost GitHub](https://github.com/catboost/catboost)
- [CatBoost GPU Training](https://catboost.ai/en/docs/features/training-on-gpu)
- [CatBoost Python API](https://catboost.ai/en/docs/concepts/python-reference_catboost)

### Python Best Practices
- [Python Dataclasses](https://docs.python.org/3/library/dataclasses.html)
- [Python Type Hints](https://docs.python.org/3/library/typing.html)
- [Python Context Managers](https://docs.python.org/3/library/contextlib.html)

---

**Version**: 13.0.0
**Date**: 2025-12-17
**Author**: AIRR-ML-25 Championship Team
**Status**: Ready for Training
