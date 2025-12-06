# Champion Pipeline Guide

## Overview

The **Champion Pipeline** (`src/pipeline/champion_pipeline.py`) is a production-ready, GPU-accelerated machine learning pipeline designed to win the AIRR-ML-25 competition.

**Target Score:** 0.82+ (Current best: 0.81364)

**Key Features:**
- Multi-scale k-mer features (k=3,4,5)
- V/J gene usage patterns
- Clonality and diversity metrics
- GPU-accelerated ensemble (XGBoost, LightGBM, CatBoost)
- Stacked meta-learning with Ridge Regression
- Feature importance-based sequence identification

---

## Architecture

### 1. Feature Engineering

#### Multi-Scale K-mers
Extract k-mer frequencies for k=3, 4, 5:
```python
sequences = ['CASSLGQAY', 'CASSYNLTK', ...]
kmers = extract_kmers_multiscale(sequences, k_values=[3,4,5])
# Result: {'k3_CAS': 2, 'k3_ASS': 2, 'k4_CASS': 2, ...}
```

#### V/J Gene Features
Family-level V gene, J gene, and VJ pair usage:
```python
v_calls = ['TRBV20-1', 'TRBV6-2', ...]
j_calls = ['TRBJ2-7', 'TRBJ1-3', ...]
vj_feats = extract_vj_features(v_calls, j_calls)
# Result: {'v_TRBV20': 0.33, 'j_TRBJ2': 0.67, 'vj_TRBV20_TRBJ2': 0.33, ...}
```

#### Diversity Metrics
Repertoire-level diversity measures:
- **Shannon Entropy:** Evenness of clonotype distribution
- **Gini-Simpson Index:** Probability two random sequences differ
- **Clonality:** Normalized entropy (0=diverse, 1=clonal)
- **D50:** Minimum clones for 50% of repertoire
- **Richness:** Number of unique sequences

```python
diversity = extract_diversity_metrics(sequences)
# Result: {'shannon': 3.45, 'clonality': 0.23, 'd50': 0.15, ...}
```

#### CDR3 Length Statistics
Distribution of sequence lengths:
```python
length_feats = extract_length_features(sequences)
# Result: {'len_mean': 14.5, 'len_std': 2.3, 'len_median': 14, ...}
```

### 2. Model Ensemble

#### Base Learners
Four complementary models:

1. **XGBoost** (GPU)
   - Tree method: CUDA-accelerated
   - Max depth: 6
   - Learning rate: 0.1
   - 300 estimators

2. **LightGBM** (GPU)
   - Device: GPU if available
   - Max depth: 6
   - Learning rate: 0.1
   - 300 estimators

3. **CatBoost** (GPU)
   - Task type: GPU if available
   - Depth: 6
   - Learning rate: 0.1
   - 300 iterations

4. **Logistic Regression** (L1)
   - Penalty: L1 (LASSO)
   - Solver: SAGA
   - C: 0.1
   - Provides feature coefficients for Task B

#### Meta-Learner
**Ridge Regression** combines base predictions:
```python
# Collect out-of-fold predictions
X_meta = [xgb_pred, lgb_pred, cat_pred, logreg_pred]

# Train meta-model
meta_model = Ridge(alpha=1.0)
meta_model.fit(X_meta, y_val)
```

### 3. Sequence Identification (Task B)

Feature importance from ensemble guides sequence scoring:

```python
# Aggregate importance from all models
importances = {
    'k3_CAS': 0.15,
    'k3_SYN': 0.12,
    'k4_CASS': 0.18,
    ...
}

# Score sequences
for seq in unique_sequences:
    score = sum(importances[kmer] for kmer in extract_kmers(seq))

# Return top 50,000
top_seqs = sequences.nlargest(50000, 'score')
```

---

## Usage

### Command Line Interface

#### Basic Usage
```bash
python src/pipeline/champion_pipeline.py \
    --train_root ./data/train_datasets/train_datasets \
    --test_root ./data/test_datasets/test_datasets \
    --out_dir ./results_champion \
    --n_jobs 8 \
    --device cuda
```

#### Using the Shell Script
```bash
# GPU execution
./run_champion_pipeline.sh cuda

# CPU execution (no GPU available)
./run_champion_pipeline.sh cpu
```

### Python API

#### Full Pipeline
```python
from src.pipeline.champion_pipeline import ImmuneStatePredictor

# Initialize
predictor = ImmuneStatePredictor(
    n_jobs=8,
    device='cuda',
    k_values=[3, 4, 5],
    val_size=0.2,
    verbose=True
)

# Train
predictor.fit('./data/train_datasets/train_datasets/train_dataset_1')

# Predict (Task A)
predictions = predictor.predict_proba('./data/test_datasets/test_datasets/test_dataset_1')

# Identify sequences (Task B)
sequences = predictor.identify_associated_sequences(
    './data/train_datasets/train_datasets/train_dataset_1',
    top_k=50000
)
```

#### Custom Configuration
```python
from src.pipeline.champion_pipeline import ImmuneStatePredictor, PipelineConfig

config = PipelineConfig(
    n_jobs=16,
    device='cuda',
    k_values=[3, 4, 5],
    val_size=0.15,
    n_folds=5,
    ensemble_weights={
        'xgboost': 0.35,
        'lightgbm': 0.35,
        'catboost': 0.20,
        'logreg': 0.10
    },
    use_public_clonotypes=True,
    use_diversity_metrics=True,
    verbose=True
)

predictor = ImmuneStatePredictor(n_jobs=16, device='cuda')
predictor.config = config
predictor.fit(train_dir)
```

---

## Performance

### Expected Results

| Component | Validation AUC | Notes |
|-----------|----------------|-------|
| XGBoost | ~0.75 | GPU-accelerated |
| LightGBM | ~0.74 | GPU-accelerated |
| CatBoost | ~0.73 | GPU-accelerated |
| LogReg | ~0.68 | Used for Task B |
| **Ensemble** | **~0.78** | Stacked meta-learner |

### Feature Contributions

| Feature Type | Count | Importance |
|--------------|-------|------------|
| 3-mers | ~8,000 | 40% |
| 4-mers | ~160,000 | 35% |
| 5-mers | ~3,200,000 | 15% |
| V/J pairs | ~500 | 5% |
| Diversity | 6 | 3% |
| Length stats | 7 | 2% |

### Computational Requirements

| Resource | Requirement | Notes |
|----------|-------------|-------|
| GPU Memory | 16 GB | RTX 5080 recommended |
| RAM | 32 GB | For large feature matrices |
| Disk | 30 GB | For intermediate results |
| Time | 2-4 hours | Full 8-dataset pipeline |

---

## Testing

### Unit Tests
```bash
python test_champion_pipeline.py
```

Tests include:
1. Interface validation
2. Configuration
3. Feature extraction
4. Small dataset training
5. Prediction format
6. Sequence format

### Validation Script
```bash
python validate_submission.py results_champion/submissions.csv
```

Checks:
- Row count (404,213)
- Column names
- Data types
- Missing values
- Probability ranges

---

## Troubleshooting

### GPU Not Detected

**Symptom:** Pipeline falls back to CPU
```
Warning: CUDA not available, falling back to CPU
```

**Solutions:**
1. Check CUDA installation: `nvidia-smi`
2. Install PyTorch with CUDA: `pip install torch --index-url https://download.pytorch.org/whl/cu121`
3. Verify GPU libraries: `python -c "import torch; print(torch.cuda.is_available())"`

### Out of Memory

**Symptom:** CUDA out of memory error
```
RuntimeError: CUDA out of memory
```

**Solutions:**
1. Reduce batch size in training
2. Use fewer k-mer sizes: `k_values=[3]`
3. Process datasets sequentially
4. Switch to CPU: `--device cpu`

### Slow Feature Extraction

**Symptom:** Loading takes hours

**Solutions:**
1. Increase parallel jobs: `--n_jobs 16`
2. Use faster storage (SSD)
3. Reduce k-mer sizes
4. Cache intermediate results

### Row Count Mismatch

**Symptom:** Submission has wrong number of rows
```
Total rows: 400,000 (expected: 404,213)
```

**Solutions:**
1. Check all 8 training datasets processed
2. Verify all test datasets included
3. Ensure 50,000 sequences per training dataset
4. Check for missing test subsets (e.g., test_dataset_7_1, test_dataset_7_2)

---

## File Structure

```
airr-ml25-package/
├── src/
│   └── pipeline/
│       ├── __init__.py
│       └── champion_pipeline.py       # Main pipeline
├── docs/
│   └── champion_pipeline_guide.md     # This file
├── test_champion_pipeline.py          # Unit tests
├── run_champion_pipeline.sh           # Execution script
└── validate_submission.py             # Validation script
```

---

## Advanced Topics

### Adding Public Clonotypes

If `src/features/public_clonotypes.py` is available:

```python
from src.features.public_clonotypes import PublicClonotypeFeaturizer

# In load_repertoire_features():
if config.use_public_clonotypes and HAS_PUBLIC_CLONOTYPES:
    public_featurizer = PublicClonotypeFeaturizer()
    public_feats = public_featurizer.transform(repertoire)
    all_features.update(public_feats)
```

### Custom Feature Engineering

Add custom features by extending `load_repertoire_features()`:

```python
def extract_custom_features(sequences: List[str]) -> Dict[str, float]:
    """Your custom features."""
    return {
        'my_feature_1': compute_feature_1(sequences),
        'my_feature_2': compute_feature_2(sequences),
    }

# In load_repertoire_features():
custom_feats = extract_custom_features(sequences)
all_features.update(custom_feats)
```

### Hyperparameter Tuning

Use Optuna for automated tuning:

```python
import optuna

def objective(trial):
    config = PipelineConfig(
        n_jobs=8,
        device='cuda',
        ensemble_weights={
            'xgboost': trial.suggest_float('w_xgb', 0.2, 0.4),
            'lightgbm': trial.suggest_float('w_lgb', 0.2, 0.4),
            'catboost': trial.suggest_float('w_cat', 0.1, 0.3),
            'logreg': trial.suggest_float('w_lr', 0.05, 0.2),
        }
    )
    # Train and evaluate
    ...
    return validation_auc

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)
```

---

## References

### Competition Resources
- [Competition Overview](https://www.kaggle.com/competitions/adaptive-immune-profiling-challenge-2025)
- [Official Code Template](https://github.com/uio-bmi/predict-airr)
- [Registered Protocol](https://github.com/uio-bmi/adaptive_immune_profiling_challenge_2025/blob/main/registered_report.pdf)

### Related Papers
- [State-of-the-art in AIRR Mining](https://www.sciencedirect.com/science/article/pii/S2452310020300524)
- [Modern Hopfield Networks for Repertoires](https://doi.org/10.1101/2020.04.12.038158)
- [immuneML Platform](https://pmc.ncbi.nlm.nih.gov/articles/PMC10312379/)

### Community Notebooks
- [XGBoost Baseline](https://www.kaggle.com/code/bakuer30/air-ml25-xgboost)
- [XGBoost + PCA](https://www.kaggle.com/code/jirkaborovec/airr-ml-25-naive-baseline-with-xgboost-pca)

---

## License

MIT License - See competition rules for submission requirements.

---

**Last Updated:** 2025-12-06
**Version:** 1.0.0
**Author:** AIRR-ML-25 Competition Team
