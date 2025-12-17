# Champion V13 Quick Start Guide

## 🎯 Overview

Champion V13 introduces **VJ gene pair features** and **CatBoost integration** for improved AIRR-ML-25 competition performance.

### Key Improvements
- ✅ Enhanced VJ gene pair feature extraction (533 VJ-related features)
- ✅ CatBoost integrated into ensemble (3-model: XGB + LGB + CB)
- ✅ Modern Python 3.10+ patterns (type hints, dataclasses, context managers)
- ✅ Structured logging and better error handling
- ✅ Full GPU acceleration (CUDA + OpenCL + CatBoost GPU)

---

## ⚡ Quick Start

### 1. Verify GPU Support
```bash
python3 test_v13_gpu.py
```

**Expected output:**
```
✓ XGBoost CUDA: OK
✓ LightGBM GPU: OK
✓ CatBoost GPU: OK
```

### 2. Test Feature Extraction
```bash
python3 test_v13_features.py
```

**Expected output:**
```
✓ All V family tests passed!
✓ All J family tests passed!
✓ All VJ pair feature tests passed!
✓ All config tests passed!
```

### 3. Run Full Training
```bash
./auto_train_v13.sh
```

**Or run directly:**
```bash
python3 champion_v13.py
```

### 4. Check Output
```bash
ls -lh submissions/submission_v13.csv
wc -l submissions/submission_v13.csv  # Should be 404214 (header + 404213 data rows)
```

---

## 📊 Feature Breakdown

### Total Features: ~8,039

| Category | Count | Description |
|----------|-------|-------------|
| **K-mers** | 5,000 | Top 3-mer frequencies |
| **VJ Pairs** | 500 | Top VJ gene pair frequencies |
| **V Families** | 20 | V gene family frequencies (e.g., TRBV20) |
| **J Families** | 10 | J gene family frequencies (e.g., TRBJ2) |
| **VJ Diversity** | 3 | Entropy, unique count, max frequency |
| **Public Clones** | 2,500 | Shared sequence frequencies |
| **Diversity** | 4 | Entropy, Gini, unique count, max frequency |
| **CDR3 Stats** | 5 | Mean, std, median, min, max length |

---

## 🤖 Model Configuration

### Ensemble Composition

```python
# XGBoost (40% weight)
XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    device='cuda'
)

# LightGBM (40% weight)
LGBMClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    device='gpu'
)

# CatBoost (20% weight) - NEW!
CatBoostClassifier(
    iterations=1000,
    depth=8,
    learning_rate=0.03,
    task_type='GPU'
)
```

### Final Prediction
```python
prediction = 0.4 * xgb_pred + 0.4 * lgb_pred + 0.2 * catboost_pred
```

---

## 🎨 Code Quality Features

### Type Hints
```python
def extract_vj_pair_features(
    df: pd.DataFrame,
    vj_vocab: List[Tuple[str, str]],
    v_vocab: List[str],
    j_vocab: List[str]
) -> Dict[str, float]:
    ...
```

### Dataclass Configuration
```python
@dataclass
class V13Config:
    n_threads: int = 8
    use_gpu: bool = True
    weight_xgb: float = 0.4
    weight_lgb: float = 0.4
    weight_cb: float = 0.2
```

### Context Managers
```python
with timer("Feature extraction"):
    X_df = extract_features_parallel(...)
```

### Structured Logging
```python
logger.info(f"Dataset {ds_id}: {ds_name}")
logger.warning(f"Failed to process {fname}: {e}")
logger.error(f"GPU check failed: {e}")
```

---

## 📈 Expected Performance

### Cross-Validation AUC
- **V12 baseline**: 0.75-0.80 per dataset
- **V13 target**: 0.76-0.81 per dataset
- **Expected improvement**: +1-2% from VJ features + CatBoost

### Training Time
- **Dataset 1-8**: ~2-5 minutes per dataset (GPU)
- **Total training**: ~25-40 minutes
- **Test predictions**: ~5-10 minutes
- **Grand total**: ~30-50 minutes

### Resource Usage
- **GPU memory**: ~4-8 GB (RTX 5080 16GB is sufficient)
- **RAM**: ~8-16 GB
- **Disk**: ~20 GB (dataset) + ~100 MB (output)

---

## 🔧 Customization

### Adjust Ensemble Weights

Edit `champion_v13.py`:
```python
@dataclass
class V13Config:
    # Try different weights (must sum to 1.0)
    weight_xgb: float = 0.33
    weight_lgb: float = 0.33
    weight_cb: float = 0.34
```

### Adjust Feature Counts

```python
@dataclass
class V13Config:
    n_top_kmers: int = 5000      # Increase for more k-mer features
    n_top_vj_pairs: int = 500    # Increase for more VJ pairs
    n_top_v_families: int = 20   # More V gene families
    n_top_j_families: int = 10   # More J gene families
    n_public_clones: int = 2500  # More shared sequences
```

### Adjust Model Hyperparameters

```python
@dataclass
class V13Config:
    # CatBoost
    cb_iterations: int = 1000      # More iterations
    cb_depth: int = 8              # Tree depth
    cb_learning_rate: float = 0.03 # Learning rate
```

---

## 🐛 Troubleshooting

### GPU Not Available

**Error:**
```
✗ XGBoost CUDA: FAILED
✗ CatBoost GPU: FAILED
```

**Solution:**
```bash
# Check NVIDIA driver
nvidia-smi

# Check CUDA version
nvcc --version

# Reinstall XGBoost with CUDA
pip install xgboost --upgrade

# Reinstall CatBoost
pip install catboost --upgrade
```

### Memory Error

**Error:**
```
RuntimeError: CUDA out of memory
```

**Solution:**
Edit `champion_v13.py`:
```python
@dataclass
class V13Config:
    n_top_features: int = 500  # Reduce from 1000
    n_top_kmers: int = 3000    # Reduce from 5000
```

### Missing Dependencies

**Error:**
```
ModuleNotFoundError: No module named 'catboost'
```

**Solution:**
```bash
pip install catboost xgboost lightgbm scikit-learn pandas numpy tqdm
```

---

## 📝 Output Format

### Submission File Structure

```
submissions/submission_v13.csv
```

**Total rows: 404,213**
- Predictions: 4,213 (Task A)
- Sequences: 400,000 (Task B: 8 datasets × 50,000)

**Columns:**
```
ID, dataset, label_positive_probability, junction_aa, v_call, j_call
```

**Example:**
```csv
ID,dataset,label_positive_probability,junction_aa,v_call,j_call
rep_001,test_dataset_1,0.75342,-999.0,-999.0,-999.0
...
train_dataset_1_seq_top_1,train_dataset_1,-999.0,CASSLGQAY,TRBV20-1,TRBJ2-7
...
```

---

## 🚀 Next Steps

### After Successful Training

1. **Validate submission format:**
   ```bash
   python3 -c "import pandas as pd; df = pd.read_csv('submissions/submission_v13.csv'); print(f'Rows: {len(df)}, Columns: {df.columns.tolist()}')"
   ```

2. **Submit to Kaggle:**
   ```bash
   kaggle competitions submit -c adaptive-immune-profiling-challenge-2025 \
       -f submissions/submission_v13.csv \
       -m "V13: VJ Pairs + CatBoost ensemble (0.4 XGB + 0.4 LGB + 0.2 CB)"
   ```

3. **Monitor leaderboard:**
   - Check public leaderboard score
   - Compare with V12 submission
   - Analyze improvement

### If Score Improves

- Document successful hyperparameters
- Consider fine-tuning ensemble weights
- Plan V14 with additional features

### If Score Doesn't Improve

- Review feature importance
- Adjust ensemble weights
- Consider VJ interaction terms
- Investigate ESM2 embeddings

---

## 📚 Files Overview

| File | Description |
|------|-------------|
| `champion_v13.py` | Main training script |
| `auto_train_v13.sh` | Automated training pipeline |
| `test_v13_features.py` | Feature extraction unit tests |
| `test_v13_gpu.py` | GPU support verification |
| `V13_CHANGELOG.md` | Detailed changelog |
| `V13_QUICKSTART.md` | This file |

---

## ✅ Pre-Flight Checklist

Before running V13:

- [ ] GPU test passes (`test_v13_gpu.py`)
- [ ] Feature tests pass (`test_v13_features.py`)
- [ ] Dataset exists (`data/train_datasets/`)
- [ ] Enough disk space (~100 MB for output)
- [ ] Enough GPU memory (~8 GB minimum)
- [ ] Python 3.10+ installed
- [ ] All dependencies installed

---

## 💡 Tips

1. **Monitor GPU usage:**
   ```bash
   watch -n 1 nvidia-smi
   ```

2. **Check logs in real-time:**
   ```bash
   ./auto_train_v13.sh | tee v13_training.log
   ```

3. **Compare with V12:**
   ```bash
   diff submissions/submission_v12_robust.csv submissions/submission_v13.csv | head -20
   ```

4. **Feature importance analysis:**
   Add to end of `champion_v13.py`:
   ```python
   # After training
   importances = ensemble.xgb.feature_importances_
   top_features = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:50]
   for name, imp in top_features:
       print(f"{name:40s}: {imp:.4f}")
   ```

---

**Version:** 13.0.0
**Date:** 2025-12-17
**Status:** ✅ Ready for Production
**Tested:** RTX 5080, CUDA 12.x, Python 3.10
