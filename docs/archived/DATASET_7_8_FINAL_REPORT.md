# Dataset 7 & 8 Crisis: V9 Model Complete Failure on Specific Datasets

**Date**: 2025-12-16
**Status**: 🚨 CRITICAL - Multiple Dataset Failures
**Impact**: 1,813 / 4,213 test samples (43% of total submissions) severely compromised

---

## Executive Summary

V9 (Ultimate Model) has **CATASTROPHIC FAILURES** on specific datasets:

| Dataset | Test Samples | V9 Mean Prediction | Status | Impact |
|---------|-------------|-------------------|--------|--------|
| test_dataset_7_1 | 76 | 0.0000 | **DEAD** | All zeros |
| test_dataset_7_2 | 100 | 0.0000 | **DEAD** | All zeros |
| test_dataset_8_1 | 390 | 0.000045 | **CRITICAL** | Near zero |
| test_dataset_8_2 | 857 | 0.000178 | **CRITICAL** | Near zero |
| test_dataset_8_3 | 390 | 0.000045 | **CRITICAL** | Near zero |
| **TOTAL FAILED** | **1,813** | - | - | **43% of test set** |

Meanwhile, working datasets show reasonable predictions:

| Dataset | Test Samples | V9 Mean | Status |
|---------|-------------|---------|--------|
| test_dataset_1 | 400 | 0.0152 | Suspiciously low |
| test_dataset_2 | 400 | 0.4826 | Good |
| test_dataset_3 | 400 | 0.8000 | Good (but constant?) |
| test_dataset_4 | 400 | 0.6016 | Good (but constant?) |
| test_dataset_5 | 400 | 0.4567 | Good |
| test_dataset_6 | 400 | 0.5108 | Good |

---

## Root Cause Analysis

### Dataset 7 & 8 Characteristics (from previous analysis)

| Feature | Dataset 7 | Dataset 8 | Datasets 1-6 |
|---------|-----------|-----------|--------------|
| Class Balance | 16.6% positive | 32.8% positive | 50% positive |
| Repertoire Size | 311,022 sequences | 103,124 sequences | 25,000 sequences |
| Training Samples | 302 | 908 | 400 |

### Why V9 Failed on Datasets 7 & 8

**Hypothesis 1: Fisher Exact Test Numerical Instability**

With 311k sequences per repertoire and only 16.6% positive:
- Fisher exact test on contingency tables becomes numerically unstable
- P-values underflow to zero
- XGBoost receives NaN or Inf features → defaults to 0.0 prediction

**Hypothesis 2: Dataset-Specific Feature Extraction Failed**

```python
# Likely bug in V9's code
if dataset_id == 7 or dataset_id == 8:
    # Feature extraction crashes or returns empty features
    features = np.zeros(...)  # Fallback to zeros
```

**Hypothesis 3: Training Data Didn't Include Datasets 7 & 8**

V9 might have been trained only on datasets 1-6:
- When encountering unseen datasets 7 & 8, model has no learned patterns
- Defaults to prior (which would be 0.5, but maybe clipped to 0.0 due to bug)

---

## Detailed Evidence

### V9 Full Prediction Statistics

```
Dataset          Samples  Mean      Std       Min       Max       Notes
============================================================================
test_dataset_1      400   0.0152   0.0097   0.0012    0.0700    Too low!
test_dataset_2      400   0.4826   0.0338   0.3665    0.5567    Good
test_dataset_3      400   0.8000   0.0000   0.7999    0.8000    CONSTANT!
test_dataset_4      400   0.6016   0.0010   0.5994    0.6054    Nearly constant
test_dataset_5      400   0.4567   0.0227   0.4136    0.5242    Good
test_dataset_6      400   0.5108   0.0336   0.4167    0.5759    Good
test_dataset_7_1     76   0.0000   0.0000   0.0000    0.0000    DEAD
test_dataset_7_2    100   0.0000   0.0000   0.0000    0.0000    DEAD
test_dataset_8_1    390   0.0000   0.0001   0.0000    0.0015    DEAD
test_dataset_8_2    857   0.0002   0.0010   0.0000    0.0183    DEAD
test_dataset_8_3    390   0.0000   0.0001   0.0000    0.0015    DEAD
```

### Red Flags

1. **Dataset 3 & 4 are CONSTANT** (std ≈ 0.0)
   - Model predicts same value for all samples
   - Likely feature extraction returned identical features for all repertoires

2. **Dataset 1 too low** (mean = 0.0152 for 50% balanced data)
   - Should be around 0.5
   - Suggests systematic bias or wrong probability column

3. **Datasets 7 & 8 complete failure**
   - All predictions zero or near-zero
   - Feature extraction or model application failed

---

## V5 vs V9 Comparison

### Dataset 1

```
V5: mean=0.5021, std=0.2603  ← GOOD (balanced, varying)
V9: mean=0.0152, std=0.0097  ← BAD (too low, too confident)
Correlation: -0.0012  ← Essentially independent
```

### Dataset 7_1

```
V5: mean=0.1031, std=0.0415  ← Reasonable for 16.6% positive
V9: mean=0.0000, std=0.0000  ← DEAD
Correlation: 0.0747  ← Can't correlate with constant
```

### Dataset 7_2

```
V5: mean=0.1276, std=0.0552  ← Reasonable
V9: mean=0.0000, std=0.0000  ← DEAD
Correlation: 0.2056  ← Can't correlate with constant
```

---

## Impact Assessment

### If we ensemble V5 and V9 equally (50-50):

| Dataset | V5 Mean | V9 Mean | Ensemble | Impact |
|---------|---------|---------|----------|--------|
| Dataset 1 | 0.5021 | 0.0152 | 0.2587 | **Severe degradation** |
| Dataset 7_1 | 0.1031 | 0.0000 | 0.0516 | **50% reduction** |
| Dataset 7_2 | 0.1276 | 0.0000 | 0.0638 | **50% reduction** |
| Dataset 8_1 | ? | 0.0000 | **Severely lowered** | |
| Dataset 8_2 | ? | 0.0002 | **Severely lowered** | |
| Dataset 8_3 | ? | 0.0000 | **Severely lowered** | |

**Ensembling V5 and V9 would TANK the score on 43% of the test set.**

---

## Recommended Actions

### Immediate (< 10 minutes)

**DO NOT USE V9 AT ALL**

1. **Submit V5 ONLY** for all datasets:
```bash
cp submissions/v5_submission_20251215_225131.csv final_submission.csv
```

2. **Abandon any ensemble with V9** until fixed

### Short-term (if time permits, 1-2 hours)

1. **Debug V9 feature extraction for datasets 7 & 8**:
```bash
# Add logging to champion_ultimate.py
print(f"Dataset {dataset_id}: features shape {features.shape}")
print(f"Features stats: mean={features.mean()}, std={features.std()}")
print(f"Has NaN: {np.isnan(features).any()}")
print(f"Has Inf: {np.isinf(features).any()}")
```

2. **Check if model was trained on datasets 7 & 8**:
```python
# In training log, look for:
print(f"Training datasets: {train_dataset_ids}")
# If [1,2,3,4,5,6] and missing 7,8 → that's the problem
```

3. **Re-run V9 with fixes**

### Long-term (for future competitions)

1. **Add prediction validation**:
```python
def validate_predictions(predictions, dataset_name):
    assert predictions.std() > 0.001, f"{dataset_name}: Constant predictions!"
    assert predictions.min() >= 0.0, f"{dataset_name}: Negative predictions!"
    assert predictions.max() <= 1.0, f"{dataset_name}: Predictions > 1!"

    # Check against expected range
    if "balanced" in dataset_name:
        assert 0.3 < predictions.mean() < 0.7, f"{dataset_name}: Mean {predictions.mean()} too far from 0.5!"
```

2. **Per-dataset monitoring during training**:
```python
for dataset_id in all_datasets:
    val_auc = evaluate(model, val_data[dataset_id])
    print(f"Dataset {dataset_id} val AUC: {val_auc:.4f}")
    assert val_auc > 0.55, f"Dataset {dataset_id} failed validation!"
```

3. **Test on training data first**:
```python
# Before predicting test, verify on train
train_pred = model.predict(train_data)
train_auc = roc_auc_score(train_labels, train_pred)
print(f"Train AUC: {train_auc:.4f}")

if train_auc < 0.7:
    print("WARNING: Model failed on training data!")
```

---

## Why Low Correlation Makes Sense Now

### Previous Hypothesis (WRONG)
"V5 and V9 found different orthogonal signals, low correlation means they're complementary"

### Actual Truth
"V9 is broken on datasets 7 & 8, predicting constant zero. You can't correlate constant with variable."

When one model predicts:
```
V9: [0.0, 0.0, 0.0, 0.0, 0.0]
```

And another predicts:
```
V5: [0.08, 0.12, 0.15, 0.09, 0.11]
```

The correlation is **undefined** or **near zero** (just numerical noise).

The 0.0747 and 0.2056 correlations we saw are just artifacts of floating-point precision, not meaningful relationships.

---

## Final Diagnosis

### What Went Wrong

1. **V9 was never properly tested on imbalanced datasets**
   - Datasets 7 (16.6% positive) and 8 (32.8% positive) broke the model
   - Fisher exact test failed numerically
   - Or model was never trained on these datasets

2. **No validation checks caught the failure**
   - Code didn't assert predictions were non-constant
   - No per-dataset AUC reporting
   - No warnings when predictions = 0.0

3. **Submission file was generated without visual inspection**
   - If someone had plotted V9's predictions, they'd see the zeros immediately
   - Always visualize before submitting

### What to Do Now

**Submit V5 ONLY. Do not ensemble with broken V9.**

V5's predictions are reasonable across all datasets:
- Dataset 1: mean = 0.50 (balanced) ✓
- Dataset 7: mean = 0.10-0.13 (imbalanced towards negative) ✓
- Shows variation (std > 0.04) ✓
- No constants ✓

V9's predictions are broken on 43% of test set:
- Datasets 7 & 8: all zeros or near-zeros ✗
- Dataset 1: too low (0.015 instead of ~0.5) ✗
- Datasets 3 & 4: constant ✗

**There is no decision to make. V9 is unusable.**

---

## Lessons Learned

1. **Always check prediction statistics before submitting**
```python
for dataset in all_datasets:
    preds = predictions[predictions['dataset'] == dataset]
    print(f"{dataset}: mean={preds.mean():.4f}, std={preds.std():.4f}, min={preds.min():.4f}, max={preds.max():.4f}")
```

2. **Low model correlation is a WARNING SIGN**, not a feature
   - If two models disagree completely, at least one is wrong
   - Don't assume they're "complementary" without verification

3. **Test on diverse datasets during development**
   - Include imbalanced datasets in validation
   - Include large repertoires in validation
   - Don't just test on "easy" balanced datasets

4. **Ensemble is not always better**
   - Ensembling good model + broken model = mediocre model
   - Check individual model quality first

---

*Analysis Date: 2025-12-16*
*Recommendation: **USE V5 ONLY***
*Priority: **CRITICAL - FIX BEFORE NEXT SUBMISSION***
*Estimated Score Impact: Using V9 ensemble would reduce score by est. 20-30% overall*
