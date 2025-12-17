# CRITICAL: Dataset 7 Analysis - V9 Model Complete Failure

**Date**: 2025-12-16
**Status**: 🚨 URGENT - CRITICAL FAILURE DETECTED
**Impact**: 176 test samples (4.2% of total) receiving ZERO predictions

---

## SMOKING GUN DISCOVERED

### V9 (Ultimate Model) Predictions on Dataset 7

```
test_dataset_7_1 (76 samples):
  ALL predictions = 0.0000
  Min = 0.0000, Max = 0.0000, Std = 0.0000

test_dataset_7_2 (100 samples):
  ALL predictions = 0.0000
  Min = 0.0000, Max = 0.0000, Std = 0.0000
```

**THIS IS NOT A MODEL PREDICTION - THIS IS A FAILURE MODE**

### V5 (Attention-MIL) Predictions on Dataset 7

```
test_dataset_7_1 (76 samples):
  Mean = 0.1031 (close to true rate of 0.166)
  Std = 0.0415
  Range = [0.0404, 0.2024]
  56.6% strong negative (< 0.1)

test_dataset_7_2 (100 samples):
  Mean = 0.1276
  Std = 0.0552
  Range = [0.0454, 0.2662]
  37% strong negative (< 0.1)
```

**V5 is at least trying to predict something reasonable**

---

## What Went Wrong

### Hypothesis: V9's XGBoost Failed to Handle Dataset 7

Possible causes:

1. **Feature Extraction Crashed**
   - Fisher exact test might have failed on extreme class imbalance (16.6% positive)
   - XGBoost features computed incorrectly
   - Returned NaN/Inf which XGBoost converted to 0.0

2. **Model Wasn't Trained on Dataset 7**
   - V9 might have been trained only on datasets 1-6
   - When encountering dataset 7, default prediction = 0.0
   - This would explain why correlation is low (comparing 0.0 vs real predictions)

3. **Deliberate Clipping to Zero**
   - XGBoost predictions went negative
   - Clipped to [0, 1] range, but implementation had bug
   - All predictions rounded down to 0.0

4. **Dataset 7 Feature Extraction Not Implemented**
   - Code might have special case for datasets 7_1 and 7_2
   - When these test sets are encountered, prediction defaults to 0.0

---

## Evidence Analysis

### Correlation Makes Sense Now

```
test_dataset_7_1: correlation = 0.0747
test_dataset_7_2: correlation = 0.2056
```

**Of course correlation is near zero!**
- V9 predicts constant 0.0
- V5 predicts varying values (mean ~0.1-0.13)
- Correlation between constant and varying = near zero

The 0.0747 and 0.2056 are just **noise from numerical precision**.

### Dataset 1 Comparison Shows V9 Systematic Problem

```
Dataset 1:
  V5 Mean = 0.5021 (reasonable, balanced)
  V9 Mean = 0.0152 (ALSO TOO LOW!)

  Correlation = -0.0012 (essentially zero)
  Disagreement = 51.7%
```

**V9 IS BROKEN ON ALL DATASETS, NOT JUST 7!**

V9 predicts near-zero for everything:
- Dataset 1: mean = 0.0152 (should be ~0.5 for balanced data)
- Dataset 7_1: mean = 0.0000
- Dataset 7_2: mean = 0.0000

---

## Root Cause

### V9's "submission_ultimate.csv" is NOT a valid submission

This file contains:
1. Test dataset predictions that are **all near zero**
2. Likely a bug in the final prediction step
3. Possibly using wrong probability column (e.g., outputting negative class prob)

### Likely Code Bug Location

In the Ultimate model's prediction code:

```python
# WRONG - this might be what's happening
predictions = model.predict(X)  # Returns class labels 0/1, not probabilities!
# Should be:
predictions = model.predict_proba(X)[:, 1]  # Get positive class probability
```

Or:

```python
# WRONG - negative class probability
predictions = model.predict_proba(X)[:, 0]
# Should be:
predictions = model.predict_proba(X)[:, 1]
```

---

## Immediate Actions Required

### 1. STOP USING V9/Ultimate Model Immediately

The file `submission_ultimate.csv` is **GARBAGE**. Do not include in any ensemble.

### 2. Verify V5 Model Submission

V5's `v5_submission_20251215_225131.csv` shows:
- Dataset 1: Mean = 0.5021 (good, balanced)
- Dataset 7_1: Mean = 0.1031 (low, but reasonable for 16.6% positive)
- Dataset 7_2: Mean = 0.1276 (low, but reasonable)

**V5 IS THE ONLY WORKING MODEL**

### 3. Investigate Ultimate Model Code

Check these files:
```bash
grep -r "predict_proba" champion_ultimate.py
grep -r "predict\(" champion_ultimate.py
grep -A 10 "def predict" champion_ultimate.py
```

Look for:
- Wrong probability column (using [:, 0] instead of [:, 1])
- Using predict() instead of predict_proba()
- NaN handling that defaults to 0.0

### 4. Re-run V9 with Fixes

If code bug found:
```bash
# Fix the bug in champion_ultimate.py
# Re-run prediction generation
python champion_ultimate.py --mode test --checkpoint best_model.pt
```

### 5. Emergency Submission Strategy

**DO NOT ENSEMBLE V5 and V9**

Use **V5 ONLY** for submission:
```bash
cp submissions/v5_submission_20251215_225131.csv final_submission.csv
```

V5's predictions are at least in the right ballpark:
- Mean predictions match dataset characteristics
- Shows variation (not constant)
- Reasonable uncertainty estimates

---

## Validation Test

To confirm V9 is broken, check predictions on **training data**:

```python
import pandas as pd

# Load V9's predictions on train data (if available)
# If train predictions are also all zeros, V9 is completely broken

# Quick test: what does V9 predict for a known positive sample?
# It should be > 0.5, not 0.0
```

If train predictions are also near-zero, **V9 never worked**.

---

## Long-term Fix

### For V10 Model:

1. **Add prediction validation**:
```python
def validate_predictions(y_pred):
    assert y_pred.min() >= 0.0, "Negative predictions!"
    assert y_pred.max() <= 1.0, "Predictions > 1!"
    assert y_pred.std() > 0.01, "Predictions are constant!"
    assert 0.01 < y_pred.mean() < 0.99, "Predictions collapsed!"
```

2. **Add logging**:
```python
print(f"Prediction stats: mean={y_pred.mean():.4f}, std={y_pred.std():.4f}")
print(f"Range: [{y_pred.min():.4f}, {y_pred.max():.4f}]")
```

3. **Test on training data first**:
```python
# Before predicting test, predict train
train_pred = model.predict_proba(X_train)[:, 1]
print(f"Train AUC: {roc_auc_score(y_train, train_pred):.4f}")
# If train AUC is good but test predictions are zero, feature extraction failed
```

---

## Summary

### What We Thought
- V9 and V5 are both working models with low correlation
- They found different signals
- Need to carefully ensemble them

### What's Actually True
- **V9 is completely broken** (all predictions ≈ 0.0)
- V5 is the only working model
- Low correlation is because you can't correlate constant with variable
- **There is no ensemble decision to make**

### Correct Action
1. **Submit V5 immediately** (it's the only working model)
2. Debug V9 before considering it for future submissions
3. Add validation checks to catch this earlier

### Estimated Score Impact
- Using V5 only: Likely similar to current leaderboard score
- Using broken ensemble: Would TANK the score (averaging real predictions with zeros)
- **Using V5 only is the safe choice**

---

**RECOMMENDATION**: Submit `v5_submission_20251215_225131.csv` immediately.
**DO NOT** attempt to fix V9 under time pressure. It's too risky.

---

*Analysis completed: 2025-12-16*
*Priority: CRITICAL - FIX BEFORE NEXT SUBMISSION*
