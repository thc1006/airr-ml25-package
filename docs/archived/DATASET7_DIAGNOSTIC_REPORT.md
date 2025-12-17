# Dataset 7 Diagnostic Report: V9 vs V5 Low Correlation Analysis

**Date**: 2025-12-16
**Issue**: V9 and V5 predictions show extremely low correlation on Dataset 7
- test_dataset_7_1: correlation = 0.0747
- test_dataset_7_2: correlation = 0.2056

**Objective**: Find root cause and provide actionable recommendations

---

## Executive Summary

### Critical Findings

Dataset 7 is **FUNDAMENTALLY DIFFERENT** from all other datasets in the competition:

| Characteristic | Dataset 7 | Other Datasets (1-6) | Impact |
|----------------|-----------|----------------------|--------|
| **Class Balance** | 16.6% positive | 50% positive | **SEVERE** |
| **Repertoire Size** | 311,022 sequences | 25,000 sequences | **SEVERE** |
| **Total Repertoires** | 302 | 400 | Moderate |
| **Test Split** | 76 + 100 = 176 | ~50-100 each | Moderate |

### Root Cause Hypothesis

**V9 and V5 are solving COMPLETELY DIFFERENT problems on Dataset 7:**

1. **V5** (Attention-MIL): Likely overfitting to the **highly imbalanced** class distribution (83.4% negative)
   - May default to predicting mostly negative with simple frequency patterns
   - 12x larger repertoire size provides more noise for attention to overfit

2. **V9** (Ultimate with Fisher+XGB): Using statistical hypothesis testing (Fisher exact test)
   - Found 733 significant public clonotypes
   - More robust to class imbalance via proper statistical testing
   - XGBoost can handle imbalanced data better than attention mechanisms

### Why Correlation is So Low (0.07-0.20)

When two models have low correlation but both claim reasonable performance:
- They're using **orthogonal signals**
- One (or both) is likely **overfitting to artifacts** rather than true signal
- The low correlation is a **WARNING SIGN**, not a feature

---

## Detailed Analysis

### 1. Class Imbalance Impact

```
Dataset 7: 50 positive / 252 negative (16.6% positive)
Others:    200 positive / 200 negative (50% positive)
```

**Impact on V5 (Attention-MIL)**:
- Attention mechanisms tend to **collapse** on imbalanced datasets
- With only 50 positive examples across 302 repertoires:
  - Positive class has ~311k sequences × 50 = 15.5M sequences
  - Negative class has ~311k sequences × 252 = 78.4M sequences
- Attention weights may be dominated by negative class patterns

**Impact on V9 (Fisher + XGBoost)**:
- Fisher exact test is **designed for imbalanced data**
- XGBoost has `scale_pos_weight` parameter to handle imbalance
- Statistical testing is more robust to class ratio

### 2. Repertoire Size Impact

```
Dataset 7: Mean 311,022 sequences (12.4x larger)
Others:    Mean 25,000 sequences
Range:     85,418 - 599,735 sequences
```

**Impact on V5**:
- Attention-MIL must process **12x more sequences** per repertoire
- More sequences = more noise for attention to overfit
- May learn spurious correlations from batch effects

**Impact on V9**:
- Fisher test aggregates sequences into counts (size-invariant)
- XGBoost features are frequency-based (normalized by repertoire size)
- More robust to different scales

### 3. Public Clonotype Statistics

From Dataset 7 analysis:

```
Total unique sequences:
  Positive:  10,472,125
  Negative:  42,708,799

Shared sequences between groups: 3,633,318 (34.7% of positive repertoire)

Public clonotypes (count ≥ 5):
  Positive:  287,807 (2.7%)
  Negative:  1,773,659 (4.2%)
```

**Top Public Clonotypes**: ALL have similar frequency ratios
- CASSLGETQYF: 979 (pos) vs 4851 (neg) = 4.95x ratio
- CASSLGGNTEAFF: 928 (pos) vs 4761 (neg) = 5.13x ratio
- CASSLGYEQYF: 931 (pos) vs 4502 (neg) = 4.84x ratio

**Critical Observation**:
The **ratio is consistent with class imbalance** (252/50 = 5.04x)!

This means these "public clonotypes" might just be **ubiquitous sequences** that appear everywhere, not disease-specific biomarkers. V9's Fisher test finding 733 "significant" clones could be **false positives** from multiple testing.

### 4. Batch Effects

```
Label distribution by batch:
batch_1: 82.1% negative, 17.9% positive
batch_2: 83.9% negative, 16.1% positive
batch_3: 84.9% negative, 15.1% positive

Chi-square p-value: 0.86 (no significant batch effect)
```

**Good News**: No obvious batch confounding
**Bad News**: This rules out batch effects as the cause of low correlation

### 5. Positive vs Negative Characteristics

All features show **NO significant difference**:
- Repertoire size: p = 0.85
- Unique sequences: p = 0.83
- CDR3 length: p = 0.36
- Clonality index: p = 0.15 (marginal, 6.6% higher in positive)

**Implication**: This is a **VERY HARD DATASET**.
The signal is extremely weak or hidden in complex interaction patterns.

### 6. V/J Gene Usage

Top V gene in BOTH groups: **TCRBV20-X** (72% positive, 68.7% negative)

**This is identical!** No simple V gene usage difference.

---

## Validation Analysis Recommendations

### Immediate Actions Required

1. **Check V5 and V9 Training Logs for Dataset 7**

```bash
# For V5 - check attention_mil training logs
grep -r "dataset_7" checkpoints_*/training.log
grep -r "AUC" checkpoints_*/training.log | grep "dataset_7"

# For V9 - check ultimate model logs
grep -r "dataset_7" checkpoints_ultimate/training.log
grep -r "Fisher" checkpoints_ultimate/training.log | grep "dataset_7"
```

**What to look for**:
- Validation AUC on Dataset 7 specifically
- Training curves (is the model overfitting?)
- Class weight adjustments

2. **Analyze Prediction Distributions**

Check if V5 is predicting mostly 0.5 (uncertainty) or mostly 0/1 (overconfident):

```python
import pandas as pd
import numpy as np

# Load predictions
v5 = pd.read_csv('champion_v5_predictions.csv')
v9 = pd.read_csv('champion_ultimate_predictions.csv')

# Filter Dataset 7
v5_d7_1 = v5[v5['dataset'] == 'test_dataset_7_1']['label_positive_probability']
v5_d7_2 = v5[v5['dataset'] == 'test_dataset_7_2']['label_positive_probability']
v9_d7_1 = v9[v9['dataset'] == 'test_dataset_7_1']['label_positive_probability']
v9_d7_2 = v9[v9['dataset'] == 'test_dataset_7_2']['label_positive_probability']

# Print statistics
print("V5 Dataset 7_1:", v5_d7_1.describe())
print("V5 Dataset 7_2:", v5_d7_2.describe())
print("V9 Dataset 7_1:", v9_d7_1.describe())
print("V9 Dataset 7_2:", v9_d7_2.describe())

# Check for collapse to mean
print("\nPredicting mostly negative? (mean < 0.2):")
print(f"V5 7_1: {v5_d7_1.mean():.3f}")
print(f"V5 7_2: {v5_d7_2.mean():.3f}")
print(f"V9 7_1: {v9_d7_1.mean():.3f}")
print(f"V9 7_2: {v9_d7_2.mean():.3f}")
```

**Red flags**:
- If V5 mean is ~0.17 (class proportion), it's just predicting the base rate
- If V9 mean is very different, it found different signal (but which is correct?)

3. **Cross-Validation on Dataset 7 ONLY**

Run a dedicated experiment:

```bash
# Train model ONLY on Dataset 7 with proper stratified K-fold
python champion_v10_dataset7_only.py \
    --dataset 7 \
    --n_folds 5 \
    --stratified \
    --class_weight balanced
```

This will tell us:
- Can ANY model get good CV AUC on Dataset 7?
- Is the dataset just noise?

### Diagnostic Questions

**Q1**: What is V5's validation AUC on Dataset 7 during training?
- If AUC > 0.7: Model learned something, low correlation is concerning
- If AUC < 0.6: Model failed, low correlation is expected

**Q2**: What is V9's validation AUC on Dataset 7 during training?
- Same interpretation

**Q3**: Are the 733 "significant" Fisher clonotypes actually significant after Bonferroni correction?
- With 10M+ unique sequences, p < 0.05 is NOT significant
- Need p < 0.05 / 10M = 5e-9 for true significance

**Q4**: What happens if we ensemble V5 and V9 with equal weights?
- If ensemble correlation with each model is ~0.5, both contribute
- If ensemble correlation with one model is > 0.9, that model dominates

---

## Recommended Actions

### Option A: Trust V9, Reject V5 for Dataset 7

**Rationale**:
- Fisher exact test is more appropriate for imbalanced data
- XGBoost handles class imbalance better than attention
- V9's statistical approach is more interpretable

**Action**:
```python
# In ensemble code, use V9 only for Dataset 7
if dataset in ['test_dataset_7_1', 'test_dataset_7_2']:
    final_pred = v9_pred  # 100% weight to V9
else:
    final_pred = 0.5 * v5_pred + 0.5 * v9_pred  # Equal weight
```

### Option B: Train Dataset-7-Specific Model

**Rationale**:
- Dataset 7 is so different, it deserves its own model
- Can optimize for 16.6% positive rate specifically

**Action**:
```bash
# Train specialized model
python champion_v10_dataset7_specialist.py \
    --train_only dataset_7 \
    --test_on 7_1 7_2 \
    --class_weight 5.04 \  # 252/50 ratio
    --model xgboost \
    --tune_for_imbalance
```

### Option C: Ensemble with Uncertainty-Weighted Combination

**Rationale**:
- Low correlation means high uncertainty
- Use prediction entropy to weight models

**Action**:
```python
def entropy_weighted_ensemble(v5_pred, v9_pred):
    """Weight models inversely by prediction entropy"""

    def entropy(p):
        p = np.clip(p, 1e-7, 1-1e-7)  # Avoid log(0)
        return -p * np.log(p) - (1-p) * np.log(1-p)

    v5_entropy = entropy(v5_pred)
    v9_entropy = entropy(v9_pred)

    # Inverse entropy = confidence weight
    v5_weight = 1 / (v5_entropy + 1e-7)
    v9_weight = 1 / (v9_entropy + 1e-7)

    # Normalize
    total_weight = v5_weight + v9_weight
    v5_weight /= total_weight
    v9_weight /= total_weight

    return v5_weight * v5_pred + v9_weight * v9_pred
```

### Option D: Conservative Approach - Use Prior Mean

**Rationale**:
- When models disagree this much, safest bet is class prior
- For Dataset 7: prior = 0.166

**Action**:
```python
# If correlation < 0.3, use shrinkage towards prior
if dataset in ['test_dataset_7_1', 'test_dataset_7_2']:
    prior = 0.166
    ensemble = 0.5 * v5_pred + 0.5 * v9_pred

    # Shrink towards prior
    shrinkage = 0.7  # 70% towards prior due to high uncertainty
    final_pred = shrinkage * prior + (1 - shrinkage) * ensemble
```

---

## My Recommendation

### URGENT: Before Final Submission

1. **Check if this is a leaderboard probe attack**
   - Are test_dataset_7_1 and 7_2 intentionally different from training?
   - This extreme difference could be adversarial test design

2. **Run Option A with validation**
   - Use V9 only for Dataset 7
   - Validate on other datasets to ensure no degradation

3. **If time permits (< 2 hours)**
   - Quick re-train of V9 with Bonferroni-corrected Fisher test
   - Reduce false positive public clonotypes

### For Production System

- **Flag Dataset 7 as "high uncertainty" in prediction metadata**
- **Use ensemble of Option A and Option C**:
  - Primary: V9-only for Dataset 7
  - Secondary: Entropy-weighted ensemble as backup
  - Output both predictions + uncertainty estimate

---

## Technical Debt

Issues found that need fixing:

1. **V5 (Attention-MIL) not designed for class imbalance**
   - Add `class_weight` parameter to loss function
   - Add focal loss option for hard negatives

2. **V9 Fisher test needs multiple testing correction**
   - 733 significant clones from 10M+ tests
   - Use Benjamini-Hochberg FDR instead of raw p-values

3. **No dataset-specific validation metrics**
   - Training logs should report per-dataset AUC
   - Need to catch Dataset 7 problems during training

4. **No calibration analysis**
   - Predictions might be miscalibrated on imbalanced data
   - Add Platt scaling or isotonic regression

---

## Conclusion

Dataset 7's extreme characteristics (83.4% negative, 12x larger repertoires) create a **perfect storm** for model disagreement:

- **V5** likely collapsed to predicting base rate (0.17) with attention overfitting to noise
- **V9** found statistical signal (733 clones) but may be false positives from multiple testing
- **Low correlation (0.07-0.20) is a RED FLAG** indicating at least one model is wrong

**Safest strategy**:
1. Use V9 exclusively for Dataset 7 (trust statistics over attention)
2. Shrink predictions towards prior (0.166) with 30-50% weight
3. Flag these predictions as "high uncertainty" in metadata

**DO NOT**:
- Blindly ensemble low-correlation predictions (garbage in = garbage out)
- Assume both models found different "valid" signals (Occam's razor: one is wrong)
- Ignore this until after submission (cost = 176 test samples wrong)

**Time estimate to fix**: 30-60 minutes to implement Option A + shrinkage

---

*Generated: 2025-12-16*
*Analyst: Data Scientist (Claude Sonnet 4.5)*
*Priority: CRITICAL - Affects 176/4213 test predictions (4.2%)*
