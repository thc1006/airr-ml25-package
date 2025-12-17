# AIRR-ML-25 Competition Summary

> TL;DR: Ranked #52/~500 with 0.51242 private LB. Public-private gap of -0.21787 shows severe overfitting despite regularization. Key lesson: public leaderboard can be misleading.

---

## Final Results

| Metric | Score | Rank | Note |
|--------|-------|------|------|
| **Private LB** | **0.51242** | **#52/~500** | Final ranking |
| Public LB | 0.73029 | - | V8 model |
| Public LB (peak) | 0.74006 | - | V5 overfitted |

**Public-Private Gap**: -0.21787 (0.73 → 0.51) indicates severe distribution shift.

**Critical Lesson**: Despite strong public scores (0.73-0.74), the massive drop on private leaderboard demonstrates that:
1. Public LB can be misleading - high public scores don't guarantee private performance
2. Cross-validation alone isn't enough - need better strategies to detect overfitting
3. Distribution shift is real - private test set characteristics differed significantly
4. Simpler models didn't help - even "robust" V8 model overfitted severely

---

## Competition Overview

**Challenge**: Predict immune states (Task A) and identify disease-associated sequences (Task B) from B/T-cell receptor repertoires.

**Hosted by**: University of Oslo & Kaggle
**Prize**: $5,000 + Nature Methods authorship
**Duration**: December 4-17, 2025
**Participants**: ~500 teams

### Dataset

**Training**: 8 datasets spanning COVID-19, cancer, autoimmune diseases (T-cells, B-cells)
**Testing**: 11 test datasets, 4,213 repertoires, ~19.94 GB total
**Data**: CDR3 sequences (junction_aa), V/J/D gene calls, read counts (templates)

### Evaluation

**Task A**: ROC-AUC for immune state prediction
**Task B**: Jaccard Similarity for top 50,000 disease-associated sequences
**Combined**: Weighted average of both tasks

---

## Approach

### V8 Model (Best Submission)

**Algorithm**: CatBoost Gradient Boosting (GPU-accelerated)

**Features** (5000 total):
- Multi-scale k-mers (k=3,4,5) with TF-IDF
- V/J gene usage + VJ pairing patterns
- Clonality metrics (Shannon entropy, Gini coefficient, D50 index)
- Public clonotype detection
- CDR3 sequence statistics

**Training**:
- Per-dataset models with 5-fold stratified CV
- L2 regularization (λ=3.0), early stopping
- CatBoost: 1000 trees, depth=6, lr=0.05
- Simple average ensemble of 5 CV folds

**Cross-Validation**: 0.7318 ± 0.0196 (8 datasets)

**Why it failed on private LB**: Distribution shift between public/private test sets. CV couldn't detect the overfitting.

---

## Results Analysis

### Leaderboard Progression

| Version | Public LB | Private LB | Delta | Status |
|---------|-----------|------------|-------|--------|
| V1-V4 | 0.65-0.69 | - | - | Baseline exploration |
| **V5** | **0.74006** | ~0.50 | -0.24 | Severe overfitting |
| V8 | 0.73029 | **0.51242** | -0.21787 | Best (still overfit) |
| V9-V14 | 0.70-0.73 | - | - | Failed experiments |

### Cross-Validation Results (V8)

| Dataset | CV AUC | Std Dev | Notes |
|---------|--------|---------|-------|
| Dataset 1 | 0.7543 | 0.0234 | COVID-19 T-cells |
| Dataset 2 | 0.7312 | 0.0289 | Cancer B-cells |
| Dataset 3 | 0.7621 | 0.0198 | Autoimmune T-cells (easiest) |
| Dataset 7 | 0.6987 | 0.0478 | Autoimmune B-cells (hardest) |
| **Mean** | **0.7318** | **0.0196** | Overall performance |

**High variance in Datasets 4,7,8 indicated distribution shift - but we missed this warning sign.**

---

## What Worked

### 1. Biological Feature Engineering
Hand-crafted features based on immunology outperformed embeddings:
- V/J gene usage reflects somatic recombination
- Clonality metrics capture immune response
- K-mers identify conserved motifs

**Impact**: +15% over baseline

### 2. Regularization
Prevented some (but not all) overfitting:
- L2 regularization (λ=3.0)
- Feature selection (top 5000)
- Cross-validation early stopping

**Impact**: +2% on private LB vs V5 (still insufficient)

### 3. CatBoost GPU Training
Fast iteration enabled extensive experiments:
- 10x speedup over CPU
- 5 minutes per full model
- 20+ experiments in final 48 hours

### 4. Simple Ensemble
Average of 5 CV folds outperformed complex stacking:
```python
prediction = mean([model_fold_i.predict(X) for i in range(5)])
```

**Impact**: +1% over single model

---

## What Didn't Work (Critical Lessons)

### 1. Over-Engineering (V5)
**Mistake**: More features ≠ better performance
- 8,000 features (V5) vs 5,000 (V8)
- Atchley physicochemical factors overfitted
- Per-dataset models didn't generalize

**Public LB**: 0.74006 (best)
**Private LB**: ~0.50 (catastrophic drop)

**Lesson**: Feature quality > quantity. Overfitting isn't always obvious from CV.

### 2. Deep Learning Attempts (V7, V9, V10)
**Mistake**: Forcing neural networks on insufficient data
- BiLSTM: 0.69 (vs 0.73 for CatBoost)
- Attention-MIL: Promising but unfinished
- ESM-2 embeddings: 48+ hours training, no improvement

**Lesson**: Not enough data for deep learning. Gradient boosting wins on tabular data.

### 3. Complex Ensembles (V12)
**Mistake**: Stacking and meta-learning added noise
- 3-model ensemble (CatBoost+LightGBM+XGBoost)
- Logistic regression meta-learner
- Weighted averaging with grid search

**Result**: Worse than simple average

**Lesson**: Simple average beats complex stacking when models are correlated.

### 4. Chasing Public Leaderboard
**Critical Mistake**: Optimizing for public LB backfired
- V5: 0.74 public → 0.50 private (-0.24 = 32% drop)
- V8: 0.73 public → 0.51 private (-0.22 = 30% drop)

**Lesson**: Public LB is a trap. Trust cross-validation over leaderboard. But even CV failed to detect distribution shift.

### 5. Ignoring High Variance Warning Signs
**Mistake**: Dataset 7 showed high CV variance (0.0478) but we ignored it
- Should have investigated why some datasets were unstable
- Should have used leave-one-dataset-out validation
- Should have analyzed train-test distribution differences

**Lesson**: High variance is a red flag for distribution shift.

---

## Key Insights for Future Competitions

### Critical Realizations

1. **Public LB Can Be Completely Misleading**
   - 0.74 → 0.50 is not a small drop, it's catastrophic
   - Public test set was NOT representative
   - No amount of CV detected this issue

2. **Cross-Validation Isn't a Silver Bullet**
   - Our CV: 0.7318 ± 0.0196 (looked great)
   - Private LB: 0.51242 (disaster)
   - CV only validates on similar distribution

3. **Regularization Isn't Enough**
   - We used L2 regularization, feature selection, early stopping
   - Still overfitted by 30%
   - Need distribution-aware validation

4. **Simplification Doesn't Guarantee Generalization**
   - V8 had fewer features than V5
   - Still dropped 30% on private LB
   - Simple ≠ robust

### What We Should Have Done

1. **Adversarial Validation**
   - Train classifier to distinguish train vs test
   - High AUC indicates distribution shift
   - We never checked this

2. **Leave-One-Dataset-Out Validation**
   - Simulate distribution shift
   - Test on completely unseen dataset
   - We did 5-fold CV within each dataset instead

3. **Distribution Analysis**
   - Compare train vs test feature distributions
   - Check for covariate shift
   - We assumed datasets were similar

4. **Conservative Submission Strategy**
   - Submit model with lowest CV variance
   - Avoid chasing public LB improvements
   - We chased 0.74 instead

---

## Technical Specifications

**Hardware**:
- CPU: AMD Ryzen 7 7800X3D (8 cores)
- GPU: NVIDIA RTX 5080 (16GB VRAM)
- RAM: 32GB DDR5

**Software**:
- Python 3.10.12
- CatBoost 1.2.2, scikit-learn 1.3.0
- pandas 2.0.0, NumPy 1.24.0

**Training Time**:
- V8: 3.3 hours (5 folds × 8 datasets)
- V5: 8 hours (complex ensemble)
- Total: ~120 GPU hours

---

## References

### Competition
- [Kaggle Page](https://www.kaggle.com/competitions/adaptive-immune-profiling-challenge-2025)
- [Code Template](https://github.com/uio-bmi/predict-airr)
- [Pre-registered Protocol](https://github.com/uio-bmi/adaptive_immune_profiling_challenge_2025/blob/main/registered_report.pdf)

### Domain Knowledge
- [AIRR Mining State-of-the-art](https://www.sciencedirect.com/science/article/pii/S2452310020300524)
- [immuneML Platform](https://pmc.ncbi.nlm.nih.gov/articles/PMC10312379/)
- [Modern Hopfield Networks](https://doi.org/10.1101/2020.04.12.038158)

---

**Final Score**: 0.51242 (Private LB)
**Final Rank**: #52 / ~500
**Key Takeaway**: Honest documentation of failure is as valuable as success stories.
