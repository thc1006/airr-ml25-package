# AIRR-ML-25 Experiment Log

> TL;DR: 14 versions tested over 13 days. V8 (CatBoost) scored 0.51242 private LB (Rank #52). Severe overfitting despite regularization.

---

## Experiment Summary

| Version | Public LB | Private LB | Status | Key Change |
|---------|-----------|------------|--------|------------|
| V1-V4 | 0.65-0.69 | - | Failed | Baseline exploration (k-mers, V/J genes) |
| **V5** | **0.74006** | ~0.50 | Overfit | Complex ensemble (8000 features) |
| V6 | 0.72 | - | Failed | Simplified V5 |
| V7 | 0.69 | - | Failed | Deep learning (insufficient data) |
| **V8** | **0.73029** | **0.51242** | Best | CatBoost + regularization |
| V9 | 0.73 | - | Failed | Attention-MIL |
| V10-V14 | 0.70-0.73 | - | Failed | Last-minute attempts |

**Public-Private Gap**: -0.21787 (0.73 → 0.51) indicates severe overfitting.

---

## V8: Final Submission (Best Model)

**File**: `src/champion_v8.py`

**Features** (5000 selected):
- Multi-scale k-mers (k=3,4,5) + TF-IDF
- V/J gene usage + VJ pairing
- Clonality metrics (Shannon, Gini, D50)
- Public clonotype detection
- CDR3 sequence statistics

**Model**: CatBoost
- 1000 trees, depth 6, lr 0.05
- L2 regularization (λ=3.0)
- GPU-accelerated training
- 5-fold stratified CV

**Cross-Validation**: 0.7318 ± 0.0196

**Why it failed**: Distribution shift between public/private test sets. CV couldn't detect the overfitting.

---

## V5: Best Public LB (Overfitted)

**File**: `experiments/champion_v5.py`

**Features**: 8000 (too many)
- All V8 features
- Atchley physicochemical factors
- Advanced clonality variants
- Positional k-mers

**Model**: Ensemble (CatBoost 50% + LightGBM 30% + XGBoost 20%)

**CV**: 0.7812 ± 0.0145
**Public LB**: 0.74006
**Private LB**: ~0.50 (severe drop)

---

## Key Learnings

### What Worked
- Multi-scale k-mers (k=3,4,5)
- V/J gene patterns
- CatBoost GPU acceleration
- Biological domain knowledge

### What Failed
- Over-engineering (V5: 8000 features)
- Complex ensembles
- Deep learning (data insufficient)
- Chasing public leaderboard

### Critical Mistake

**Public-Private Distribution Mismatch**
- Public test was NOT representative
- CV failed to detect overfitting
- Should have trusted CV > leaderboard

---

## Experiment Timeline

**Week 1** (Dec 4-10): Baseline exploration (V1-V4)
**Week 2** (Dec 11-14): Feature engineering breakthrough (V5)
**Week 3** (Dec 15-17): Final push (V6-V14)

**Total Training Time**: ~120 GPU hours (RTX 5080)

---

## Feature Importance (V8)

Top 10 features:
1. Shannon entropy (12.3%)
2. Gini coefficient (10.8%)
3. K-mer: CASS (k=4) (8.9%)
4. V-gene: TRBV20 (7.2%)
5. J-gene: TRBJ2-7 (6.5%)
6. D50 index (5.8%)
7. Mean CDR3 length (5.4%)
8. K-mer: GGG (k=3) (4.9%)
9. VJ pair: V20-J2-7 (4.6%)
10. Public clone ratio (4.1%)

**Insight**: Diversity + k-mers + V/J genes drive predictions.

---

## Failed Experiments Archive

**Deep Learning** (V7, V10, V14):
- BiLSTM, ESM-2 embeddings
- Issue: Insufficient data, too slow

**Complex Ensembles** (V5, V12):
- Stacking, weighted averaging
- Issue: Overfitting amplification

**Attention Mechanisms** (V9):
- MIL with attention pooling
- Issue: Ran out of time

---

**Total Experiments**: 14 versions + 20+ variants
**Final Result**: Private LB 0.51242 (Rank #52/~500)
**Key Takeaway**: Simple + robust > complex + optimized
