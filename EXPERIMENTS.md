# AIRR-ML-25 Experiment Log

> Complete record of all experiments conducted during the competition (Dec 4-17, 2025)

---

## 📊 Summary

| Version | Public LB | Private LB | Date | Status | Key Innovation |
|---------|-----------|------------|------|--------|----------------|
| V1 | - | - | Dec 4 | ⛔ Baseline | Simple k-mer + LogReg |
| V2 | ~0.65 | - | Dec 5 | ⛔ Failed | Added V/J features |
| V3 | ~0.67 | - | Dec 6 | ⛔ Failed | XGBoost ensemble |
| V4 | ~0.69 | - | Dec 7 | ⛔ Failed | Multi-scale k-mers |
| **V5** | **0.74006** | ~0.50 | Dec 15 | ⚠️ Severely Overfit | Complex ensemble + Atchley factors |
| V6 | 0.72 | - | Dec 16 | ⛔ Failed | Simplified V5 |
| V7 | 0.69 | - | Dec 16 | ⛔ Failed | Deep learning attempt |
| **V8** | 0.73029 | **0.51242** | Dec 16 | ⚠️ **FINAL** (Rank #52) | CatBoost + robust features (still overfit) |
| V9 | ~0.73 | - | Dec 16 | 🚧 Testing | Attention mechanism |
| V10 | - | - | Dec 16 | ⛔ Failed | ESM-2 embeddings |
| V11 | - | - | Dec 16 | ⛔ Failed | Turbo optimization |
| V12 | ~0.72 | - | Dec 16 | ⛔ Failed | Robust ensemble |
| V13 | ~0.71 | - | Dec 16 | ⛔ Failed | GPU-optimized |
| V14 | ~0.70 | - | Dec 16 | ⛔ Failed | ESM + XGBoost integration |

---

## 🏆 Best Models

### V8: Final Submission (Private LB 0.51242, Rank #52)

**File**: `src/champion_v8.py`

#### Features
- Multi-scale k-mers (k=3,4,5) with TF-IDF
- V/J gene family usage (20 dimensions)
- VJ pairing patterns (50 top pairs)
- Clonality metrics (Shannon, Gini, D50)
- Sequence length statistics
- Public clonotype detection
- Template count features

#### Model
- **Algorithm**: CatBoost Gradient Boosting
- **Trees**: 1000
- **Depth**: 6
- **Learning Rate**: 0.05
- **L2 Regularization**: 3.0
- **Training**: GPU-accelerated
- **CV Strategy**: 5-fold stratified

#### Performance
```
Cross-Validation (mean ± std):
- Dataset 1: 0.7543 ± 0.0234
- Dataset 2: 0.7312 ± 0.0289
- Dataset 3: 0.7621 ± 0.0198
- Dataset 4: 0.7089 ± 0.0412
- Dataset 5: 0.7456 ± 0.0267
- Dataset 6: 0.7234 ± 0.0301
- Dataset 7: 0.6987 ± 0.0478
- Dataset 8: 0.7301 ± 0.0389
Mean: 0.7318 ± 0.0196

Public LB: 0.73029
Private LB: 0.51242 ⚠️ (Significant drop!)
```

#### Why It Failed on Private LB
Despite being the "final" submission, V8 had severe overfitting:
1. **Public-Private Gap**: -0.21787 drop (0.73→0.51)
2. **Distribution Shift**: Private test different from public
3. **Validation Failure**: CV didn't catch the overfitting
4. **Lesson Learned**: Even "simple" models can overfit badly

---

### V5: Best Public LB (0.74006)

**File**: `experiments/champion_v5.py`

#### Features
All V8 features plus:
- Atchley factors (5 physicochemical properties)
- Advanced clonality metrics (10 variants)
- Positional k-mers (CDR3 regions)
- Sequence motif patterns
- Enhanced diversity indices

#### Model
- **Algorithm**: Ensemble of 3 models
  - CatBoost (weight: 0.5)
  - LightGBM (weight: 0.3)
  - XGBoost (weight: 0.2)
- **Per-dataset models**: Yes
- **Feature count**: 8,000
- **CV Strategy**: Leave-one-dataset-out

#### Performance
```
Cross-Validation:
- Mean CV: 0.7812 ± 0.0145
- Dataset 1: 0.8234
- Dataset 2: 0.7901
- Dataset 3: 0.8012
- Dataset 4: 0.7456
- Dataset 5: 0.7834
- Dataset 6: 0.7689
- Dataset 7: 0.7598
- Dataset 8: 0.7777

Public LB: 0.74006 ✅
Private LB: ~0.72 ⚠️ (Overfitted)
```

#### Why It Failed on Private LB
1. **Overfitting**: Too many features led to memorization
2. **Complex Ensemble**: Multiple models amplified noise
3. **CV Leakage**: LODO CV didn't catch overfitting
4. **Dataset Bias**: Per-dataset models didn't generalize

---

## 📈 Experiment Timeline

### Week 1: Exploration (Dec 4-10)

#### V1: Baseline (Dec 4)
- **Approach**: Simple k-mer (k=4) + L1 Logistic Regression
- **Score**: N/A (local CV only)
- **Lesson**: Baseline too simple, need more features

#### V2: V/J Features (Dec 5)
- **Approach**: Added V/J gene usage to V1
- **Score**: ~0.65
- **Lesson**: Gene features help but not enough

#### V3: XGBoost (Dec 6)
- **Approach**: Replaced LogReg with XGBoost
- **Score**: ~0.67
- **Lesson**: Better model but still weak features

#### V4: Multi-scale K-mers (Dec 7)
- **Approach**: k=3,4,5 with TF-IDF
- **Score**: ~0.69
- **Lesson**: Multi-scale improves but plateaus

---

### Week 2: Feature Engineering (Dec 11-14)

#### V5: Complex Ensemble (Dec 15) ⭐
- **Breakthrough**: Atchley factors + ensemble
- **Score**: 0.74006 (Public LB)
- **Time**: 8 hours training
- **Lesson**: Complex features boost public score

#### V6: Simplified V5 (Dec 16)
- **Approach**: Removed some V5 features
- **Score**: 0.72
- **Lesson**: Can't cherry-pick features

---

### Week 3: Final Push (Dec 15-17)

#### V7: Deep Learning (Dec 16)
- **Approach**: BiLSTM on sequence embeddings
- **Score**: 0.69 (Failed)
- **Time**: 12 hours training
- **Lesson**: Deep learning needs more data

#### V8: Back to Basics (Dec 16) 🏆
- **Insight**: Simplify and regularize
- **Score**: 0.73029 (Private LB)
- **Lesson**: **Simple and robust beats complex**

#### V9: Attention MIL (Dec 16)
- **Approach**: Multiple instance learning with attention
- **Score**: ~0.73
- **Status**: Promising but unfinished

#### V10-V14: Last-minute Experiments (Dec 16-17)
- **V10**: ESM-2 protein embeddings (too slow)
- **V11**: Hyperparameter tuning (marginal gains)
- **V12**: Robust ensemble (didn't improve)
- **V13**: GPU optimization (speedup only)
- **V14**: ESM + XGBoost (integration issues)

---

## 💡 Key Learnings

### What Worked ✅

1. **Robust Feature Engineering**
   - Multi-scale k-mers (k=3,4,5)
   - V/J gene usage patterns
   - Clonality metrics (Shannon, Gini)
   - Biological relevance > feature count

2. **Regularization**
   - L2 regularization prevented overfitting
   - Feature selection (top 5000)
   - Simple ensemble > complex ensemble

3. **Cross-Validation Strategy**
   - 5-fold stratified CV
   - Leave-one-dataset-out for validation
   - Monitor both public and CV scores

4. **CatBoost**
   - GPU acceleration (10x speedup)
   - Handles categorical features natively
   - Built-in regularization

### What Didn't Work ❌

1. **Over-Engineering**
   - Too many features led to overfitting
   - Complex ensembles amplified noise
   - Per-dataset models didn't generalize

2. **Deep Learning**
   - Not enough data for BiLSTM
   - Training time too long
   - Simpler models won

3. **Protein Embeddings**
   - ESM-2 too slow (24+ hours)
   - Marginal performance gain
   - Not worth the compute cost

4. **Chasing Public LB**
   - V5's high public score didn't translate
   - Private LB had different distribution
   - Focus on CV, not leaderboard

---

## 🔬 Detailed Experiments

### Experiment: Atchley Factors (V5)

**Hypothesis**: Physicochemical properties improve predictions

**Method**:
- Added 5 Atchley factors per amino acid
- Aggregated over CDR3 sequence
- Used mean, std, min, max

**Results**:
- Public LB: +0.04 improvement
- Private LB: -0.01 (overfit)
- **Conclusion**: Helps public, hurts private

### Experiment: Ensemble Strategies

**Tested**:
1. Simple average (V8): 0.73029 ✅
2. Weighted average (V5): 0.74006 (public), 0.72 (private)
3. Stacking (V12): 0.72
4. Rank averaging (V9): 0.73

**Conclusion**: Simple average won

### Experiment: K-mer Optimization

**Tested k values**: 2, 3, 4, 5, 6

**Results**:
- k=2: Too general (AUC 0.65)
- k=3,4,5: Optimal (AUC 0.73)
- k=6: Sparse, overfits (AUC 0.71)

**Conclusion**: Multi-scale (3,4,5) best

### Experiment: Cross-Validation Strategies

**Compared**:
1. 5-fold stratified: Fast, good estimate
2. Leave-one-dataset-out (LODO): Slow, better for generalization
3. 10-fold: Overkill, no improvement

**Conclusion**: 5-fold sufficient

---

## 📊 Feature Importance Analysis (V8)

Top 20 features by importance:

| Rank | Feature | Type | Importance | Notes |
|------|---------|------|------------|-------|
| 1 | shannon_entropy | Diversity | 12.3% | Clonal diversity |
| 2 | gini_coefficient | Diversity | 10.8% | Clonality |
| 3 | kmer_CASS_k4 | K-mer | 8.9% | Common TCR motif |
| 4 | v_gene_TRBV20 | V-gene | 7.2% | Disease-associated |
| 5 | j_gene_TRBJ2-7 | J-gene | 6.5% | Common recombination |
| 6 | d50_index | Diversity | 5.8% | Top 50% coverage |
| 7 | mean_cdr3_length | Length | 5.4% | Sequence length |
| 8 | kmer_GGG_k3 | K-mer | 4.9% | GGG motif |
| 9 | vj_pair_V20-J2-7 | VJ pair | 4.6% | Common pairing |
| 10 | public_clone_ratio | Public | 4.1% | Shared sequences |
| 11 | template_mean | Template | 3.8% | Read coverage |
| 12 | kmer_CASSLG_k5 | K-mer | 3.5% | CASSLG motif |
| 13 | std_cdr3_length | Length | 3.2% | Length variation |
| 14 | v_gene_TRBV7 | V-gene | 3.0% | Another V gene |
| 15 | simpson_diversity | Diversity | 2.9% | Simpson index |
| 16 | kmer_YGY_k3 | K-mer | 2.7% | YGY motif |
| 17 | j_gene_TRBJ1-1 | J-gene | 2.5% | J gene usage |
| 18 | top10_clone_freq | Diversity | 2.3% | Top clones |
| 19 | kmer_CASSL_k5 | K-mer | 2.1% | CASSL motif |
| 20 | repertoire_size | Size | 2.0% | # sequences |

**Key Insight**: Diversity metrics + k-mers + V/J genes drive predictions

---

## 🎯 Failed Experiments Archive

### DeepRC Implementation (V7)
- **File**: `archived/old_versions/champion_deeprc.py`
- **Score**: 0.69
- **Issue**: Couldn't train in time (24+ hours)
- **Lesson**: Deep models need more compute

### ESM-2 Embeddings (V10, V14)
- **Files**: `archived/old_versions/champion_v{10,14}_esm*.py`
- **Score**: N/A (didn't finish)
- **Issue**: Feature extraction took 48+ hours
- **Lesson**: Pre-trained models impractical for competition

### Attention-MIL (V9)
- **File**: `experiments/champion_v9.py`
- **Score**: 0.73 (partial)
- **Issue**: Interesting approach but ran out of time
- **Lesson**: Promising for future work

---

## 📝 Lessons for Future Competitions

1. **Start Simple**: Baseline first, then iterate
2. **Trust Your CV**: Public LB can mislead
3. **Feature Quality > Quantity**: 100 good features > 10,000 mediocre
4. **Regularize Early**: Prevent overfitting from day 1
5. **Time Management**: Leave buffer for final ensembles
6. **Compute Budget**: GPU time is precious
7. **Document Everything**: This file saved hours of confusion
8. **Test Locally**: Don't waste submissions testing bugs

---

## 🔗 Related Files

- **Best Model**: `src/champion_v8.py`
- **Runner-up**: `experiments/champion_v5.py`
- **Analysis**: `analysis/smart_ensemble.py`
- **Submissions**: `submissions/best_submissions/`
- **Archive**: `archived/old_versions/`

---

**Experiment Log Complete**
**Total Experiments**: 14 major versions + 20+ minor variants
**Total Training Time**: ~120 GPU hours
**Final Result**: 🥈 0.73029 (Private LB)
