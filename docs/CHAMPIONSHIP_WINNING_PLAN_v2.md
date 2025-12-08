# AIRR-ML-25 Championship Winning Plan v2.0

> **Generated**: 2025-12-07
> **Current Score**: 0.65176 (Enhanced v1)
> **Target Score**: 0.82+ (Top 3 requires ~0.81+)
> **Gap to Close**: ~0.17 (26% improvement needed)
> **Days Remaining**: 10 days (deadline: Dec 17, 06:59 UTC)

---

## Executive Summary

### Current Leaderboard (2025-12-07)

| Rank | Team | Score |
|------|------|-------|
| 1 | SajayR | 0.82518 |
| 2 | GROZD | 0.81998 |
| 3 | GoBlue | 0.80992 |
| ... | ... | ... |
| Us | Enhanced v1 | 0.65176 |

### Critical Insight

**We have a fundamental flaw in Task B (sequence identification)** that is likely costing us 0.10-0.15 points. The official baseline achieves 0.72866 with simple k=3 k-mers + LogReg, yet we're at 0.65176 with a more complex approach.

### Root Cause Analysis

| Issue | Impact | Priority |
|-------|--------|----------|
| Task B: Using k-mer frequency instead of binary presence | -0.10 to -0.15 | **P0** |
| Task B: Using ensemble importance instead of LogReg coefficients | -0.05 to -0.10 | **P0** |
| Task A: Missing V/J gene usage features | -0.03 to -0.05 | P1 |
| Task A: No attention-based aggregation | -0.03 to -0.05 | P2 |
| Task A: No protein language model embeddings | -0.02 to -0.04 | P3 |

---

## Part 1: Immediate Fixes (Day 1-2)

### 1.1 Fix Task B Sequence Identification [P0]

**Current Implementation (WRONG)**:
```python
# From our current code - WRONG APPROACH
def _identify_sequences(self, train_dir_path, top_k=50000):
    # Uses ensemble feature importance
    # Uses k-mer FREQUENCY counts
    for kmer in sequence:
        score += kmer_importance[kmer] * kmer_count  # WRONG!
```

**Correct Implementation (Official Baseline)**:
```python
def identify_associated_sequences(self, train_dir_path, top_k=50000):
    # Get LogReg coefficients (NOT ensemble importance)
    logreg_coef = self.model.logreg_model.coef_[0]
    scaler = self.model.scaler
    unscaled_coef = logreg_coef / scaler.scale_

    # Score using BINARY k-mer presence
    for seq in sequences:
        score = 0.0
        seen_kmers = set()
        for i in range(len(seq) - k + 1):
            kmer = seq[i:i+k]
            if kmer in kmer_to_coef and kmer not in seen_kmers:
                score += kmer_to_coef[kmer]  # Add once per unique k-mer
                seen_kmers.add(kmer)
        scores.append(score)
```

**Expected Impact**: +0.10 to +0.15

### 1.2 Validate Submission Format

```python
# Ensure correct format
assert submission_df.shape[0] == 404213
assert list(submission_df.columns) == ['ID', 'dataset', 'label_positive_probability',
                                        'junction_aa', 'v_call', 'j_call']
# Match ID order with sample_submissions.csv
sample_order = pd.read_csv('sample_submissions.csv')['ID'].tolist()
submission_df = submission_df.set_index('ID').loc[sample_order].reset_index()
```

---

## Part 2: Enhanced Feature Engineering (Day 3-4)

### 2.1 Multi-Scale K-mer Features

Based on literature (MotifBoost, KDDC), multi-scale k-mers capture different pattern levels:

```python
K_MER_SIZES = [3, 4, 5]  # k=3: motifs, k=4,5: longer patterns

def extract_multiscale_kmers(sequences, k_values=[3, 4, 5]):
    features = {}
    for k in k_values:
        kmer_counts = Counter()
        for seq in sequences:
            for i in range(len(seq) - k + 1):
                kmer_counts[seq[i:i+k]] += 1
        # Normalize by total k-mers
        total = sum(kmer_counts.values())
        features.update({f'k{k}_{kmer}': count/total
                        for kmer, count in kmer_counts.items()})
    return features
```

### 2.2 V/J Gene Usage Features [HIGH IMPACT]

Literature strongly supports V/J gene usage as predictive:
- Different diseases show distinct V/J gene biases
- VJ pairing patterns are disease-specific

```python
def extract_vj_features(repertoire_df):
    features = {}

    # V gene usage (normalized)
    v_counts = repertoire_df['v_call'].value_counts(normalize=True)
    for v_gene, freq in v_counts.items():
        features[f'v_{v_gene}'] = freq

    # J gene usage (normalized)
    j_counts = repertoire_df['j_call'].value_counts(normalize=True)
    for j_gene, freq in j_counts.items():
        features[f'j_{j_gene}'] = freq

    # VJ pair usage (top 100 pairs)
    vj_pairs = repertoire_df.groupby(['v_call', 'j_call']).size()
    vj_pairs = vj_pairs / vj_pairs.sum()
    for (v, j), freq in vj_pairs.nlargest(100).items():
        features[f'vj_{v}_{j}'] = freq

    return features
```

**Expected Impact**: +0.03 to +0.05

### 2.3 Clonality and Diversity Metrics

From immunology literature, these metrics distinguish healthy vs disease states:

```python
import numpy as np
from scipy.stats import entropy

def extract_diversity_features(repertoire_df):
    # Clone frequencies (by junction_aa)
    clone_counts = repertoire_df['junction_aa'].value_counts()
    frequencies = clone_counts / clone_counts.sum()

    features = {
        # Shannon entropy
        'shannon_entropy': entropy(frequencies),

        # Gini-Simpson index
        'gini_simpson': 1 - np.sum(frequencies ** 2),

        # Clonality (normalized entropy)
        'clonality': 1 - entropy(frequencies) / np.log(len(frequencies)),

        # Richness (unique clones)
        'richness': len(clone_counts),

        # D50 (clones accounting for 50% of repertoire)
        'D50': calculate_d50(frequencies),

        # Top clone frequency
        'top_clone_freq': frequencies.iloc[0],

        # Top 10 clones frequency
        'top10_clone_freq': frequencies.iloc[:10].sum(),
    }
    return features

def calculate_d50(frequencies):
    cumsum = frequencies.sort_values(ascending=False).cumsum()
    return (cumsum < 0.5).sum() + 1
```

**Expected Impact**: +0.02 to +0.03

### 2.4 CDR3 Length Distribution

```python
def extract_length_features(repertoire_df):
    lengths = repertoire_df['junction_aa'].str.len()
    return {
        'cdr3_len_mean': lengths.mean(),
        'cdr3_len_std': lengths.std(),
        'cdr3_len_median': lengths.median(),
        'cdr3_len_q25': lengths.quantile(0.25),
        'cdr3_len_q75': lengths.quantile(0.75),
        'cdr3_len_mode': lengths.mode().iloc[0] if len(lengths.mode()) > 0 else 0,
        'cdr3_len_skew': lengths.skew(),
        'cdr3_len_kurt': lengths.kurtosis(),
    }
```

---

## Part 3: Advanced Model Architecture (Day 5-7)

### 3.1 Stacking Ensemble

Based on Kaggle champion patterns and Mal-ID paper (AUROC 0.959):

```
Level 1 (Base Models):
├── XGBoost (GPU)
│   ├── max_depth: 6, learning_rate: 0.05
│   └── n_estimators: 500, device: 'cuda'
├── LightGBM (GPU)
│   ├── num_leaves: 63, learning_rate: 0.05
│   └── n_estimators: 500, device: 'gpu'
├── CatBoost (GPU)
│   ├── depth: 6, learning_rate: 0.05
│   └── iterations: 500, task_type: 'GPU'
├── L1 Logistic Regression
│   └── C: 0.1, solver: 'saga'
└── Random Forest
    └── n_estimators: 200, max_depth: 10

Level 2 (Meta-Learner):
└── Ridge Regression (CV-optimized alpha)
```

```python
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import RidgeCV

class StackingEnsemble:
    def __init__(self, base_models, meta_model=None):
        self.base_models = base_models
        self.meta_model = meta_model or RidgeCV(alphas=[0.1, 1, 10])

    def fit(self, X, y, n_folds=5):
        kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        meta_features = np.zeros((len(X), len(self.base_models)))

        for i, model in enumerate(self.base_models):
            for train_idx, val_idx in kfold.split(X, y):
                model.fit(X[train_idx], y[train_idx])
                meta_features[val_idx, i] = model.predict_proba(X[val_idx])[:, 1]

        self.meta_model.fit(meta_features, y)

        # Retrain base models on full data
        for model in self.base_models:
            model.fit(X, y)

        return self

    def predict_proba(self, X):
        meta_features = np.column_stack([
            model.predict_proba(X)[:, 1] for model in self.base_models
        ])
        return self.meta_model.predict(meta_features)
```

**Expected Impact**: +0.03 to +0.05

### 3.2 Per-Dataset Fine-tuning

Different datasets may have different distributions:

```python
class DatasetAwareEnsemble:
    def __init__(self, base_ensemble):
        self.unified_model = base_ensemble
        self.dataset_models = {}

    def fit(self, X, y, dataset_ids):
        # Train unified model on all data
        self.unified_model.fit(X, y)

        # Train per-dataset models
        for dataset_id in np.unique(dataset_ids):
            mask = dataset_ids == dataset_id
            if mask.sum() >= 50:  # Minimum samples
                self.dataset_models[dataset_id] = clone(self.unified_model)
                self.dataset_models[dataset_id].fit(X[mask], y[mask])

        return self

    def predict_proba(self, X, dataset_id):
        unified_pred = self.unified_model.predict_proba(X)
        if dataset_id in self.dataset_models:
            dataset_pred = self.dataset_models[dataset_id].predict_proba(X)
            # Blend 70% unified + 30% dataset-specific
            return 0.7 * unified_pred + 0.3 * dataset_pred
        return unified_pred
```

---

## Part 4: Deep Learning (Day 8-9)

### 4.1 DeepRC-Style Attention Mechanism

From NeurIPS 2020 paper, Modern Hopfield Networks achieve state-of-the-art:

```python
import torch
import torch.nn as nn

class DeepRCLite(nn.Module):
    """
    Simplified DeepRC architecture for immune repertoire classification.
    Key innovation: attention-based sequence aggregation.
    """
    def __init__(self, input_dim=21, hidden_dim=64, num_heads=4):
        super().__init__()

        # Sequence encoder (1D CNN)
        self.seq_encoder = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
        )

        # Attention pooling (Modern Hopfield-inspired)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim))

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, sequences, mask=None):
        # sequences: (batch, n_sequences, seq_len, features)
        batch_size, n_seq, seq_len, n_feat = sequences.shape

        # Encode each sequence
        x = sequences.view(-1, n_feat, seq_len)  # (batch*n_seq, features, seq_len)
        x = self.seq_encoder(x)  # (batch*n_seq, hidden, 1)
        x = x.squeeze(-1).view(batch_size, n_seq, -1)  # (batch, n_seq, hidden)

        # Attention-based aggregation
        query = self.query.expand(batch_size, -1, -1)
        attn_out, attn_weights = self.attention(query, x, x, key_padding_mask=mask)

        # Classify
        out = self.classifier(attn_out.squeeze(1))
        return out, attn_weights
```

**Expected Impact**: +0.03 to +0.05

### 4.2 ESM-2 Protein Language Model Embeddings

Based on Nature 2023 and TEPCAM research:

```python
from transformers import EsmModel, EsmTokenizer
import torch

class ESM2Encoder:
    def __init__(self, model_name="facebook/esm2_t6_8M_UR50D"):
        self.tokenizer = EsmTokenizer.from_pretrained(model_name)
        self.model = EsmModel.from_pretrained(model_name)
        self.model.eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda()

    @torch.no_grad()
    def encode_sequences(self, sequences, batch_size=32):
        embeddings = []
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i:i+batch_size]
            inputs = self.tokenizer(batch, return_tensors="pt", padding=True)
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            outputs = self.model(**inputs)
            # Mean pooling
            emb = outputs.last_hidden_state.mean(dim=1)
            embeddings.append(emb.cpu().numpy())
        return np.vstack(embeddings)

    def encode_repertoire(self, repertoire_df, sample_size=1000):
        # Sample sequences for efficiency
        seqs = repertoire_df['junction_aa'].dropna().tolist()
        if len(seqs) > sample_size:
            seqs = np.random.choice(seqs, sample_size, replace=False)

        emb = self.encode_sequences(seqs)

        # Aggregate: mean and max pooling
        return np.concatenate([emb.mean(axis=0), emb.max(axis=0)])
```

**Expected Impact**: +0.04 to +0.06

---

## Part 5: Task B Optimization (Day 9-10)

### 5.1 Multi-Signal Sequence Scoring

Combine multiple signals for sequence identification:

```python
def score_sequences_multisignal(sequences_df, model, features):
    scores = {}

    # Signal 1: LogReg coefficients (official method)
    logreg_scores = score_by_logreg_coefficients(sequences_df, model)

    # Signal 2: Differential abundance (positive vs negative repertoires)
    diff_scores = calculate_differential_abundance(sequences_df)

    # Signal 3: Attention weights (if DeepRC is available)
    attn_scores = get_attention_weights(sequences_df, model)

    # Signal 4: Public clonotype frequency
    public_scores = score_public_clonotypes(sequences_df)

    # Weighted combination (optimize weights via validation)
    final_scores = (
        0.5 * normalize(logreg_scores) +
        0.2 * normalize(diff_scores) +
        0.2 * normalize(attn_scores) +
        0.1 * normalize(public_scores)
    )

    return final_scores

def calculate_differential_abundance(sequences_df, positive_repertoires, negative_repertoires):
    """Calculate enrichment in positive vs negative repertoires."""
    scores = []
    for seq in sequences_df['junction_aa']:
        freq_pos = count_in_repertoires(seq, positive_repertoires) / len(positive_repertoires)
        freq_neg = count_in_repertoires(seq, negative_repertoires) / len(negative_repertoires)
        # Log fold change
        if freq_neg > 0:
            scores.append(np.log2((freq_pos + 1e-6) / (freq_neg + 1e-6)))
        else:
            scores.append(np.log2(freq_pos + 1e-6) + 10)  # Large positive
    return scores
```

---

## Part 6: Implementation Timeline

### Day 1 (Dec 7) - Critical Fixes
- [ ] Fix Task B to use binary k-mer + LogReg coefficients
- [ ] Validate submission format against sample_submissions.csv
- [ ] Submit fixed version → Target: 0.72-0.75

### Day 2 (Dec 8) - Feature Engineering Part 1
- [ ] Implement V/J gene usage features
- [ ] Implement diversity metrics
- [ ] Submit enhanced version → Target: 0.75-0.77

### Day 3-4 (Dec 9-10) - Feature Engineering Part 2
- [ ] Implement multi-scale k-mers (k=3,4,5)
- [ ] Implement CDR3 length features
- [ ] Add public clonotype features
- [ ] Submit → Target: 0.77-0.79

### Day 5-6 (Dec 11-12) - Ensemble Building
- [ ] Implement XGBoost + LightGBM + CatBoost stacking
- [ ] Implement per-dataset fine-tuning
- [ ] Optimize ensemble weights
- [ ] Submit → Target: 0.79-0.81

### Day 7-8 (Dec 13-14) - Deep Learning
- [ ] Implement DeepRC-lite attention
- [ ] Implement ESM-2 embeddings
- [ ] Integrate with ensemble
- [ ] Submit → Target: 0.81-0.82

### Day 9-10 (Dec 15-16) - Final Optimization
- [ ] Optimize Task B sequence scoring
- [ ] Final ensemble weight tuning
- [ ] Multiple submission attempts
- [ ] Final submission → Target: 0.82+

---

## Part 7: Risk Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| ESM-2 too slow | Medium | High | Use smallest model (8M params), precompute embeddings |
| Overfitting to public LB | High | High | Leave-one-dataset-out CV, regularization |
| GPU OOM | Low | Medium | Batch processing, gradient checkpointing |
| Time runs out | Medium | High | Prioritize P0 fixes first, submit daily |
| Task B still broken | Medium | High | Test locally with train labels first |

---

## Part 8: Success Metrics

### Submission Score Tracking

| Date | Version | Score | Delta | Notes |
|------|---------|-------|-------|-------|
| Dec 6 | Enhanced v1 | 0.65176 | - | Current best |
| Dec 6 | GPU XGBoost | 0.63350 | -0.018 | Wrong ID order initially |
| Dec 7 | Task B Fix | TBD | +0.07? | Expected after P0 fix |
| Dec 8 | +V/J Features | TBD | +0.03? | - |
| ... | ... | ... | ... | ... |

### Target Milestones

- **Dec 10**: Break 0.75 (enter top 50)
- **Dec 13**: Break 0.80 (enter top 10)
- **Dec 16**: Break 0.82 (challenge for top 3)

---

## Appendix A: Key Literature References

1. **DeepRC** (NeurIPS 2020): Modern Hopfield Networks enable immune repertoire classification
   - DOI: 10.1101/2020.04.12.038158
   - Key: Attention-based MIL achieves state-of-the-art

2. **ESM-2** (Science 2023): Language models for protein sequences
   - Key: Pre-trained embeddings capture evolutionary information

3. **Mal-ID** (2022): Multi-disease classification with 6-model ensemble
   - AUROC: 0.959
   - Key: Ensemble diversity is critical

4. **MotifBoost** (2021): K-mer + GBDT for data-efficient learning
   - Key: Multi-scale k-mers capture different pattern levels

5. **TEPCAM** (2024): Self-attention for epitope-TCR binding prediction
   - Key: Attention weights identify important residues

---

## Appendix B: Code Templates

### Quick Start for Day 1

```bash
# Activate environment
source .venv/bin/activate

# Run fixed Task B implementation
python -c "
from main import KmerClassifier, ImmuneStatePredictor
import pandas as pd

# Train model
predictor = ImmuneStatePredictor(n_jobs=4)
predictor.fit('./data/train_datasets/train_dataset_1')

# Verify Task B uses LogReg coefficients
sequences = predictor.identify_associated_sequences(
    './data/train_datasets/train_dataset_1',
    top_k=50000
)
print(f'Task B sequences: {len(sequences)}')
print(sequences.head())
"
```

---

*Plan Version: 2.0*
*Last Updated: 2025-12-07*
*Author: Claude (Competition Strategy Agent)*
