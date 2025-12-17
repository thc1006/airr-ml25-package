# AIRR-ML-25 Winning Strategy Report

## Executive Summary

**Current Status**: Score 0.74006 (achieved 2025-12-15)
**Target**: 0.84590+ (1st place: SajayR)
**Gap**: 10.6% improvement needed
**Deadline**: December 17, 2025 06:59 UTC (~36 hours remaining)

---

## 1. Leaderboard Analysis

| Rank | Team | Score | Gap from Us |
|------|------|-------|-------------|
| 1 | SajayR | 0.84590 | +10.6% |
| 2 | WoLongFengChu | 0.84476 | +10.5% |
| 3 | gordianknot | 0.84003 | +10.0% |
| 4 | GROZD | 0.83668 | +9.7% |
| 5 | GoBlue | 0.83293 | +9.3% |
| ... | ... | ... | ... |
| ~25 | **Us (v5)** | **0.74006** | baseline |

**Key Insight**: Top teams are clustered at 0.83-0.85, suggesting a common winning approach exists.

---

## 2. Current Implementation Analysis (Champion v5)

### What We Have
- K-mer features (k=3,4): 8,420 features total
- Public clone mining (enrichment ratio based)
- XGBoost + LightGBM ensemble with GPU
- Per-dataset scale_pos_weight for class imbalance
- V/J gene one-hot encoding

### What's Missing (Critical Gaps)
1. **ESM2 Protein Language Model Embeddings** - Biggest missing piece
2. **Multi-scale k-mers** (k=5,6) - More sequence context
3. **Diversity indices** (Shannon, Gini, D50, clonality)
4. **TCR clustering / meta-clonotypes** - Better public clone features
5. **Task B optimization** - Current method is basic

---

## 3. Research-Backed Winning Approaches

### 3.1 ESM2 Embeddings (Priority: CRITICAL)

**Evidence**:
- Paper: "Do domain-specific protein language models outperform general ones?" (ImmunoInformatics 2024)
- Finding: ESM2-650M Layer 6 embeddings significantly outperform domain-specific models
- Expected gain: **+5-8%**

**Implementation**:
```python
from transformers import AutoModel, AutoTokenizer

# Load ESM2-650M
model = AutoModel.from_pretrained("facebook/esm2_t33_650M_UR50D")
tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")

# Extract layer 6 embeddings (not final layer!)
def get_esm_embedding(sequence):
    inputs = tokenizer(sequence, return_tensors="pt")
    outputs = model(**inputs, output_hidden_states=True)
    # Use layer 6 (index 6) - proven best for TCR
    return outputs.hidden_states[6].mean(dim=1)

# Aggregate per repertoire
def repertoire_embedding(sequences):
    embeddings = [get_esm_embedding(seq) for seq in sequences[:1000]]  # Sample
    return torch.stack(embeddings).mean(dim=0)
```

### 3.2 Public Clone / Meta-clonotype Features (Priority: HIGH)

**Evidence**:
- Emerson et al. (2017): CMV classification using public TCRs
- immuneML replication: Found 143/164 CMV-associated TCRs
- Expected gain: **+2-4%**

**Current vs Improved**:
```python
# Current: Simple enrichment ratio
enrichment = (pos_freq + 1e-6) / (neg_freq + 1e-6)

# Improved: Statistical significance + clustering
from scipy.stats import fisher_exact

def calculate_public_clone_score(seq, pos_repertoires, neg_repertoires):
    pos_count = sum(1 for r in pos_repertoires if seq in r)
    neg_count = sum(1 for r in neg_repertoires if seq in r)

    # Fisher's exact test for significance
    table = [[pos_count, len(pos_repertoires) - pos_count],
             [neg_count, len(neg_repertoires) - neg_count]]
    odds_ratio, p_value = fisher_exact(table)

    return -np.log10(p_value + 1e-10) * np.sign(np.log(odds_ratio + 1e-10))
```

### 3.3 Diversity Indices (Priority: MEDIUM-HIGH)

**Evidence**:
- Multiple papers show diversity correlates with immune state
- Expected gain: **+1-2%**

```python
from scipy.stats import entropy

def calculate_diversity_indices(clone_counts):
    total = sum(clone_counts)
    freqs = [c/total for c in clone_counts]

    # Shannon entropy
    shannon = entropy(freqs, base=2)

    # Gini index
    sorted_freqs = sorted(freqs)
    n = len(sorted_freqs)
    gini = 1 - 2 * sum((n - i) * f for i, f in enumerate(sorted_freqs)) / (n * sum(freqs))

    # D50 (clones covering 50% of repertoire)
    cumsum = 0
    d50 = 0
    for f in sorted(freqs, reverse=True):
        cumsum += f
        d50 += 1
        if cumsum >= 0.5:
            break
    d50_norm = d50 / len(freqs)

    # Clonality
    clonality = 1 - shannon / np.log2(len(freqs))

    return {
        'shannon': shannon,
        'gini': gini,
        'd50': d50_norm,
        'clonality': clonality,
        'unique_clones': len(freqs),
        'max_clone_freq': max(freqs)
    }
```

### 3.4 Multi-scale K-mers (Priority: MEDIUM)

**Evidence**:
- k=3,4 capture local patterns
- k=5,6 capture longer motifs and CDR3 core patterns
- Expected gain: **+1-2%**

```python
# Current: k=3,4 (8,420 features)
# Add: k=5,6 (with top-variance selection)

def extract_multiscale_kmers(sequences, k_values=[3,4,5,6], top_n=50000):
    all_kmers = {}
    for k in k_values:
        kmer_counts = defaultdict(int)
        for seq in sequences:
            for i in range(len(seq) - k + 1):
                kmer_counts[seq[i:i+k]] += 1
        all_kmers.update(kmer_counts)

    # Select top variance k-mers
    # ... variance calculation across repertoires
    return top_n_features
```

### 3.5 Task B Optimization (Priority: HIGH)

**Evidence**:
- Task B (sequence identification) is weighted equally with Task A
- Current approach: k-mer enrichment + XGBoost importance
- Better: TF-IDF, mutual information, attention weights

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import mutual_info_classif

def improved_sequence_scoring(sequences, labels):
    # TF-IDF on CDR3 sequences
    tfidf = TfidfVectorizer(analyzer='char', ngram_range=(3,5))
    X = tfidf.fit_transform(sequences)

    # Mutual information with labels
    mi_scores = mutual_info_classif(X, labels)

    # Combine with existing importance scores
    combined_scores = 0.5 * mi_scores + 0.5 * existing_scores

    return combined_scores
```

---

## 4. Priority Action Plan

### Phase 1: Immediate (Next 12 hours) - Target: 0.78+

| Action | Expected Gain | Time | Priority |
|--------|--------------|------|----------|
| Add diversity indices | +1-2% | 2h | HIGH |
| Improve Task B scoring (TF-IDF + MI) | +2-3% | 3h | HIGH |
| Add k=5 k-mers | +1% | 2h | MEDIUM |
| Dataset-specific hyperparameter tuning | +1% | 3h | MEDIUM |

### Phase 2: Medium-term (12-24 hours) - Target: 0.82+

| Action | Expected Gain | Time | Priority |
|--------|--------------|------|----------|
| ESM2 embeddings integration | +3-5% | 6h | CRITICAL |
| Public clone statistical testing | +1-2% | 3h | HIGH |
| Meta-clonotype clustering | +1-2% | 4h | MEDIUM |

### Phase 3: Final Push (Last 12 hours) - Target: 0.84+

| Action | Expected Gain | Time | Priority |
|--------|--------------|------|----------|
| Ensemble with ESM-based model | +2-3% | 4h | CRITICAL |
| Fine-tune submission format | +0.5% | 2h | HIGH |
| Final model stacking | +1% | 4h | HIGH |

---

## 5. Technical Implementation Details

### 5.1 ESM2 Pipeline (GPU Required)

```bash
# Install
pip install transformers torch

# Memory: ~8GB VRAM for ESM2-650M
# Time: ~2-3 hours for all repertoires
```

```python
import torch
from transformers import AutoModel, AutoTokenizer

class ESM2FeatureExtractor:
    def __init__(self, model_name="facebook/esm2_t33_650M_UR50D"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model.eval()

    def extract_repertoire_features(self, sequences, sample_size=500):
        """Extract mean embedding from sampled sequences"""
        sampled = random.sample(sequences, min(sample_size, len(sequences)))

        embeddings = []
        with torch.no_grad():
            for seq in sampled:
                inputs = self.tokenizer(seq, return_tensors="pt").to(self.device)
                outputs = self.model(**inputs, output_hidden_states=True)
                # Layer 6 embedding (proven best)
                emb = outputs.hidden_states[6].mean(dim=1).cpu().numpy()
                embeddings.append(emb)

        return np.mean(embeddings, axis=0).flatten()
```

### 5.2 Improved Public Clone Mining

```python
def mine_significant_public_clones(train_data, significance_threshold=0.01):
    """Mine public clones with statistical significance testing"""

    # Separate positive and negative repertoires
    pos_reps = [r for r, l in train_data if l == 1]
    neg_reps = [r for r, l in train_data if l == 0]

    # Count sequence occurrences
    seq_pos_counts = defaultdict(int)
    seq_neg_counts = defaultdict(int)

    for rep in pos_reps:
        for seq in set(rep):  # Unique sequences per repertoire
            seq_pos_counts[seq] += 1

    for rep in neg_reps:
        for seq in set(rep):
            seq_neg_counts[seq] += 1

    # Statistical testing
    significant_seqs = {}
    all_seqs = set(seq_pos_counts.keys()) | set(seq_neg_counts.keys())

    for seq in all_seqs:
        pos_count = seq_pos_counts.get(seq, 0)
        neg_count = seq_neg_counts.get(seq, 0)

        # Fisher's exact test
        table = [[pos_count, len(pos_reps) - pos_count],
                 [neg_count, len(neg_reps) - neg_count]]
        odds_ratio, p_value = fisher_exact(table)

        if p_value < significance_threshold:
            significant_seqs[seq] = {
                'odds_ratio': odds_ratio,
                'p_value': p_value,
                'pos_freq': pos_count / len(pos_reps),
                'neg_freq': neg_count / len(neg_reps),
                'direction': 'positive' if odds_ratio > 1 else 'negative'
            }

    return significant_seqs
```

---

## 6. Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| ESM2 OOM on GPU | HIGH | Use smaller sample size, batch processing |
| Overfitting to public LB | HIGH | Leave-one-dataset-out validation |
| Time constraint | CRITICAL | Parallelize on two machines |
| Submission errors | HIGH | Strict format validation before submit |

---

## 7. Two-Machine Strategy

### Machine 1 (Current)
- Continue with k-mer + ensemble optimization
- Add diversity indices
- Improve Task B scoring
- Target: v6 submission (0.78+)

### Machine 2 (Migration Target)
- ESM2 embedding extraction
- Deep learning approaches (DeepRC if time permits)
- Meta-clonotype clustering
- Target: v7 ESM-enhanced submission (0.82+)

### Final Ensemble
- Combine v6 and v7 predictions
- Weighted average based on CV performance
- Target: 0.84+ final submission

---

## 8. Key References

1. **ESM2 for TCR**: "Do domain-specific protein language models outperform general ones?" - ImmunoInformatics 2024
2. **DeepRC**: Widrich et al. "Immune repertoire classification with attention-based deep massive MIL" - NeurIPS 2020
3. **Public Clones**: Emerson et al. "Immunosequencing identifies signatures of CMV" - Nature Genetics 2017
4. **immuneML**: Pavlovic et al. "The immuneML ecosystem" - Nature Machine Intelligence 2023
5. **Mal-ID**: Zaslavsky et al. "Disease diagnostics using BCR and TCR sequences" - Science 2025

---

## 9. Success Metrics

| Milestone | Score Target | Status |
|-----------|-------------|--------|
| Baseline v5 | 0.74 | ACHIEVED |
| v6 (Diversity + Task B) | 0.78 | PENDING |
| v7 (ESM2) | 0.82 | PENDING |
| Final Ensemble | 0.84+ | PENDING |

---

*Generated: 2025-12-15*
*Last Score: 0.74006*
*Next Target: 0.78+ (v6)*
