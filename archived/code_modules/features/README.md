# Gene Usage Feature Extraction

This module provides tools for extracting V/J gene usage features from adaptive immune receptor repertoires (AIRR-seq data).

## Overview

Gene usage patterns are important biomarkers in immune repertoire analysis. Different disease states often show characteristic usage patterns of specific V genes, J genes, and VJ gene pairs. This module extracts these patterns as numerical features for machine learning.

## Features

### 1. Individual Gene Usage Functions

```python
from src.features.gene_usage import (
    extract_v_gene_usage,
    extract_j_gene_usage,
    extract_vj_pair_usage
)

# Extract V gene frequencies
v_usage = extract_v_gene_usage(repertoire_df)
# Returns: {'TRBV20-1': 0.45, 'TRBV7-3': 0.22, ...}

# Extract J gene frequencies
j_usage = extract_j_gene_usage(repertoire_df)
# Returns: {'TRBJ2-7': 0.38, 'TRBJ1-1': 0.19, ...}

# Extract VJ pair frequencies
vj_usage = extract_vj_pair_usage(repertoire_df, top_k=50)
# Returns: {'TRBV20-1_TRBJ2-7': 0.15, ...}
```

### 2. GeneUsageFeaturizer Class

The main interface for feature extraction in ML pipelines:

```python
from src.features.gene_usage import GeneUsageFeaturizer

# Initialize featurizer
featurizer = GeneUsageFeaturizer(
    top_v_genes=50,    # Track top 50 V genes
    top_j_genes=15,    # Track top 15 J genes
    top_vj_pairs=100   # Track top 100 VJ pairs
)

# Fit on training data (learns vocabulary)
featurizer.fit(train_repertoires)

# Transform single repertoire
features = featurizer.transform(repertoire_df)
# Returns: numpy array of shape (165,) = 50 + 15 + 100

# Transform multiple repertoires
features = featurizer.transform_many(repertoires)
# Returns: numpy array of shape (n_repertoires, 165)

# Get feature names
names = featurizer.feature_names
# Returns: ['v_TRBV20-1', 'v_TRBV7-3', ..., 'j_TRBJ2-7', ..., 'vj_TRBV20-1_TRBJ2-7', ...]
```

### 3. Diversity Metrics

Calculate gene usage diversity within repertoires:

```python
from src.features.gene_usage import get_gene_diversity

# Shannon entropy (higher = more diverse)
shannon = get_gene_diversity(repertoire_df, 'v_call', metric='shannon')

# Simpson index (higher = more diverse)
simpson = get_gene_diversity(repertoire_df, 'v_call', metric='simpson')

# Gini coefficient (lower = more diverse)
gini = get_gene_diversity(repertoire_df, 'v_call', metric='gini')
```

## Complete Pipeline Example

```python
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from src.features.gene_usage import GeneUsageFeaturizer, get_gene_diversity

# 1. Load data
train_repertoires = [...]  # List of DataFrames
test_repertoires = [...]

# 2. Initialize and fit featurizer
featurizer = GeneUsageFeaturizer(
    top_v_genes=50,
    top_j_genes=15,
    top_vj_pairs=100
)
featurizer.fit(train_repertoires)

# 3. Extract features
X_train = featurizer.transform_many(train_repertoires)
X_test = featurizer.transform_many(test_repertoires)

# 4. Add diversity metrics
train_diversity = []
for rep in train_repertoires:
    shannon = get_gene_diversity(rep, 'v_call', metric='shannon')
    simpson = get_gene_diversity(rep, 'v_call', metric='simpson')
    train_diversity.append([shannon, simpson])

X_train = np.hstack([X_train, np.array(train_diversity)])
# Similarly for test data...

# 5. Train classifier
clf = LogisticRegression(C=0.1, penalty='l1', solver='liblinear')
clf.fit(X_train, y_train)

# 6. Predict
y_pred = clf.predict_proba(X_test)[:, 1]
```

## Key Features

### Gene Name Normalization

The module automatically normalizes gene names to handle allelic variants:

```python
'TRBV20-1*01' -> 'TRBV20-1'
'TRBV20-1*02' -> 'TRBV20-1'
```

This groups together functional variants of the same gene.

### Missing Data Handling

- Missing gene calls (NaN, None) are handled gracefully
- Mapped to 'UNKNOWN' internally
- Excluded from features (not informative)

### Sparse Representation

The featurizer:
- Learns vocabulary from training data
- Returns 0.0 for genes not seen during fit
- Handles new/rare genes in test data gracefully

## Biological Interpretation

### V Gene Usage
- Variable genes determine antigen recognition specificity
- Disease-specific repertoires often show biased V gene usage
- Example: Certain autoimmune diseases show expansion of specific V genes

### J Gene Usage
- Joining genes contribute to CDR3 structure
- Less diverse than V genes (~15 vs ~50 functional genes in human TCR beta)
- Can show disease-associated patterns

### VJ Pairing
- Non-random pairing patterns exist
- Disease states may show preferential VJ combinations
- More specific biomarker than individual gene usage

### Diversity Metrics
- **Shannon Entropy**: Measures information content
  - Low: Oligoclonal (few genes dominate)
  - High: Polyclonal (balanced usage)
- **Simpson Index**: Probability two random sequences have different genes
- **Gini Coefficient**: Economic inequality measure adapted for gene usage

## Performance Considerations

For large datasets:

1. **Memory**: Features are stored as float32 to reduce memory usage
2. **Speed**: Vectorized operations using pandas/numpy
3. **Scalability**: Can process 1000+ repertoires in seconds

## Testing

Run the test suite:

```bash
python3 test_gene_usage.py
```

See example usage:

```bash
python3 example_gene_usage.py
```

## Integration with AIRR-ML-25 Competition

This module is designed for the AIRR-ML-25 competition:

### Task A (Immune State Prediction)
Use gene usage features as input to binary classifier

### Task B (Sequence Identification)
Use feature importance to identify disease-associated sequences:
1. Train classifier with gene usage features
2. Extract feature coefficients (L1 logistic regression)
3. Map important genes back to sequences

## Dependencies

- pandas >= 1.3.0
- numpy >= 1.20.0
- scipy (optional, for feature importance analysis)

## References

- [State-of-the-art in AIRR Mining](https://www.sciencedirect.com/science/article/pii/S2452310020300524)
- [immuneML Platform](https://pmc.ncbi.nlm.nih.gov/articles/PMC10312379/)

## File Structure

```
src/features/
├── gene_usage.py          # Main module
├── README.md              # This file
test_gene_usage.py         # Unit tests
example_gene_usage.py      # Complete example
```

## API Reference

### Functions

- `normalize_gene_name(gene: str) -> str`
- `extract_v_gene_usage(repertoire, v_col, normalize) -> Dict[str, float]`
- `extract_j_gene_usage(repertoire, j_col, normalize) -> Dict[str, float]`
- `extract_vj_pair_usage(repertoire, v_col, j_col, normalize, top_k) -> Dict[str, float]`
- `get_gene_diversity(repertoire, gene_col, metric) -> float`

### Classes

- `GeneUsageFeaturizer(v_col, j_col, top_v_genes, top_j_genes, top_vj_pairs)`
  - Methods: `fit()`, `transform()`, `transform_many()`
  - Properties: `feature_names`, `n_features`, `v_genes_`, `j_genes_`, `vj_pairs_`

## License

MIT License - Part of the AIRR-ML-25 competition submission
