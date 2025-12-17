# Champion V5 Package - AIRR-ML-25 Competition

## Overview

This package contains the **Champion V5** solution for the AIRR-ML-25: Adaptive Immune Profiling Challenge 2025, achieving a **Public Score of 0.74006** (historical best, +9.0% improvement).

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run training and prediction
python champion_v5.py

# Output: submissions/v5_submission_YYYYMMDD_HHMMSS.csv
```

## Performance

| Metric | Value |
|--------|-------|
| **Public Score** | 0.74006 |
| **Previous Best** | 0.67887 |
| **Improvement** | +9.0% |
| **Submission File** | v5_fixed_genes.csv |
| **Submission Date** | 2025-12-15 21:01 UTC |

## Key Features

1. **Public Clone Mining**: Identifies disease-specific shared sequences
2. **Multi-scale K-mers**: k=3,4 with positional awareness
3. **XGBoost + LightGBM Ensemble**: GPU-accelerated training
4. **Per-Dataset Models**: Adaptive class weighting for imbalanced datasets
5. **Advanced Features**:
   - V/J gene families
   - Physicochemical properties
   - Diversity metrics
   - Public clone enrichment scores

## Technical Architecture

```
Data Loading
    ↓
Public Clone Mining (per dataset)
    ↓
Feature Extraction (500+ features)
    ├── K-mers (k=3,4)
    ├── Positional K-mers
    ├── V/J Gene Families
    ├── Physicochemical Properties
    ├── Diversity Metrics
    └── Public Clone Features
    ↓
GPU Feature Selection (Top 500)
    ↓
XGBoost + LightGBM Training (5-Fold CV)
    ↓
Stacking Ensemble (Learned Weights)
    ↓
Prediction + Task B Sequence Identification
    ↓
Submission File (404,213 rows)
```

## Files Included

- `champion_v5.py` (28 KB) - Main training and prediction script
- `v5_fixed_genes.csv` (34 MB) - Successful submission file
- `v5_submission_20251215_210121.csv` (34 MB) - Original generated submission
- `CHAMPION_V5_REPORT.md` (52 KB) - Comprehensive technical report
- `CLAUDE.md` (18 KB) - Project guidelines and competition context
- `requirements.txt` (761 bytes) - Python dependencies

## Hardware Requirements

- **GPU**: NVIDIA RTX 5080 (16GB VRAM) or equivalent
- **CUDA**: 12.x
- **RAM**: 32 GB recommended
- **Storage**: ~20 GB for datasets

## Usage

```python
# Basic usage
python champion_v5.py

# The script will:
# 1. Mine public clones from 8 training datasets
# 2. Extract features from all repertoires
# 3. Train XGBoost + LightGBM ensemble per dataset
# 4. Predict on 11 test datasets (Task A)
# 5. Identify top 50,000 sequences per dataset (Task B)
# 6. Generate submission file
```

## Configuration

Key parameters in `champion_v5.py`:

```python
class Config:
    K_LIST = [3, 4]              # K-mer sizes
    TOP_KMER = 500               # Top features to select
    MAX_SEQUENCES_PER_FILE = 50000

    # Public clone settings
    PUB_MAX_FILES = 30
    PUB_MIN_FREQ = 0.15
    PUB_ENRICH = 5.0
    PUB_TOP_N = {1:2000, 2:2000, ..., 7:5000, 8:3000}

    # Training settings
    N_SPLITS = 5
    RANDOM_STATE = 42
    EARLY_STOP = 100

    # Per-dataset class weights
    SCALE_POS_WEIGHT = {
        1:1.0, 2:1.0, 3:1.0, 4:1.0, 5:1.0, 6:1.0,
        7:5.0,  # HCV - severe imbalance
        8:2.0   # IBD - moderate imbalance
    }
```

## Cross-Validation Results (Estimated)

| Dataset | CV AUC | Class Balance | Characteristics |
|---------|--------|---------------|-----------------|
| Dataset 1 | 0.78-0.82 | Balanced | General cohort |
| Dataset 2 | 0.76-0.80 | Balanced | General cohort |
| Dataset 3 | 0.75-0.79 | Balanced | General cohort |
| Dataset 4 | 0.74-0.78 | Balanced | General cohort |
| Dataset 5 | 0.73-0.77 | Balanced | General cohort |
| Dataset 6 | 0.72-0.76 | Balanced | General cohort |
| Dataset 7 | 0.68-0.72 | **Severe imbalance (5x)** | HCV |
| Dataset 8 | 0.70-0.74 | Moderate imbalance (2x) | IBD |
| **Weighted Avg** | **~0.74** | - | - |

## Next Steps to Win

See `CHAMPION_V5_REPORT.md` Section 4 for detailed strategies:

1. **ESM-2 Protein Language Model Embeddings** (+2-3% expected)
2. **Attention-Based MIL Aggregation** (+1-2% expected)
3. **Multi-layer Stacking Ensemble** (+1-2% expected)
4. **SHAP-driven Task B Sequence Selection** (+0.5-1% expected)

**Target: Public Score 0.85+ to win 1st place**

## Troubleshooting

### GPU Out of Memory
```python
# Reduce batch size or max sequences
Config.MAX_SEQUENCES_PER_FILE = 30000  # Instead of 50000
```

### Slow Training
```python
# Reduce number of features
Config.TOP_KMER = 300  # Instead of 500
```

### Submission Validation Error
```bash
# Check format
python -c "import pandas as pd; df = pd.read_csv('v5_fixed_genes.csv'); print(f'Rows: {len(df)}, Expected: 404213')"
```

## Citation

If you use this code, please cite:

```
AIRR-ML-25 Champion V5 Solution
Competition: Adaptive Immune Profiling Challenge 2025
Date: December 2025
Public Score: 0.74006
```

## License

MIT License - Free to use for research and competition purposes

## Contact

For questions about the implementation, see `CHAMPION_V5_REPORT.md` or the original competition forum.

---

**Competition**: AIRR-ML-25: Adaptive Immune Profiling Challenge 2025
**Deadline**: December 17, 2025 (06:59 UTC)
**Current Rank**: Top tier (Public Score 0.74006)
**Target**: 1st place (Score > 0.84590)
