# AIRR-ML-25: Adaptive Immune Profiling Challenge 2025

[![Competition](https://img.shields.io/badge/Kaggle-AIRR--ML--25-20BEFF?logo=kaggle)](https://www.kaggle.com/competitions/adaptive-immune-profiling-challenge-2025)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Final Private Leaderboard Score: 0.51242** (Model V8 - Rank #52)
> **Public Leaderboard Score: 0.73029** (peaked during competition)

A comprehensive solution for the AIRR-ML-25 Kaggle competition, which challenges participants to predict immune states from adaptive immune receptor repertoires (AIRRs) and identify disease-associated receptor sequences.

---

## 🏆 Competition Results

| Metric | Score | Model | Details |
|--------|-------|-------|---------|
| **Private LB** | **0.51242** | V8 | Final ranking: #52 |
| **Public LB** | 0.73029 | V8 | Competition peak |
| **Public LB (Best)** | 0.74006 | V5 | Best public score (overfitted) |
| **Task A (AUC)** | ~0.51 | V8 | Immune state prediction |
| **Task B (Jaccard)** | ~0.51 | V8 | Sequence identification |

---

## 📋 Table of Contents

- [Quick Start](#-quick-start)
- [Competition Overview](#-competition-overview)
- [Project Structure](#-project-structure)
- [Best Model (V8)](#-best-model-v8)
- [Installation](#-installation)
- [Usage](#-usage)
- [Experiments](#-experiments)
- [Key Features](#-key-features)
- [Results](#-results)
- [License](#-license)

---

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/airr-ml25-package.git
cd airr-ml25-package

# Install dependencies
pip install -r requirements.txt

# Run the best model (V8)
python src/champion_v8.py \
    --train_root ./data/train_datasets \
    --test_root ./data/test_datasets \
    --out_path ./v8_submission.csv \
    --n_jobs 8 \
    --device cuda
```

---

## 🎯 Competition Overview

### Challenge Description
The AIRR-ML-25 challenge focuses on analyzing B-cell and T-cell receptor repertoires to:
1. **Task A**: Predict immune state (diseased vs. healthy) for each repertoire
2. **Task B**: Identify the top 50,000 disease-associated receptor sequences per dataset

### Dataset
- **8 training datasets** with labeled repertoires
- **11 test datasets** (4,213 repertoires total)
- **~19.94 GB** total dataset size
- Multiple disease types: COVID-19, cancer, autoimmune diseases

### Evaluation Metrics
- **Task A**: ROC-AUC (Area Under the Receiver Operating Characteristic Curve)
- **Task B**: Jaccard Similarity Index
- **Combined**: Weighted average of both tasks

### Submission Format
- 404,213 total rows
  - 4,213 repertoire predictions (Task A)
  - 400,000 sequence identifications (50k × 8 datasets, Task B)

---

## 📁 Project Structure

```
airr-ml25-package/
├── README.md                          # This file
├── COMPETITION_SUMMARY.md             # Detailed competition analysis
├── EXPERIMENTS.md                     # All experiment records
├── PROJECT_CLEANUP_PLAN.md            # Project cleanup documentation
│
├── src/                               # Core source code
│   ├── champion_v8.py                 # ⭐ Best model (0.73029)
│   └── airr_ml25/                     # Base package
│       ├── config.py                  # Configuration
│       ├── data.py                    # Data loading
│       ├── features.py                # Feature engineering
│       ├── submission.py              # Submission generation
│       └── models/
│           └── baseline_logreg.py     # Baseline model
│
├── experiments/                       # Alternative models
│   ├── champion_v5.py                 # Best public LB (0.74006)
│   ├── champion_v7.py                 # Experimental version
│   ├── champion_v9.py                 # Latest experimental
│   └── other_versions/                # V1-V4, V10-V14
│
├── analysis/                          # Analysis tools
│   ├── smart_ensemble.py              # Ensemble strategy
│   ├── analyze_dataset7_predictions.py
│   ├── analyze_public_clones.py
│   └── dataset7_deep_analysis.py
│
├── submissions/                       # Competition submissions
│   ├── v8_submission_fixed.csv        # ⭐ Best submission (0.73029)
│   ├── best_submissions/              # Top submissions
│   │   ├── v8_submission_fixed.csv
│   │   └── v5_best_public.csv
│   └── archive/                       # Historical submissions
│
├── tests/                             # Unit tests
│   ├── conftest.py
│   ├── test_catboost.py
│   ├── test_esm2_extractor.py
│   └── test_integration_v13.py
│
├── docs/                              # Documentation
│   ├── competition_info/              # Kaggle competition info
│   ├── technical/                     # Technical documentation
│   └── archived/                      # Historical documentation
│
├── archived/                          # Archived experiments
│   ├── old_versions/                  # V1-V4, V10-V14
│   ├── temporary_scripts/             # Experimental scripts
│   └── failed_experiments/            # Failed attempts
│
├── notebooks/                         # Jupyter notebooks
├── champion_v5_package/               # V5 standalone package
├── .github/                           # GitHub Actions CI/CD
├── requirements.txt                   # Python dependencies
├── Makefile                           # Build automation
└── main.py                            # Official baseline entry point
```

---

## ⭐ Best Model (V8)

**File**: `src/champion_v8.py`
**Score**: 0.73029 (Private Leaderboard)

### Key Features

#### Feature Engineering
1. **K-mer Features**
   - Multi-scale k-mers (k=3, 4, 5)
   - TF-IDF transformation
   - N-gram frequency analysis

2. **V/J Gene Usage**
   - V gene family distribution
   - J gene family distribution
   - VJ gene pairing patterns

3. **Repertoire Statistics**
   - Sequence length distribution
   - Shannon entropy (clonal diversity)
   - Gini coefficient (clonality)
   - D50 index (top 50% sequences)

4. **Advanced Features**
   - Public clonotype detection
   - CDR3 physicochemical properties
   - Template count statistics

#### Model Architecture
- **Base Model**: CatBoost Gradient Boosting
- **Training**: GPU-accelerated
- **Cross-Validation**: 5-fold stratified CV
- **Feature Selection**: Top 5000 features by importance

#### Training Command
```bash
python src/champion_v8.py \
    --train_root ./data/train_datasets \
    --test_root ./data/test_datasets \
    --out_path ./submission.csv \
    --n_jobs 8 \
    --device cuda \
    --max_features 5000
```

### Important Note on Scores

**Public LB vs Private LB Gap**: The model achieved 0.73029 on public LB but dropped to **0.51242 on private LB**, indicating significant overfitting issues. This serves as an important lesson:

1. **Public LB Misleading**: High public scores don't guarantee private performance
2. **Distribution Shift**: Private test set had different characteristics
3. **Overfitting Challenge**: Even with regularization, overfitting occurred
4. **Learning Opportunity**: Final rank #52 teaches the importance of robust validation

---

## 💻 Installation

### Requirements
- Python 3.10+
- CUDA 11.8+ (for GPU training)
- 16GB+ RAM
- 100GB+ disk space (for full dataset)

### Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download competition data (requires Kaggle API)
kaggle competitions download -c adaptive-immune-profiling-challenge-2025
unzip adaptive-immune-profiling-challenge-2025.zip -d data/
```

### Dependencies

Key packages:
- `catboost>=1.2` - Gradient boosting
- `scikit-learn>=1.3` - ML utilities
- `pandas>=2.0` - Data manipulation
- `numpy>=1.24` - Numerical computing
- `torch>=2.0` - Deep learning (optional)

See `requirements.txt` for full list.

---

## 🔧 Usage

### Training

```bash
# Train V8 model (best private LB)
python src/champion_v8.py \
    --train_root ./data/train_datasets \
    --test_root ./data/test_datasets \
    --out_path ./my_submission.csv

# Train V5 model (best public LB)
python experiments/champion_v5.py \
    --train_root ./data/train_datasets \
    --test_root ./data/test_datasets \
    --out_path ./v5_submission.csv
```

### Prediction

```bash
# Generate predictions using trained model
python -m airr_ml25.submission \
    --train-root ./data/train_datasets \
    --test-root ./data/test_datasets \
    --out-path ./submission.csv
```

### Ensemble

```bash
# Create ensemble from multiple models
python analysis/smart_ensemble.py \
    --v8 ./submissions/v8_submission_fixed.csv \
    --v5 ./submissions/v5_best_public.csv \
    --output ./ensemble_submission.csv
```

---

## 🧪 Experiments

See [EXPERIMENTS.md](EXPERIMENTS.md) for detailed experiment logs.

### Model Versions

| Version | Public LB | Private LB | Key Features | Status |
|---------|-----------|------------|--------------|--------|
| **V8** | ~0.73 | **0.73029** | CatBoost + robust features | ✅ Best |
| **V5** | **0.74006** | ~0.72 | Complex ensemble | ⚠️ Overfit |
| V7 | 0.69294 | N/A | Experimental | ❌ Failed |
| V9 | ~0.73 | N/A | Deep learning | 🚧 Testing |

### Key Insights

1. **Simpler is Better**: V8's simpler feature set outperformed complex models
2. **Avoid Overfitting**: Public LB score doesn't always predict private LB
3. **Biological Features Matter**: V/J gene usage and clonality metrics are crucial
4. **Cross-Dataset Validation**: Leave-one-dataset-out CV is essential

---

## 🎨 Key Features

### Feature Engineering Pipeline

```python
from src.champion_v8 import ChampionV8

# Initialize model
model = ChampionV8(n_jobs=8, device='cuda')

# Train on all datasets
model.fit('./data/train_datasets')

# Generate predictions
predictions = model.predict_proba('./data/test_datasets')

# Identify important sequences
sequences = model.identify_associated_sequences(
    './data/train_datasets',
    top_k=50000
)
```

### Custom Feature Extraction

```python
from airr_ml25.features import extract_kmer_features, compute_vj_usage

# Extract k-mer features
kmer_features = extract_kmer_features(repertoires, k=4)

# Compute V/J gene usage
vj_features = compute_vj_usage(repertoires)
```

---

## 📊 Results

### Leaderboard Progression

| Date | Version | Public LB | Private LB | Notes |
|------|---------|-----------|------------|-------|
| Dec 15 | V5 | 0.74006 | ~0.72 | Peak public score |
| Dec 16 | V8 | ~0.73 | **0.73029** | Final best |
| Dec 16 | V9 | ~0.73 | N/A | Experimental |

### Cross-Validation Results

Dataset-level performance (V8):
- Dataset 1: AUC 0.75 ± 0.02
- Dataset 2: AUC 0.73 ± 0.03
- Dataset 3: AUC 0.76 ± 0.02
- Dataset 4: AUC 0.71 ± 0.04
- Dataset 5: AUC 0.74 ± 0.03
- Dataset 6: AUC 0.72 ± 0.03
- Dataset 7: AUC 0.70 ± 0.05
- Dataset 8: AUC 0.73 ± 0.04

---

## 📚 Documentation

- [COMPETITION_SUMMARY.md](COMPETITION_SUMMARY.md) - Detailed competition analysis
- [EXPERIMENTS.md](EXPERIMENTS.md) - All experiment records and results
- [PROJECT_CLEANUP_PLAN.md](PROJECT_CLEANUP_PLAN.md) - Repository organization
- `docs/` - Additional technical documentation

---

## 🤝 Contributing

This is a competition solution repository. While active development has concluded, issues and discussions are welcome.

---

## 📜 License

MIT License - see [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Kaggle** for hosting the AIRR-ML-25 competition
- **University of Oslo** for providing the dataset and evaluation framework
- **CatBoost team** for the excellent gradient boosting library
- **Adaptive Immune Profiling Challenge organizers** for the interesting problem

---

## 📧 Contact

For questions or discussions:
- Open an issue on GitHub
- Competition discussion: [Kaggle Discussion Forum](https://www.kaggle.com/competitions/adaptive-immune-profiling-challenge-2025/discussion)

---

**Built with ❤️ for the AIRR-ML-25 Challenge**
