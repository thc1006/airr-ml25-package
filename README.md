# AIRR-ML-25 Competition Solution

[![Kaggle](https://img.shields.io/badge/Competition-AIRR--ML--25-20BEFF?logo=kaggle)](https://www.kaggle.com/competitions/adaptive-immune-profiling-challenge-2025)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **TL;DR**: Ranked #52/290 teams (Private LB: 0.51242) • Public LB: 0.73029 • CatBoost + k-mer features • GPU optimized

---

## Final Results

| Metric | Score | Rank | Note |
|--------|-------|------|------|
| **Private LB** | **0.51242** | **#52/290** | Final ranking |
| Public LB | 0.73029 | - | V8 model |
| Public LB (peak) | 0.74006 | - | V5 overfitted |

**Key Lesson**: Public-private gap of -0.21787 shows severe overfitting despite regularization.

---

## Quick Start

```bash
# 1. Setup
git clone https://github.com/thc1006/airr-ml25-package.git && cd airr-ml25-package
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Get data (19GB)
kaggle competitions download -c adaptive-immune-profiling-challenge-2025
unzip adaptive-immune-profiling-challenge-2025.zip -d ./data/

# 3. Run (auto-detects GPU/CPU & cores)
python main.py

# Output: ./submissions/v8_submission_<timestamp>.csv
# Training time: ~2-4 hours (RTX 5080)
```

**Hardware**: Auto-detects NVIDIA GPU + optimal CPU cores. No config needed.

---

## Project Structure

```
airr-ml25-package/
├── main.py                    # Entry point (run this)
├── requirements.txt           # Dependencies
├── README.md                  # This file
├── EXPERIMENTS.md             # Complete experiment log (V1-V14)
├── COMPETITION_SUMMARY.md     # Detailed competition analysis
├── FINAL_SUMMARY.md           # Lessons learned & reflections
├── LICENSE                    # MIT License
├── Makefile                   # Build automation
│
├── src/                       # Core source code
│   ├── champion_v8.py         # Best model (0.51242 private LB)
│   ├── utils_parallel.py      # Auto hardware optimization
│   └── __init__.py            # Package initialization
│
├── experiments/               # Alternative model versions
│   ├── champion_v5.py         # Best public LB (0.74006, overfitted)
│   ├── champion_v7.py         # Deep learning attempt
│   ├── champion_v9.py         # Attention-MIL experiment
│   └── other_versions/        # V1-V4, V6, V10-V14
│
├── submissions/               # Competition submissions
│   ├── v8_submission_fixed.csv         # Final submission (0.51242)
│   ├── v5_submission_*.csv             # V5 submissions
│   ├── best_submissions/               # Top submissions archive
│   └── archive/                        # Historical submissions
│
├── analysis/                  # Analysis scripts
│   ├── smart_ensemble.py              # Ensemble analysis
│   ├── analyze_dataset7_predictions.py
│   ├── analyze_public_clones.py
│   └── dataset7_deep_analysis.py
│
├── tests/                     # Unit tests
│   ├── test_main.py           # Main entry point tests
│   ├── conftest.py            # Pytest configuration
│   └── README.md              # Testing documentation
│
├── docs/                      # Documentation
│   ├── competition_info/      # Official Kaggle competition info
│   ├── challenge_overview.md  # Competition summary
│   ├── data_format.md         # Dataset format specifications
│   └── archived/              # Historical documentation
│
├── archived/                  # Archived experiments & code
│   ├── old_versions/          # V1-V4, V6, V10-V14 implementations
│   ├── temporary_scripts/     # Experimental scripts
│   ├── failed_experiments/    # Failed approaches
│   ├── code_modules/          # Old src/ submodules
│   ├── config_experiments/    # YAML experiment configs
│   ├── notebooks_reference/   # Jupyter notebook examples
│   ├── tests_old/             # Old test files
│   ├── scripts/               # Old bash scripts
│   └── logs/                  # Training logs
│
└── data/                      # Competition data (gitignored)
    ├── train_datasets/        # 8 training datasets
    ├── test_datasets/         # 11 test datasets
    └── sample_submissions.csv # Sample submission format
```

---

## Model V8 (Best Submission)

**Algorithm**: CatBoost Gradient Boosting (GPU-accelerated)

**Features** (5000 total):
- Multi-scale k-mers (k=3,4,5) with TF-IDF
- V/J gene usage + VJ pairing patterns
- Clonality metrics (Shannon, Gini, D50)
- Public clonotype detection
- CDR3 sequence statistics

**Training**:
- 5-fold stratified cross-validation
- L2 regularization (λ=3.0)
- 1000 trees, depth=6, lr=0.05
- Early stopping (100 rounds)

**Cross-Validation**: 0.7318 ± 0.0196 (8 datasets)

**Why it failed on private LB**: Distribution shift between public/private test sets. CV didn't catch the overfitting.

---

## Experiments Summary

14 model versions tested. See [EXPERIMENTS.md](EXPERIMENTS.md) for details.

| Version | Public LB | Private LB | Status | Key Change |
|---------|-----------|------------|--------|------------|
| V1-V4 | 0.65-0.69 | - | Failed | Baseline exploration |
| **V5** | **0.74006** | ~0.50 | Overfit | Complex ensemble |
| V6 | 0.72 | - | Failed | Simplified V5 |
| V7 | 0.69 | - | Failed | Deep learning |
| **V8** | **0.73029** | **0.51242** | Best | CatBoost + regularization |
| V9-V14 | 0.70-0.73 | - | Failed | Last-minute attempts |

---

## Key Learnings

### What Worked
- Multi-scale k-mer features (k=3,4,5)
- V/J gene usage patterns
- CatBoost with GPU acceleration
- Stratified cross-validation
- Biological feature engineering

### What Failed
- Over-engineering (V5: 8000 features → overfitted)
- Deep learning (insufficient data)
- Protein embeddings (ESM-2 too slow)
- Chasing public leaderboard (misleading)
- Per-dataset models (didn't generalize)

### Critical Mistake

**Public-Private Gap**: -0.21787 (0.73 → 0.51)
- Public test set was NOT representative of private
- Cross-validation couldn't detect this
- Should have used more conservative validation
- Lesson: Trust CV > leaderboard

---

## Advanced Usage

```bash
# Check hardware configuration
python src/utils_parallel.py

# Run with custom settings
python -m src.champion_v8 --n_jobs 8 --device cuda

# Run specific experiment
python experiments/champion_v5.py  # Best public LB

# Run tests
pytest tests/ -v
```

---

## Documentation

- **[EXPERIMENTS.md](EXPERIMENTS.md)**: Complete experiment log (V1-V14)
- **[COMPETITION_SUMMARY.md](COMPETITION_SUMMARY.md)**: Detailed analysis & approach
- **[FINAL_SUMMARY.md](FINAL_SUMMARY.md)**: Reflections & lessons learned

---

## System Requirements

**Minimum**:
- Python 3.10+
- 8GB RAM
- 4 CPU cores

**Recommended**:
- Python 3.10+
- 32GB RAM
- 8+ CPU cores
- NVIDIA GPU (16GB VRAM)

**Tested On**:
- RTX 5080 16GB + Ryzen 7 7800X3D (2-4 hours training)
- CPU-only (8-12 hours training)

---

## License

MIT License - See [LICENSE](LICENSE)

---

## Acknowledgments

- **Prof. Chien-Chao Tseng (曾建超教授)**: National Yang Ming Chiao Tung University, for guidance and support
- **Yen-Ting Kuo (郭彥廷學長)**: Winlab, for technical assistance and mentorship
- **Ficus Sapiens**: For various aspects of support and collaboration
- **Kaggle & University of Oslo**: For hosting the competition
- **CatBoost Team**: Excellent gradient boosting library
- **Community**: Shared insights and discussions

---

**Competition**: [AIRR-ML-25 on Kaggle](https://www.kaggle.com/competitions/adaptive-immune-profiling-challenge-2025)
**Final Ranking**: #52 out of 290 teams (375 participants)
**Key Takeaway**: Honest documentation of failure is as valuable as success stories.
