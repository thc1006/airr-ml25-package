# AIRR-ML-25: Adaptive Immune Profiling Challenge - Code (Notebooks)

## Overview

This page contains community-contributed Kaggle notebooks for the competition. These notebooks include exploratory data analysis (EDA), baseline models, and solution approaches shared by participants.

---

## Featured Notebooks (Sorted by Hotness)

| # | Title | Author | Last Run Time | Votes | Link |
|---|-------|--------|---------------|-------|------|
| 1 | **Example Baseline Predictor using Code Template** | Chakravarthi Kanduri | 2025-11-06 05:25:20 | 110 | [Link](https://www.kaggle.com/code/ckanduri/example-baseline-predictor-using-code-template) |
| 2 | **AIR_ML25_XGBoost** | Bakuer30 | 2025-11-30 13:18:27 | 43 | [Link](https://www.kaggle.com/code/bakuer30/air-ml25-xgboost) |
| 3 | **Code Template** | Chakravarthi Kanduri | 2025-11-04 10:34:29 | 39 | [Link](https://www.kaggle.com/code/ckanduri/code-template) |
| 4 | **AIRR-ML🧬25: naive baseline with XGBoost+PCA** | Jirka Borovec | 2025-11-14 22:08:57 | 28 | [Link](https://www.kaggle.com/code/jirkaborovec/airr-ml-25-naive-baseline-with-xgboost-pca) |
| 5 | **AIR ML25 XGBoost \| Grid Search** | AnthonyTherrien | 2025-11-30 00:10:38 | 25 | [Link](https://www.kaggle.com/code/anthonytherrien/air-ml25-xgboost-grid-search) |
| 6 | **AIRR-ML🧬25: EDA & Convert dataset to parquet** | Jirka Borovec | 2025-11-14 14:52:54 | 15 | [Link](https://www.kaggle.com/code/jirkaborovec/airr-ml-25-eda-convert-dataset-to-parquet) |
| 7 | **Baseline ML for AIRR-ML-25** | Barkat Ali Arbab | 2025-11-13 07:33:30 | 12 | [Link](https://www.kaggle.com/code/barkataliarbab/baseline-ml-for-airr-ml-25) |
| 8 | **EDA AIRR-ML** | Gabriel Cabas | 2025-11-17 04:18:46 | 7 | [Link](https://www.kaggle.com/code/gabrielcabas/eda-airr-ml) |
| 9 | **Blended with Random Noise** | Liam Arden | 2025-11-20 03:47:21 | 5 | [Link](https://www.kaggle.com/code/liamarden/blended-with-random-noise) |
| 10 | **AIRR-ML-25 Start EDA** | Daniel Mario Buchberger | 2025-11-30 06:59:36 | 5 | [Link](https://www.kaggle.com/code/danielmbuchberger/airr-ml-25-start-eda) |
| 11 | **AIRRML25_tabpfn** | d_kriuchkova | 2025-11-21 17:15:34 | 4 | [Link](https://www.kaggle.com/code/dkriuchkova/airrml25-tabpfn) |
| 12 | **AIRR-ML-25: Solution** | Imaad Mahmood | 2025-12-01 04:37:13 | 0 | [Link](https://www.kaggle.com/code/imaadmahmood/airr-ml-25-solution) |
| 13 | **AIRR-ML-25-correction sur la submission** | LazyUnicorn | 2025-12-01 14:55:41 | 0 | [Link](https://www.kaggle.com/code/nocarbonintelligence/airr-ml-25-correction-sur-la-submission) |
| 14 | **AIRR-ML-25** | LazyUnicorn | 2025-12-01 12:19:34 | 0 | [Link](https://www.kaggle.com/code/nocarbonintelligence/airr-ml-25) |
| 15 | **AIRR-isolation et téléchargements des CSV train** | LazyUnicorn | 2025-11-30 09:30:53 | 0 | [Link](https://www.kaggle.com/code/nocarbonintelligence/airr-isolation-et-t-l-chargements-des-csv-train) |

---

## Key Official Notebooks

### 1. Code Template (Official)
- **Author**: Chakravarthi Kanduri (Competition Host)
- **URL**: https://www.kaggle.com/code/ckanduri/code-template
- **Votes**: 39
- **Description**: Official code template that participants are strongly encouraged to follow for unified model interface.

### 2. Example Baseline Predictor using Code Template (Official)
- **Author**: Chakravarthi Kanduri (Competition Host)
- **URL**: https://www.kaggle.com/code/ckanduri/example-baseline-predictor-using-code-template
- **Votes**: 110
- **Description**: Example implementation showing how to use the code template to build a baseline predictor.

---

## Popular Community Approaches

### XGBoost-based Solutions
- **AIR_ML25_XGBoost** by Bakuer30 (43 votes)
- **AIR ML25 XGBoost | Grid Search** by AnthonyTherrien (25 votes)
- **AIRR-ML🧬25: naive baseline with XGBoost+PCA** by Jirka Borovec (28 votes)

### EDA Notebooks
- **AIRR-ML🧬25: EDA & Convert dataset to parquet** by Jirka Borovec (15 votes)
- **EDA AIRR-ML** by Gabriel Cabas (7 votes)
- **AIRR-ML-25 Start EDA** by Daniel Mario Buchberger (5 votes)

### Alternative Approaches
- **AIRRML25_tabpfn** by d_kriuchkova (4 votes) - TabPFN approach
- **Baseline ML for AIRR-ML-25** by Barkat Ali Arbab (12 votes)

---

## Notebook Categories

- **All**: View all notebooks
- **Your Work**: Your own notebooks
- **Shared With You**: Notebooks shared with you
- **Bookmarks**: Your bookmarked notebooks

## Sort Options

- Hotness (default)
- Most Votes
- Most Recent
- Best Score

---

## Code Template Requirements (From Overview)

To win the prize money and be considered for the scientific manuscript authorship, participants are strongly encouraged to:

1. Fork the official code template: https://github.com/uio-bmi/predict-airr
2. Implement the `ImmuneStatePredictor` class in `predictor.py`
3. Ensure unified command-line interface:
   ```bash
   python3 -m submission.main --train_dir /path/to/train_dir --test_dir /path/to/test_dir --out_dir /path/to/output_dir --n_jobs 4 --device cpu
   ```
4. Update `requirements.txt` with exact dependencies
