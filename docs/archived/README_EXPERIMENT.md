# 實驗管理系統 - AIRR-ML-25

> 從 0.67 到 0.82+ 的結構化實驗框架

---

## 目錄結構

```
airr-ml25-package/
├── experiments/                  # 所有實驗結果
│   ├── exp_001_k345_tfidf/
│   │   ├── config.yaml           # 實驗配置
│   │   ├── logs/                 # 訓練日誌
│   │   ├── models/               # 訓練好的模型
│   │   ├── outputs/              # 提交檔案
│   │   ├── metrics.json          # 驗證指標
│   │   └── kaggle_result.json    # Kaggle 回傳分數
│   ├── exp_002_vj_features/
│   └── ...
│
├── configs/                      # 實驗配置模板
│   ├── exp_001_k345_tfidf.yaml
│   ├── exp_002_vj_features.yaml
│   ├── exp_003_diversity.yaml
│   └── ...
│
├── scripts/                      # 實驗管理腳本
│   ├── train_experiment.py       # 訓練入口
│   ├── submit_experiment.py      # 提交入口
│   └── compare_experiments.py    # 比較工具
│
├── src/airr_ml25/                # 核心代碼庫
│   ├── features/                 # 特徵工程
│   │   ├── kmer.py
│   │   ├── vj_gene.py
│   │   ├── diversity.py
│   │   └── embeddings.py
│   ├── models/                   # 模型實現
│   │   ├── xgboost_model.py
│   │   ├── lightgbm_model.py
│   │   ├── ensemble.py
│   │   └── deep_models.py
│   └── tasks/                    # 任務模組
│       ├── task_a.py
│       └── task_b.py
│
└── docs/                         # 文檔
    ├── strategic_roadmap_0.67_to_0.82.md  # 戰略路線圖
    ├── experiment_log.md                   # 實驗日誌
    ├── QUICKSTART.md                       # 快速開始
    └── model_roadmap.md                    # 模型路線圖
```

---

## 實驗工作流程

### 1. 創建新實驗

```bash
# 複製配置模板
cp configs/exp_001_k345_tfidf.yaml configs/exp_XXX_my_idea.yaml

# 修改配置
vim configs/exp_XXX_my_idea.yaml

# 訓練模型
python scripts/train_experiment.py \
    --config configs/exp_XXX_my_idea.yaml \
    --exp-name exp_XXX_my_idea
```

### 2. 提交實驗

```bash
# 驗證並提交到 Kaggle
python scripts/submit_experiment.py \
    --exp-name exp_XXX_my_idea \
    --message "描述這個實驗的改進"

# 結果會自動保存並更新日誌
```

### 3. 比較實驗

```bash
# 比較多個實驗
python scripts/compare_experiments.py \
    --exps exp_001 exp_002 exp_003 \
    --baseline-score 0.66987 \
    --output experiments/comparison.html
```

---

## 配置檔案格式

### 基本結構

```yaml
experiment:
  name: exp_XXX_description
  description: "詳細描述"
  author: "你的名字"
  date: "2025-12-08"

data:
  train_root: "./data/train_datasets"
  test_root: "./data/test_datasets"
  datasets: [1, 2, 3, 4, 5, 6, 7, 8]

features:
  type: "kmer"  # kmer, vj_gene, diversity, combined, embeddings
  kmer:
    k_values: [3, 4, 5]
    use_tfidf: true
    max_features: null
    min_df: 2

model:
  type: "xgboost"  # xgboost, lightgbm, catboost, ensemble, deep
  xgboost:
    n_estimators: 300
    max_depth: 6
    learning_rate: 0.05
    tree_method: "gpu_hist"
    gpu_id: 0

training:
  cv:
    method: "stratified_kfold"
    n_splits: 5
    random_state: 42
  early_stopping_rounds: 50

task_b:
  method: "kmer_importance"
  top_k: 50000

output:
  model_path: "models/model.pkl"
  submission_path: "outputs/submission.csv"
```

---

## 特徵類型

### 1. K-mer Features

```yaml
features:
  type: "kmer"
  kmer:
    k_values: [3, 4, 5]
    use_tfidf: true
    max_features: null
    min_df: 2
    sublinear_tf: true
```

**實現**: `src/airr_ml25/features/kmer.py`

### 2. V/J Gene Features

```yaml
features:
  type: "vj_gene"
  vj_gene:
    v_gene:
      type: "frequency"
      top_n: 100
      normalize: true
    j_gene:
      type: "frequency"
      top_n: 20
    vj_pair:
      type: "frequency"
      top_n: 500
```

**實現**: `src/airr_ml25/features/vj_gene.py` (待實現)

### 3. Diversity Metrics

```yaml
features:
  type: "diversity"
  diversity:
    shannon_entropy: true
    gini_coefficient: true
    d50: true
    cdr3_length:
      mean: true
      std: true
      skewness: true
```

**實現**: `src/airr_ml25/features/diversity.py` (待實現)

### 4. Combined Features

```yaml
features:
  type: "combined"
  kmer: { ... }
  vj_gene: { ... }
  diversity: { ... }
```

### 5. Embeddings (Phase 3)

```yaml
features:
  type: "embeddings"
  embeddings:
    model: "esm2_t33_650M_UR50D"
    pooling: "attention"
    batch_size: 32
    max_length: 512
```

**實現**: `src/airr_ml25/features/embeddings.py` (待實現)

---

## 模型類型

### 1. XGBoost

```yaml
model:
  type: "xgboost"
  xgboost:
    n_estimators: 300
    max_depth: 6
    learning_rate: 0.05
    tree_method: "gpu_hist"
```

**實現**: `src/airr_ml25/models/xgboost_model.py` (待實現)

### 2. LightGBM

```yaml
model:
  type: "lightgbm"
  lightgbm:
    n_estimators: 300
    max_depth: 6
    learning_rate: 0.05
    device: "gpu"
```

**實現**: `src/airr_ml25/models/lightgbm_model.py` (待實現)

### 3. Ensemble

```yaml
model:
  type: "ensemble"
  ensemble:
    models:
      - type: "xgboost"
        weight: 0.4
      - type: "lightgbm"
        weight: 0.3
      - type: "catboost"
        weight: 0.3
    fusion: "weighted_average"
```

**實現**: `src/airr_ml25/models/ensemble.py` (待實現)

---

## Task B 策略

### 1. K-mer Importance

```yaml
task_b:
  method: "kmer_importance"
  top_k: 50000
  aggregation: "max"
```

基於模型的 k-mer feature importance 排序序列。

### 2. SHAP Values

```yaml
task_b:
  method: "shap"
  top_k: 50000
  background_samples: 100
```

使用 SHAP 計算每個序列的重要性。

### 3. Combined

```yaml
task_b:
  method: "combined"
  top_k: 50000
  aggregation: "weighted_sum"
  weights:
    kmer: 0.6
    vj_gene: 0.3
    diversity: 0.1
```

結合多種特徵的重要性。

---

## 驗證策略

### 1. Stratified K-Fold CV

```yaml
training:
  cv:
    method: "stratified_kfold"
    n_splits: 5
    random_state: 42
```

適合：評估模型穩定性

### 2. Leave-One-Dataset-Out CV

```yaml
training:
  cv:
    method: "lodo"
    random_state: 42
```

適合：評估跨資料集泛化能力（最接近 private LB）

### 3. Time-based Split

```yaml
training:
  cv:
    method: "time_based"
    train_ratio: 0.8
```

適合：如果有時間資訊，避免 temporal leakage

---

## 實驗追蹤

### 自動追蹤的資訊

1. **配置**: `experiments/exp_XXX/config.yaml`
2. **訓練日誌**: `experiments/exp_XXX/logs/*.log`
3. **驗證指標**: `experiments/exp_XXX/metrics.json`
4. **Kaggle 結果**: `experiments/exp_XXX/kaggle_result.json`
5. **模型檔案**: `experiments/exp_XXX/models/*.pkl`

### 實驗日誌

所有實驗會自動記錄到 `docs/experiment_log.md`，包括：
- 實驗名稱和描述
- Local CV 和 Public LB 分數
- 配置差異
- 時間戳

---

## 最佳實踐

### 1. 命名規範

```
exp_001_k345_tfidf          # Good: 編號 + 簡短描述
exp_002_vj_features         # Good
my_experiment               # Bad: 沒有編號
test123                     # Bad: 不描述性
```

### 2. 配置管理

- 每個實驗都要有完整的配置檔案
- 使用版本控制 (Git)
- 記錄隨機種子

### 3. 實驗順序

1. 先跑簡單的 baseline
2. 逐步添加特徵
3. 每次只改變一個變數
4. 記錄所有結果

### 4. 提交策略

- 本地 CV 改進 > 0.01 才提交
- 不要追逐 public LB
- 保留提交次數到最後

---

## 開發指南

### 添加新的特徵類型

1. 創建 `src/airr_ml25/features/my_feature.py`
2. 實現特徵提取函數
3. 在配置檔案中添加參數
4. 在 `train_experiment.py` 中註冊

### 添加新的模型類型

1. 創建 `src/airr_ml25/models/my_model.py`
2. 實現訓練函數
3. 在配置檔案中添加參數
4. 在 `train_experiment.py` 中註冊

### 運行單元測試

```bash
# TODO: 添加測試
pytest tests/
```

---

## 故障排除

### 常見錯誤

1. **ModuleNotFoundError**
   ```bash
   pip install -e .
   ```

2. **CUDA out of memory**
   ```yaml
   # 修改 config.yaml
   model:
     xgboost:
       tree_method: "hist"  # CPU
   ```

3. **Submission format error**
   ```bash
   python scripts/validate_submission.py \
       experiments/exp_XXX/outputs/submission.csv
   ```

---

## 資源與參考

- **戰略路線圖**: [docs/strategic_roadmap_0.67_to_0.82.md](docs/strategic_roadmap_0.67_to_0.82.md)
- **快速開始**: [docs/QUICKSTART.md](docs/QUICKSTART.md)
- **實驗日誌**: [docs/experiment_log.md](docs/experiment_log.md)
- **Kaggle 競賽**: https://www.kaggle.com/competitions/adaptive-immune-profiling-challenge-2025

---

**維護者**: Competition Master Agent
**最後更新**: 2025-12-08
