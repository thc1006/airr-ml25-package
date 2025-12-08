# 快速開始指南 - AIRR-ML-25 實驗管理

> 從 0.67 到 0.82+ 的快速執行路徑

---

## 第一步：了解當前狀況

```bash
# 檢查最新排行榜
kaggle competitions leaderboard adaptive-immune-profiling-challenge-2025 --show | head -20

# 檢查我們的提交歷史
kaggle competitions submissions -c adaptive-immune-profiling-challenge-2025 | head -10
```

**當前最佳**: 0.66987
**目標**: 0.82+
**差距**: +0.15013

---

## 第二步：執行 Phase 1 實驗（Day 1-3）

### 實驗 1: Multi-scale k-mer (k=3,4,5) with TF-IDF

#### 預期提升: +0.03-0.05

```bash
# 1. 訓練模型
python scripts/train_experiment.py \
    --config configs/exp_001_k345_tfidf.yaml \
    --exp-name exp_001_k345_tfidf

# 2. 驗證結果
ls -lh experiments/exp_001_k345_tfidf/outputs/

# 3. 提交到 Kaggle
python scripts/submit_experiment.py \
    --exp-name exp_001_k345_tfidf \
    --message "Multi-scale k-mer (3,4,5) with TF-IDF weighting"

# 結果會自動保存到:
# - experiments/exp_001_k345_tfidf/kaggle_result.json
# - docs/experiment_log.md (自動更新)
```

---

### 實驗 2: Add V/J Gene Features

#### 預期提升: +0.02-0.03

```bash
# 等待 exp_001 完成並且分數 > 0.70 後再執行

python scripts/train_experiment.py \
    --config configs/exp_002_vj_features.yaml \
    --exp-name exp_002_vj_features

python scripts/submit_experiment.py \
    --exp-name exp_002_vj_features \
    --message "Multi-scale k-mer + V/J gene usage features"
```

---

### 實驗 3: Add Diversity Metrics

#### 預期提升: +0.01-0.02

```bash
python scripts/train_experiment.py \
    --config configs/exp_003_diversity.yaml \
    --exp-name exp_003_diversity

python scripts/submit_experiment.py \
    --exp-name exp_003_diversity \
    --message "Multi-scale k-mer + V/J + diversity and clonality metrics"
```

---

## 第三步：比較實驗結果

```bash
# 比較所有 Phase 1 實驗
python scripts/compare_experiments.py \
    --exps exp_001_k345_tfidf exp_002_vj_features exp_003_diversity \
    --baseline-score 0.66987 \
    --output experiments/phase1_comparison.html

# 在瀏覽器中打開
xdg-open experiments/phase1_comparison.html  # Linux
# open experiments/phase1_comparison.html    # macOS
```

---

## 第四步：Phase 1 檢查點（Day 3 晚上）

### 決策樹

```bash
# 檢查最佳實驗的分數
best_score=$(python -c "
import json
from pathlib import Path

scores = []
for exp in ['exp_001_k345_tfidf', 'exp_002_vj_features', 'exp_003_diversity']:
    path = Path(f'experiments/{exp}/kaggle_result.json')
    if path.exists():
        with open(path) as f:
            results = json.load(f)
            if results and 'publicScore' in results[-1]:
                scores.append(results[-1]['publicScore'])

if scores:
    print(max(scores))
else:
    print('N/A')
")

echo "Phase 1 最佳分數: $best_score"

# 決策
if (( $(echo "$best_score >= 0.75" | bc -l) )); then
    echo "✅ Phase 1 成功！繼續執行 Phase 2"
elif (( $(echo "$best_score >= 0.70" | bc -l) )); then
    echo "⚠️ Phase 1 部分成功，檢查 Task B 並繼續 Phase 2"
else
    echo "❌ Phase 1 未達標，需要重新審視策略"
fi
```

---

## 第五步：Phase 2 實驗（Day 4-6）

### 如果 Phase 1 達到 0.75+，執行以下實驗：

#### 實驗 4: Model Ensemble

```bash
# TODO: 需要創建 exp_004_ensemble.yaml

python scripts/train_experiment.py \
    --config configs/exp_004_ensemble.yaml \
    --exp-name exp_004_ensemble

python scripts/submit_experiment.py \
    --exp-name exp_004_ensemble \
    --message "XGBoost + LightGBM + CatBoost ensemble"
```

#### 實驗 5: Cross-dataset Generalization

```bash
# TODO: 需要創建 exp_005_cross_dataset.yaml

python scripts/train_experiment.py \
    --config configs/exp_005_cross_dataset.yaml \
    --exp-name exp_005_cross_dataset

python scripts/submit_experiment.py \
    --exp-name exp_005_cross_dataset \
    --message "Dataset ID features + LODO-CV validation"
```

---

## 第六步：Phase 3 實驗（Day 7-9，如果需要）

### 只有在 Phase 2 後仍未達到 0.82 時才執行

#### 實驗 7: ESM-2 Embeddings

```bash
# TODO: 需要創建 exp_007_esm2.yaml
# 警告: 需要大量 GPU 記憶體和時間

python scripts/train_experiment.py \
    --config configs/exp_007_esm2.yaml \
    --exp-name exp_007_esm2

python scripts/submit_experiment.py \
    --exp-name exp_007_esm2 \
    --message "ESM-2 protein embeddings with attention pooling"
```

---

## 常見問題排除

### Q1: 訓練腳本找不到模組

```bash
# 確保在虛擬環境中
source .venv/bin/activate

# 安裝項目為可編輯模式
pip install -e .

# 或者設置 PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:${PWD}/src"
```

### Q2: GPU 記憶體不足

```bash
# 檢查 GPU 使用情況
nvidia-smi

# 修改配置檔案，降低批次大小或使用 CPU
# 在 config.yaml 中:
# model:
#   xgboost:
#     tree_method: "hist"  # 使用 CPU
#     # 而不是 "gpu_hist"
```

### Q3: 提交檔案格式錯誤

```bash
# 使用驗證工具
python -c "
import pandas as pd
df = pd.read_csv('experiments/exp_001_k345_tfidf/outputs/submission.csv')
print('Rows:', len(df))
print('Columns:', df.columns.tolist())
print('Missing values:', df.isnull().sum().sum())
print('ID format (first 5):', df['ID'].head().tolist())
"
```

### Q4: Kaggle API 認證失敗

```bash
# 確保 kaggle.json 在正確位置
ls -l ~/.kaggle/kaggle.json

# 或者複製到正確位置
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

---

## 每日檢查清單

### 早上（規劃）
- [ ] 檢查昨天提交的結果
- [ ] 更新 docs/experiment_log.md
- [ ] 規劃今天的實驗
- [ ] 檢查 GPU 是否可用

### 晚上（回顧）
- [ ] 記錄今天完成的實驗
- [ ] 分析失敗的原因
- [ ] 比較實驗結果
- [ ] 規劃明天的行動

---

## 緊急應變

### 如果距離截止日期只剩 2 天，但分數 < 0.75

```bash
# 1. 停止所有實驗
# 2. 檢查社群最佳 notebook
# 3. 快速實作社群最佳方法
# 4. 提交所有已完成的版本
# 5. 嘗試不同的 ensemble 組合

# 快速 ensemble 腳本 (TODO)
python scripts/quick_ensemble.py \
    --models exp_001 exp_002 exp_003 \
    --weights 0.4 0.3 0.3 \
    --output emergency_ensemble.csv

python scripts/submit_experiment.py \
    --exp-name emergency \
    --submission-file emergency_ensemble.csv \
    --message "Emergency ensemble of best models"
```

---

## 資源連結

- **戰略路線圖**: `docs/strategic_roadmap_0.67_to_0.82.md`
- **實驗日誌**: `docs/experiment_log.md`
- **模型路線圖**: `docs/model_roadmap.md`
- **排行榜**: https://www.kaggle.com/competitions/adaptive-immune-profiling-challenge-2025/leaderboard

---

**最後更新**: 2025-12-08
**下次更新**: Phase 1 完成後
