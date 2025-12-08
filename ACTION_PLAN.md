# 行動計劃：從 0.67 到 0.82+ 的 9 天作戰

> **當前**: 0.66987 | **目標**: 0.82+ | **剩餘**: 9 天

---

## 今天就開始 (Day 1: Dec 8)

### 早上 (3 小時)

```bash
# 1. 檢查環境
source .venv/bin/activate
python -c "import xgboost; print(xgboost.__version__)"
nvidia-smi

# 2. 閱讀戰略文檔
cat docs/strategic_roadmap_0.67_to_0.82.md | less
cat docs/QUICKSTART.md | less

# 3. 開始第一個實驗
python scripts/train_experiment.py \
    --config configs/exp_001_k345_tfidf.yaml \
    --exp-name exp_001_k345_tfidf
```

**預期結果**: 訓練需要 1-2 小時

### 下午 (3 小時)

```bash
# 4. 等待訓練完成，檢查結果
cat experiments/exp_001_k345_tfidf/logs/*.log | tail -50

# 5. 驗證提交檔案
python -c "
import pandas as pd
df = pd.read_csv('experiments/exp_001_k345_tfidf/outputs/submission.csv')
print('Rows:', len(df))
print('Missing:', df.isnull().sum().sum())
"

# 6. 提交到 Kaggle
python scripts/submit_experiment.py \
    --exp-name exp_001_k345_tfidf \
    --message "Multi-scale k-mer (3,4,5) with TF-IDF weighting"
```

**預期結果**: Public score 0.70-0.72 (+0.03-0.05)

### 晚上 (2 小時)

```bash
# 7. 如果 exp_001 成功 (>0.70)，開始 exp_002
python scripts/train_experiment.py \
    --config configs/exp_002_vj_features.yaml \
    --exp-name exp_002_vj_features

# 8. 更新實驗日誌
vim docs/experiment_log.md

# 9. 規劃明天
echo "Tomorrow: Submit exp_002, start exp_003" >> NEXT_SESSION_TODO.md
```

---

## Day 2 (Dec 9)

### 目標: 達到 0.73-0.75

```bash
# 早上: 提交 exp_002
python scripts/submit_experiment.py \
    --exp-name exp_002_vj_features \
    --message "Multi-scale k-mer + V/J gene usage features"

# 下午: 訓練並提交 exp_003
python scripts/train_experiment.py \
    --config configs/exp_003_diversity.yaml \
    --exp-name exp_003_diversity

python scripts/submit_experiment.py \
    --exp-name exp_003_diversity \
    --message "Multi-scale k-mer + V/J + diversity metrics"

# 晚上: 比較 Phase 1 所有實驗
python scripts/compare_experiments.py \
    --exps exp_001_k345_tfidf exp_002_vj_features exp_003_diversity \
    --baseline-score 0.66987 \
    --output experiments/phase1_comparison.html
```

**預期結果**: 最佳分數 0.73-0.75

---

## Day 3 (Dec 10)

### 里程碑檢查: Phase 1

```bash
# 早上: 檢查所有 Phase 1 結果
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

print(max(scores) if scores else 0)
")

echo "Phase 1 最佳分數: $best_score"
```

### 決策點

**如果 >= 0.75**: 繼續 Phase 2
```bash
# 開始設計 ensemble
vim configs/exp_004_ensemble.yaml
```

**如果 0.70-0.74**: 檢查 Task B
```bash
# 分析 Task B 的 Jaccard score
# 可能需要改進序列選擇邏輯
```

**如果 < 0.70**: 緊急調整
```bash
# 檢查社群最佳 notebook
# 快速實作社群方法
```

---

## Day 4-6: Phase 2 (中期優化)

### 目標: 0.75 → 0.80

#### Day 4: Ensemble 開發
```bash
# 訓練多個模型
python scripts/train_experiment.py --config configs/exp_004_ensemble.yaml --exp-name exp_004_ensemble
python scripts/submit_experiment.py --exp-name exp_004_ensemble --message "XGBoost+LightGBM+CatBoost ensemble"
```

#### Day 5: Cross-dataset Optimization
```bash
# 優化跨資料集泛化
python scripts/train_experiment.py --config configs/exp_005_cross_dataset.yaml --exp-name exp_005_cross_dataset
python scripts/submit_experiment.py --exp-name exp_005_cross_dataset --message "Dataset ID features + LODO-CV"
```

#### Day 6: Task B Optimization
```bash
# 改進序列選擇
python scripts/train_experiment.py --config configs/exp_006_task_b.yaml --exp-name exp_006_task_b
python scripts/submit_experiment.py --exp-name exp_006_task_b --message "SHAP-based sequence ranking"
```

**預期結果**: Day 6 晚上達到 0.78-0.80

---

## Day 7-9: Phase 3 (最終衝刺)

### 目標: 0.80 → 0.82+

#### 決策點 (Day 7 早上)

**如果 >= 0.80**: 穩健優化
- 繼續優化 ensemble
- 微調超參數
- 目標 0.82-0.83

**如果 0.77-0.79**: 嘗試 ESM-2
```bash
# 安裝依賴
pip install transformers fair-esm

# 訓練 ESM-2 模型
python scripts/train_experiment.py --config configs/exp_007_esm2.yaml --exp-name exp_007_esm2

# 預計需要 6-12 小時
```

**如果 < 0.77**: 緊急模式
- 並行嘗試所有 Phase 3 方法
- 提交所有已完成的版本
- 測試不同的 ensemble 權重

---

## 提交節奏控制

### 保守策略 (推薦)

| Day | 實驗 | 提交次數 | 累計 |
|-----|------|---------|------|
| 1 | exp_001 | 1 | 1 |
| 2 | exp_002, exp_003 | 2 | 3 |
| 3 | 最佳版本 | 1 | 4 |
| 4 | exp_004 | 2 | 6 |
| 5 | exp_005 | 2 | 8 |
| 6 | exp_006 | 3 | 11 |
| 7 | exp_007 | 3 | 14 |
| 8 | 優化版本 | 4 | 18 |
| 9 | 最終衝刺 | 5 | 23 |

**剩餘**: 22 次備用

---

## 每日檢查清單

### 早上 (30 分鐘)
- [ ] 檢查昨天提交的結果
- [ ] 更新 docs/experiment_log.md
- [ ] 檢查排行榜變化
- [ ] 規劃今天的實驗

### 進行中 (持續)
- [ ] 監控訓練進度
- [ ] 記錄異常和錯誤
- [ ] 保存中間結果

### 晚上 (30 分鐘)
- [ ] 記錄今天完成的實驗
- [ ] 分析失敗的原因
- [ ] 比較實驗結果
- [ ] 規劃明天的行動
- [ ] Git commit 和 push

---

## 緊急應變計劃

### 如果 Day 3 後 < 0.70
1. 停止當前策略
2. 檢查社群最佳 notebook
3. 快速實作社群方法
4. 重新評估特徵工程

### 如果 Day 6 後 < 0.77
1. 重新審視 Task B 實作
2. 檢查是否有明顯的 bug
3. 考慮使用預訓練模型
4. 增加提交頻率

### 如果 Day 9 早上 < 0.80
1. 提交所有已完成的版本
2. 測試不同的 ensemble 權重
3. 調整 Task B 的 top-k threshold
4. 使用所有剩餘提交次數

---

## 關鍵成功因素

### 必須做到
1. 每個實驗都要有完整記錄
2. 使用正確的驗證策略 (LODO-CV)
3. 不要 overfit public leaderboard
4. 保持代碼模組化和可重現

### 應該避免
1. 連續熬夜（效率會下降）
2. 追逐 public LB（會 overfit）
3. 同時進行多個大改動
4. 忘記 Git commit

### 可以彈性調整
1. 實驗順序（根據結果調整）
2. 提交節奏（根據進度調整）
3. Phase 3 方法（根據需要調整）

---

## 勝算評估

| 目標 | 機率 | 條件 |
|------|------|------|
| 0.75+ (進入排行榜) | 85% | 完成 Phase 1 |
| 0.80+ (進入前十) | 60% | 完成 Phase 2 |
| 0.82+ (超越 GROZD) | 35% | 需要 Phase 3 成功 |

---

## 立即行動

### 現在就做這些

1. **檢查環境**
   ```bash
   source .venv/bin/activate
   python -c "import xgboost, pandas, numpy; print('OK')"
   ```

2. **閱讀文檔** (15 分鐘)
   - docs/strategic_roadmap_0.67_to_0.82.md
   - docs/QUICKSTART.md

3. **開始第一個實驗** (2 小時)
   ```bash
   python scripts/train_experiment.py \
       --config configs/exp_001_k345_tfidf.yaml \
       --exp-name exp_001_k345_tfidf
   ```

4. **等待結果** (1 小時)
   - 監控訓練日誌
   - 準備提交

5. **提交並記錄**
   ```bash
   python scripts/submit_experiment.py \
       --exp-name exp_001_k345_tfidf \
       --message "Multi-scale k-mer (3,4,5) with TF-IDF"
   ```

---

## 需要幫助？

### 當前狀態
- 戰略路線圖: ✅ 已完成
- 實驗管理系統: ✅ 已完成
- 配置模板: ✅ 已完成
- 執行腳本: ✅ 已完成

### 待實現
- [ ] src/airr_ml25/features/ 模組
- [ ] src/airr_ml25/models/ 模組
- [ ] 訓練邏輯串接
- [ ] Task B 實作

### 下一步
1. 實作特徵提取模組
2. 實作模型訓練邏輯
3. 測試完整流程
4. 開始真正的實驗

---

**開始時間**: 現在
**預計完成**: 2025-12-16
**最終提交**: 2025-12-17 06:59 UTC

**記住**:
- 每天進步一點點
- 記錄所有實驗
- 不要 overfit public LB
- 享受過程！

---

**創建者**: Competition Master Agent
**日期**: 2025-12-08
