# 實驗日誌 - AIRR-ML-25 Competition

> **目標**: 從 0.66987 提升到 0.82+
> **更新日期**: 2025-12-08
> **追蹤所有實驗、提交、和結果**

---

## 實驗追蹤表

| Exp ID | 名稱 | 日期 | Local CV | Public LB | Delta | 狀態 |
|--------|------|------|----------|-----------|-------|------|
| baseline | k=3,4 部分 | 2025-12-08 | N/A | 0.66987 | baseline | ✅ |
| exp_001 | k=3,4,5 TF-IDF | - | - | - | - | 📋 計劃中 |
| exp_002 | + V/J features | - | - | - | - | 📋 計劃中 |
| exp_003 | + Diversity | - | - | - | - | 📋 計劃中 |

---

## 提交歷史

### Baseline - 2025-12-08 07:44
- **描述**: XGBoost k=3,4 k-mer features (Dataset 8: k=3 GPU optimized)
- **狀態**: complete
- **Public Score**: 0.66987
- **備註**: 當前最佳，但 Dataset 8 使用不一致的特徵（k=3 vs k=4）

### submission.csv - 2025-12-08 03:58
- **描述**: k=4 k-mer baseline (CPU) with L1 LogReg - 7 datasets + Dataset 8 baseline
- **狀態**: complete
- **Public Score**: 0.62244
- **備註**: 使用 L1 LogReg，性能較差

### corrected_submission.csv - 2025-12-07 18:09
- **描述**: GPU XGBoost k=3 - corrected ID order
- **狀態**: complete
- **Public Score**: 0.63350
- **備註**: 純 k=3 XGBoost

---

## Phase 1: 快速勝利 (Day 1-3)

### 目標: 0.66987 → 0.75 (+0.08)

#### 實驗計劃

##### exp_001: Multi-scale k-mer (k=3,4,5) with TF-IDF
- **預期提升**: +0.03-0.05
- **時間**: 6 小時
- **優先級**: 最高
- **狀態**: 📋 待執行

**行動清單**:
- [ ] 實作 k=3,4,5 特徵提取
- [ ] 使用 TF-IDF 替代原始計數
- [ ] 稀疏矩陣優化
- [ ] 訓練 8 個 datasets
- [ ] 提交並記錄

**預期 AUC**: 0.70-0.72

---

##### exp_002: V/J Gene Features
- **預期提升**: +0.02-0.03
- **時間**: 4 小時
- **優先級**: 高
- **狀態**: 📋 待執行

**行動清單**:
- [ ] 解析 v_call, j_call
- [ ] 計算 V/J usage frequency
- [ ] 創建 VJ pair features
- [ ] 合併到 k-mer features
- [ ] 提交並記錄

**預期 AUC**: 0.72-0.75

---

##### exp_003: Diversity Metrics
- **預期提升**: +0.01-0.02
- **時間**: 3 小時
- **優先級**: 中
- **狀態**: 📋 待執行

**行動清單**:
- [ ] Shannon entropy
- [ ] Gini coefficient
- [ ] D50
- [ ] CDR3 length statistics
- [ ] 提交並記錄

**預期 AUC**: 0.73-0.76

---

## Phase 2: 中期優化 (Day 4-6)

### 目標: 0.75 → 0.80 (+0.05)

#### 實驗計劃

##### exp_004: Model Ensemble
- **預期提升**: +0.02-0.03
- **時間**: 8 小時
- **優先級**: 高
- **狀態**: 📋 待執行

**模型**:
- XGBoost (current best)
- LightGBM
- CatBoost
- Logistic Regression
- Random Forest

**預期 AUC**: 0.76-0.78

---

##### exp_005: Cross-dataset Generalization
- **預期提升**: +0.01-0.02
- **時間**: 6 小時
- **優先級**: 高
- **狀態**: 📋 待執行

**策略**:
- Dataset ID as feature
- Per-dataset normalization
- LODO-CV

**預期 AUC**: 0.77-0.79

---

##### exp_006: Task B Optimization
- **預期提升**: +0.01-0.02
- **時間**: 5 小時
- **優先級**: 中
- **狀態**: 📋 待執行

**方法**:
- Feature importance mapping
- SHAP values
- Sequence deduplication

**預期 Jaccard**: 從 ~0.3 到 ~0.5

---

## Phase 3: 高風險高回報 (Day 7-9)

### 目標: 0.80 → 0.82+ (+0.02-0.04)

#### 實驗計劃

##### exp_007: ESM-2 Embeddings
- **預期提升**: +0.03-0.05
- **時間**: 12 小時
- **優先級**: 高（如果需要）
- **狀態**: 📋 待執行

**模型**: ESM-2 (650M parameters)
**預期 AUC**: 0.82-0.85

---

##### exp_008: Graph Neural Networks
- **預期提升**: +0.02-0.04
- **時間**: 10 小時
- **優先級**: 低（高風險）
- **狀態**: 📋 待執行

**架構**: GCN/GAT
**預期 AUC**: 0.81-0.84

---

##### exp_009: Meta-learning
- **預期提升**: +0.01-0.03
- **時間**: 8 小時
- **優先級**: 低（高風險）
- **狀態**: 📋 待執行

**方法**: MAML or Prototypical Networks
**預期 AUC**: 0.81-0.83

---

## 分析與洞察

### 當前問題診斷

1. **Dataset 8 特徵不一致**
   - 當前最佳 (0.66987) 使用 k=3 for Dataset 8, k=4 for others
   - 需要統一特徵提取

2. **特徵維度低**
   - k=3: 8,000
   - k=4: 160,000
   - k=5: 3,200,000
   - 需要增加特徵多樣性

3. **缺乏生物學特徵**
   - 沒有 V/J gene 資訊
   - 沒有 clonality metrics
   - 沒有 diversity 指標

4. **Task B 可能拖累總分**
   - 當前方法可能不optimal
   - 需要改進序列重要性排序

### 關鍵決策點

#### Day 3 檢查點
- **如果 >= 0.75**: 繼續 Phase 2
- **如果 0.70-0.74**: 檢查 Task B，調整策略
- **如果 < 0.70**: 緊急切換 Phase 3

#### Day 6 檢查點
- **如果 >= 0.80**: 衝刺 Phase 3，目標 Top 3
- **如果 0.77-0.79**: 繼續優化 ensemble
- **如果 < 0.77**: 回溯檢查實作

#### Day 9 最終決策
- 提交所有準備好的版本
- 選擇最佳 2 個作為 final submission

---

## 排行榜追蹤

### Top 10 團隊（2025-12-08）

| Rank | Team | Score | Gap to Us |
|------|------|-------|-----------|
| 1 | SajayR | 0.83717 | +0.16730 |
| 2 | WoLongFengChu | 0.83623 | +0.16636 |
| 3 | GoBlue | 0.83047 | +0.16060 |
| 4 | GROZD | 0.81998 | +0.15011 |
| 5 | NewBEE | 0.78676 | +0.11689 |
| 6 | Simon Makumi | 0.76886 | +0.09899 |
| 7 | Agneev | 0.76574 | +0.09587 |
| 8 | Tcr | 0.76149 | +0.09162 |
| 9 | whycallmedad | 0.76011 | +0.09024 |
| 10 | Hi F | 0.75932 | +0.08945 |

**我們的位置**: 未上榜 (需要 0.74147 才能進入 Top 18)

---

## 待辦事項

### 立即行動（今天）
- [ ] 執行 exp_001 (k=3,4,5 TF-IDF)
- [ ] 驗證提交檔案格式
- [ ] 提交並追蹤結果

### 短期（Day 1-3）
- [ ] 完成 Phase 1 所有實驗
- [ ] 達到 0.75+ 目標
- [ ] 更新實驗日誌

### 中期（Day 4-6）
- [ ] 開始 ensemble 開發
- [ ] 優化 Task B
- [ ] 達到 0.80+ 目標

### 長期（Day 7-9）
- [ ] 評估是否需要深度學習
- [ ] 最終衝刺
- [ ] 達到 0.82+ 目標

---

## 提交策略

### 保守策略（推薦）
- Day 1: 1 次
- Day 2: 2 次
- Day 3: 2 次
- Day 4-6: 每天 2-3 次
- Day 7-9: 每天 3-5 次
- **Total**: 24 次（保留 21 次備用）

### 激進策略（如果落後）
- Day 1-3: 每天 3 次
- Day 4-6: 每天 4 次
- Day 7-9: 每天 5 次
- **Total**: 36 次（保留 9 次備用）

---

**下次更新**: Day 1 晚上（exp_001 完成後）
