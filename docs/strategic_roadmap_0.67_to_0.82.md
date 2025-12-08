# AIRR-ML-25 戰略優化路線圖：從 0.67 到 0.82+

> **制定日期**: 2025-12-08
> **當前最佳分數**: 0.66987 (k=3,4 XGBoost)
> **目標分數**: 0.82+ (超越 Top 3: 0.81998)
> **剩餘時間**: 9 天 (截止日期: 2025-12-17 06:59 UTC)
> **剩餘提交**: 5 次/天 × 9 天 = 45 次
> **當前排名**: 未上榜 (Top 18 最低分: 0.74147)

---

## 競爭形勢分析

### 頂尖團隊得分分布
```
Rank 1: SajayR           0.83717  (+0.16730 vs 我們)
Rank 2: WoLongFengChu    0.83623  (+0.16636)
Rank 3: GoBlue           0.83047  (+0.16060)
Rank 4: GROZD            0.81998  (+0.15011)
----- 0.80 分界線 -----
Rank 5: NewBEE           0.78676  (+0.11689)
----- 0.75 分界線 -----
Rank 18: GPCH2159        0.74147  (+0.07160)
----- 我們的位置 -----
Current: 0.66987         (需要 +0.07160 才能上榜)
```

### 關鍵洞察
1. **Top 4 集中在 0.81-0.84**: 這是明顯的技術突破點
2. **0.75-0.79 是中等水平**: 需要跨越的第一道門檻
3. **我們落後 0.16+**: 需要系統性改進，不是調參能解決的

---

## 階段性目標與時間分配

### 第一里程碑：進入排行榜 (0.75+)
**目標**: 提升 +0.08，達到 0.75
**時間**: Day 1-3 (Dec 8-10)
**提交預算**: 10 次

### 第二里程碑：進入前十 (0.80+)
**目標**: 提升 +0.05，達到 0.80
**時間**: Day 4-6 (Dec 11-13)
**提交預算**: 15 次

### 最終目標：超越 GROZD (0.82+)
**目標**: 提升 +0.02，達到 0.82
**時間**: Day 7-9 (Dec 14-16)
**提交預算**: 20 次

---

## Phase 1: 快速勝利 (Quick Wins) - Day 1-3

### 目標：0.66987 → 0.75 (+0.08)

### 1.1 多尺度 k-mer 優化 (預期 +0.03-0.05)
**時間**: 6 小時
**難度**: 低
**風險**: 極低
**實施**:
```python
# 當前: k=3,4 (部分 dataset)
# 改進: k=3,4,5 (全部 8 datasets 一致)

Feature dimension:
- k=3: 8,000 features
- k=4: 160,000 features
- k=5: 3,200,000 features (稀疏矩陣)
- Total: ~3.37M features (TF-IDF weighted)
```

**行動清單**:
- [ ] 修改特徵提取函數支援 k=[3,4,5]
- [ ] 使用 TF-IDF 替代原始計數
- [ ] 稀疏矩陣處理 (scipy.sparse)
- [ ] GPU XGBoost 優化記憶體使用
- [ ] 訓練 8 個 dataset (預估 2 小時)
- [ ] 生成提交檔案並提交

**預期 AUC**: 0.70-0.72

---

### 1.2 V/J 基因特徵工程 (預期 +0.02-0.03)
**時間**: 4 小時
**難度**: 低
**風險**: 低
**實施**:
```python
# 新增特徵：
1. V gene usage (one-hot, ~300 features)
2. J gene usage (one-hot, ~50 features)
3. VJ pair combinations (top 500 pairs)
4. V/J family usage (aggregated)
```

**行動清單**:
- [ ] 解析 v_call, j_call 欄位
- [ ] 計算每個 repertoire 的 V/J 使用頻率
- [ ] 創建 VJ pair 共現矩陣
- [ ] 合併到現有特徵矩陣
- [ ] 重新訓練並提交

**預期 AUC**: 0.72-0.75

---

### 1.3 克隆性與多樣性指標 (預期 +0.01-0.02)
**時間**: 3 小時
**難度**: 低
**風險**: 低
**實施**:
```python
# 新增統計特徵：
1. Shannon entropy (序列多樣性)
2. Gini coefficient (克隆擴增度)
3. D50 (前 50% 序列所需的克隆型數)
4. CDR3 length distribution (mean, std, skew, kurtosis)
5. Unique sequence ratio
```

**行動清單**:
- [ ] 實作多樣性指標計算函數
- [ ] 為每個 repertoire 計算統計特徵
- [ ] 添加到特徵矩陣
- [ ] 特徵標準化處理
- [ ] 重新訓練並提交

**預期 AUC**: 0.73-0.76

---

### Phase 1 里程碑檢查點 (Day 3 晚上)
**如果達到 0.75+**:
- 繼續執行 Phase 2

**如果僅達到 0.70-0.74**:
- 檢查特徵重要性
- 調試 Task B 的 Jaccard score
- 可能需要調整策略

**如果低於 0.70**:
- 緊急切換到 Phase 3 的深度學習方法

---

## Phase 2: 中期優化 (Medium-term) - Day 4-6

### 目標：0.75 → 0.80 (+0.05)

### 2.1 模型集成策略 (預期 +0.02-0.03)
**時間**: 8 小時
**難度**: 中
**風險**: 中
**實施**:
```python
# Ensemble 架構：
1. XGBoost (current best)
2. LightGBM (faster, different splits)
3. CatBoost (handles categorical V/J genes)
4. Logistic Regression (simple baseline)
5. Random Forest (diversity)

# 融合策略：
- Task A: Weighted averaging (optimize weights by validation AUC)
- Task B: Union of top sequences with voting threshold
```

**行動清單**:
- [ ] 訓練 5 個不同模型（每個 dataset）
- [ ] 使用 StratifiedKFold (5-fold) 調校權重
- [ ] 實作 soft voting for Task A
- [ ] 實作 sequence union for Task B
- [ ] 提交最佳 ensemble

**預期 AUC**: 0.76-0.78

---

### 2.2 跨資料集泛化優化 (預期 +0.01-0.02)
**時間**: 6 小時
**難度**: 中
**風險**: 中
**實施**:
```python
# 問題：不同 dataset 有不同的：
# - 測序平台 (sequencing platform)
# - 讀深度 (read depth)
# - 疾病類型 (disease type)
# - 人口統計 (demographics)

# 解決方案：
1. Dataset ID as feature (one-hot encoding)
2. Per-dataset normalization
3. Domain adaptation techniques
4. Leave-one-dataset-out CV
```

**行動清單**:
- [ ] 分析 8 個 dataset 的統計差異
- [ ] 添加 dataset ID one-hot features
- [ ] 實作 per-dataset 特徵標準化
- [ ] 使用 LODO-CV 驗證泛化能力
- [ ] 調整模型使其對 dataset shift 魯棒

**預期 AUC**: 0.77-0.79

---

### 2.3 Task B 優化：序列重要性排序 (預期 +0.01-0.02)
**時間**: 5 小時
**難度**: 中
**風險**: 低
**實施**:
```python
# 當前問題：Task B 可能拖累總分

# 改進策略：
1. 使用模型的 feature importance 直接映射到序列
2. 計算每個序列的 SHAP value
3. 基於 k-mer 重要性聚合序列得分
4. 去除冗餘序列 (高相似度序列只保留一個)
```

**行動清單**:
- [ ] 提取 XGBoost feature importance
- [ ] 映射 k-mer 回原始序列
- [ ] 計算序列層級的重要性得分
- [ ] 去重並選擇 top 50,000
- [ ] 驗證 Jaccard similarity 提升

**預期 Jaccard**: 從 ~0.3 提升到 ~0.5

---

### Phase 2 里程碑檢查點 (Day 6 晚上)
**如果達到 0.80+**:
- 準備衝刺 Phase 3，目標 Top 3

**如果達到 0.77-0.79**:
- 繼續優化 ensemble 和 Task B
- 可能需要嘗試深度學習

**如果低於 0.77**:
- 回溯檢查 Phase 1 的實作
- 可能需要重新審視特徵工程

---

## Phase 3: 高風險高回報 (High-risk High-reward) - Day 7-9

### 目標：0.80 → 0.82+ (+0.02-0.04)

### 3.1 蛋白質語言模型嵌入 (預期 +0.03-0.05)
**時間**: 12 小時
**難度**: 高
**風險**: 高
**實施**:
```python
# 使用預訓練模型：
1. ESM-2 (Meta AI): 最先進的蛋白質語言模型
   - Model: esm2_t33_650M_UR50D (650M parameters)
   - Output: 1280-dim embeddings per sequence

2. ProtBERT (Rostlab): BERT for proteins
   - Model: prot_bert_bfd
   - Output: 1024-dim embeddings

3. Antiberty (Sanofi): Specialized for antibody sequences
   - Model: antiberty
   - Output: 512-dim embeddings

# Repertoire aggregation:
- Mean pooling
- Max pooling
- Attention-weighted pooling
- Statistical moments (mean, std, skew, kurtosis)
```

**行動清單**:
- [ ] 安裝 transformers, fair-esm 套件
- [ ] 下載 ESM-2 模型 (約 2.5 GB)
- [ ] 批次處理所有序列 (GPU 加速)
- [ ] 實作多種 pooling 策略
- [ ] 與現有特徵合併
- [ ] 訓練深度模型 (MLP or Transformer)
- [ ] 提交並驗證效果

**挑戰**:
- 記憶體需求大 (每個 repertoire 數千條序列)
- 推論時間長 (可能需要 4-6 小時)
- 可能需要降維 (PCA/UMAP)

**預期 AUC**: 0.82-0.85 (如果成功)

---

### 3.2 圖神經網路：序列相似性圖 (預期 +0.02-0.04)
**時間**: 10 小時
**難度**: 極高
**風險**: 極高
**實施**:
```python
# 架構：
1. 構建序列相似性圖：
   - 節點: 每條 CDR3 序列
   - 邊: 基於編輯距離 < threshold 的序列對

2. 圖特徵：
   - Node features: k-mer embeddings
   - Edge features: alignment score, V/J gene match

3. GNN 模型：
   - Graph Convolutional Network (GCN)
   - Graph Attention Network (GAT)
   - 或 GraphSAINT for large graphs

4. Repertoire representation:
   - Graph-level pooling (global mean/max)
   - Hierarchical graph classification
```

**行動清單**:
- [ ] 使用 pytorch-geometric 或 DGL
- [ ] 計算序列間編輯距離 (高效實作)
- [ ] 構建稀疏鄰接矩陣
- [ ] 設計 GNN 架構
- [ ] 訓練並調參
- [ ] 與其他模型 ensemble

**挑戰**:
- 圖構建非常耗時
- 每個 repertoire 可能有數千個節點
- 需要高效的圖採樣策略

**預期 AUC**: 0.81-0.84 (如果成功)

---

### 3.3 Meta-learning: 學習如何學習新 dataset (預期 +0.01-0.03)
**時間**: 8 小時
**難度**: 高
**風險**: 高
**實施**:
```python
# 動機：測試集有 11 個 dataset，訓練集只有 8 個
# 需要模型快速適應新的 dataset

# 方法：
1. Model-Agnostic Meta-Learning (MAML)
2. Prototypical Networks
3. 或簡化版：multi-task learning with dataset-specific heads

# 架構：
- Shared encoder (學習通用 repertoire 表徵)
- Dataset-specific decoders (處理 dataset shift)
- Meta-learning 目標：最小化 adaptation loss
```

**行動清單**:
- [ ] 實作 MAML 或 Prototypical Networks
- [ ] 設計 few-shot learning 策略
- [ ] 使用 support/query split 模擬新 dataset
- [ ] 訓練 meta-learner
- [ ] 在測試集上快速 fine-tune
- [ ] 提交並評估

**挑戰**:
- 複雜度高，容易出錯
- 需要仔細設計驗證策略
- 訓練時間長

**預期 AUC**: 0.81-0.83 (如果成功)

---

### Phase 3 決策樹

```
Day 7 早上評估：

IF current_score >= 0.80:
    → 穩健策略：優化 ensemble + Task B
    → 目標：0.82-0.83

ELIF current_score >= 0.77:
    → 嘗試 ESM-2 embeddings (3.1)
    → 如果 24 小時內無進展，回退到 ensemble

ELIF current_score < 0.77:
    → 緊急模式：並行嘗試所有 Phase 3 方法
    → 孤注一擲策略
```

---

## 實驗管理策略

### 目錄結構重構

```
airr-ml25-package/
├── experiments/                  # 新增：所有實驗的統一目錄
│   ├── exp_001_k345_tfidf/      # 實驗編號 + 簡短描述
│   │   ├── config.yaml           # 超參數配置
│   │   ├── train.log             # 訓練日誌
│   │   ├── metrics.json          # 驗證指標
│   │   ├── model.pkl             # 訓練好的模型
│   │   ├── submission.csv        # 提交檔案
│   │   └── kaggle_result.txt     # Kaggle 回傳分數
│   │
│   ├── exp_002_vj_features/
│   ├── exp_003_diversity/
│   ├── exp_004_ensemble_xgb_lgb/
│   └── ...
│
├── src/airr_ml25/
│   ├── features/                 # 特徵工程模組
│   │   ├── kmer.py              # k-mer extraction
│   │   ├── vj_gene.py           # V/J gene features
│   │   ├── diversity.py         # Clonality metrics
│   │   └── embeddings.py        # ESM-2, ProtBERT
│   │
│   ├── models/                   # 模型模組
│   │   ├── xgboost_model.py
│   │   ├── lightgbm_model.py
│   │   ├── ensemble.py
│   │   └── deep_models.py       # MLP, GNN
│   │
│   ├── tasks/                    # 任務模組
│   │   ├── task_a.py            # Probability prediction
│   │   └── task_b.py            # Sequence identification
│   │
│   └── utils/
│       ├── experiment.py         # 實驗管理工具
│       ├── validation.py         # CV strategies
│       └── submission.py         # 提交檔案生成
│
├── configs/                      # 配置檔案
│   ├── baseline.yaml
│   ├── ensemble.yaml
│   └── esm2.yaml
│
├── scripts/                      # 執行腳本
│   ├── train_experiment.py       # 統一訓練入口
│   ├── evaluate_experiment.py    # 統一評估入口
│   ├── submit_experiment.py      # 統一提交入口
│   └── compare_experiments.py    # 實驗比較工具
│
└── docs/
    ├── experiment_log.md         # 實驗日誌（本檔案）
    └── leaderboard_tracker.md    # 排行榜追蹤
```

---

### 實驗追蹤工作流程

#### 1. 啟動新實驗
```bash
# 自動創建實驗目錄並初始化
python scripts/train_experiment.py \
    --config configs/exp_001_k345_tfidf.yaml \
    --exp-name exp_001_k345_tfidf

# 內部流程：
# 1. 創建 experiments/exp_001_k345_tfidf/
# 2. 複製 config.yaml
# 3. 初始化日誌
# 4. 開始訓練
# 5. 自動保存 model, metrics, submission
```

#### 2. 評估實驗
```bash
# 本地驗證
python scripts/evaluate_experiment.py \
    --exp-name exp_001_k345_tfidf \
    --metrics auc roc accuracy

# 輸出：
# - Validation AUC per dataset
# - Cross-validation scores
# - Task B Jaccard estimate
```

#### 3. 提交實驗
```bash
# 提交到 Kaggle
python scripts/submit_experiment.py \
    --exp-name exp_001_k345_tfidf \
    --message "Multi-scale k-mer (3,4,5) with TF-IDF"

# 內部流程：
# 1. 驗證 submission.csv 格式
# 2. 使用 Kaggle API 提交
# 3. 等待結果
# 4. 保存到 kaggle_result.txt
# 5. 更新 docs/experiment_log.md
```

#### 4. 比較實驗
```bash
# 生成對比報告
python scripts/compare_experiments.py \
    --exps exp_001 exp_002 exp_003 \
    --output experiments/comparison.html

# 輸出：
# - 特徵重要性對比
# - AUC 曲線對比
# - 超參數差異表
```

---

### 每日提交策略

#### 保守策略 (推薦)
```
Day 1: 1 次提交 (驗證 k=3,4,5 改進)
Day 2: 2 次提交 (V/J features + diversity)
Day 3: 2 次提交 (最佳單模型 + ensemble)
----- Milestone 1: 應該達到 0.75 -----

Day 4: 2 次提交 (優化 ensemble)
Day 5: 2 次提交 (dataset adaptation)
Day 6: 3 次提交 (Task B 優化 + 最佳版本)
----- Milestone 2: 應該達到 0.80 -----

Day 7: 3 次提交 (ESM-2 初版)
Day 8: 4 次提交 (ESM-2 優化 + ensemble)
Day 9: 5 次提交 (最終衝刺，所有版本)
----- Final Goal: 0.82+ -----

Total: 24 次提交 (保留 21 次作為備用)
```

#### 激進策略 (如果進度落後)
```
Day 1-3: 每天 3 次提交 (快速迭代)
Day 4-6: 每天 4 次提交 (中期優化)
Day 7-9: 每天 5 次提交 (最終衝刺)

Total: 36 次提交 (保留 9 次作為備用)
```

---

### 避免 Overfitting Public Leaderboard

#### 策略 1: 本地驗證優先
```python
# 永遠相信本地 CV，而非 public LB

# 如果：
local_cv_auc = 0.78
public_lb_auc = 0.75

# 說明：
# - Local CV 可能更可靠（如果設計得當）
# - Public LB 可能有 data leakage 或偏差
# - 繼續優化 local CV，不要追逐 public LB
```

#### 策略 2: Stratified CV + LODO-CV
```python
# 使用多種驗證策略：

1. Stratified 5-Fold CV (within each dataset)
   → 評估模型穩定性

2. Leave-One-Dataset-Out CV
   → 評估泛化能力（最接近 private LB）

3. Time-based split (如果有時間資訊)
   → 避免 temporal leakage

# 提交決策：
if (lodo_cv_auc > current_best) and (stratified_cv_std < 0.05):
    submit()
else:
    continue_tuning()
```

#### 策略 3: Ensemble 多樣性
```python
# 不要 ensemble 太相似的模型

# 確保 ensemble 成員具備：
1. 不同的模型架構 (XGBoost, LightGBM, NN)
2. 不同的特徵子集 (k-mer only, VJ only, combined)
3. 不同的訓練資料 (bagging, different CV folds)

# 計算 ensemble 多樣性：
from sklearn.metrics import matthews_corrcoef

diversity = []
for i, model_i in enumerate(models):
    for j, model_j in enumerate(models):
        if i < j:
            pred_i = model_i.predict(X_val)
            pred_j = model_j.predict(X_val)
            div = 1 - matthews_corrcoef(pred_i, pred_j)
            diversity.append(div)

print(f"Average diversity: {np.mean(diversity):.3f}")
# 目標: > 0.3
```

#### 策略 4: 提交節奏控制
```python
# 避免過度依賴 public feedback

Rule 1: 每個主要改進只提交 1 次
- 不要反覆調參然後提交（會 overfit）

Rule 2: 每次提交必須有本地驗證支持
- Local CV improvement > 0.01 才考慮提交

Rule 3: 保留 20% 提交次數到最後 2 天
- 最後衝刺可能需要快速迭代

Rule 4: 如果連續 3 次提交沒有改進
- 停止提交，回去檢查驗證策略
```

---

## 風險管理與應變計劃

### 風險 1: Phase 1 後仍低於 0.70
**應變**:
- 立即檢查 Task B 是否拖累分數
- 分析 per-dataset AUC，找出最弱的 dataset
- 考慮使用社群最佳 notebook 的特徵
- 縮短 Phase 2，直接進入 Phase 3

### 風險 2: ESM-2 模型訓練失敗或 OOM
**應變**:
- 使用較小的 ESM-2 模型 (t12_35M 而非 t33_650M)
- 批次大小減半
- 使用梯度累積
- 或放棄 ESM-2，專注於 ensemble

### 風險 3: 時間不足，Day 9 仍未達 0.82
**應變**:
- 提交所有已訓練的版本
- 嘗試不同的 ensemble 權重（簡單但有效）
- 調整 Task B 的 top-k threshold
- 最後一天提交 5 次，測試不同組合

### 風險 4: Private LB 與 Public LB 差異大
**應變**:
- 這就是為什麼要用 LODO-CV
- 信任本地驗證，不要 overfit public LB
- 如果最後發現差距大，已經來不及了
- **預防勝於治療：從一開始就用正確的驗證策略**

---

## 關鍵成功因素 (KSF)

### 技術面
1. **特徵工程質量** > 模型選擇
   - 免疫組學領域，domain knowledge 很重要
   - k-mer, V/J gene, clonality 是基礎

2. **Task B 不能拖累** Task A
   - Task B 的 Jaccard score 可能影響很大
   - 必須確保 Task B 的邏輯正確

3. **Ensemble 多樣性** > Ensemble 大小
   - 3 個互補的模型 > 10 個相似的模型
   - 關注 model disagreement

### 執行面
1. **每天檢查進度**
   - 使用 docs/experiment_log.md 追蹤
   - 晚上 review，早上規劃

2. **快速失敗，快速修正**
   - 如果一個想法 4 小時沒進展，放棄
   - 不要陷入 sunk cost fallacy

3. **保持程式碼整潔**
   - 模組化，可重用
   - 良好的實驗管理很重要

---

## 每日行動清單

### Day 1 (Dec 8)
- [ ] 實作 k=3,4,5 multi-scale k-mer
- [ ] 訓練 8 個 dataset
- [ ] 提交並記錄分數
- [ ] 如果 > 0.70，繼續 V/J features

### Day 2 (Dec 9)
- [ ] 實作 V/J gene features
- [ ] 實作 diversity metrics
- [ ] 重新訓練並提交
- [ ] 目標：0.73-0.75

### Day 3 (Dec 10)
- [ ] 檢查 Phase 1 成果
- [ ] 開始 ensemble 開發
- [ ] 提交最佳版本
- [ ] **Milestone check**: 必須達到 0.75+

### Day 4 (Dec 11)
- [ ] 完成 ensemble (XGBoost + LightGBM + CatBoost)
- [ ] 優化 Task B sequence selection
- [ ] 提交 ensemble 版本

### Day 5 (Dec 12)
- [ ] Dataset adaptation
- [ ] Cross-dataset validation
- [ ] 目標：0.78-0.80

### Day 6 (Dec 13)
- [ ] 檢查 Phase 2 成果
- [ ] **Milestone check**: 必須達到 0.80+
- [ ] 決定是否進入 Phase 3

### Day 7 (Dec 14)
- [ ] 開始 ESM-2 開發（如果需要）
- [ ] 或繼續優化 ensemble

### Day 8 (Dec 15)
- [ ] ESM-2 ensemble
- [ ] 或嘗試 meta-learning
- [ ] 目標：0.82+

### Day 9 (Dec 16)
- [ ] 最終優化
- [ ] 提交所有變體
- [ ] 選擇最佳 2 個作為 final submission

---

## 預期結果總結

| Phase | Days | 目標 AUC | 預期提升 | 風險 | 信心度 |
|-------|------|---------|---------|------|--------|
| Phase 1 | 1-3 | 0.75 | +0.08 | 低 | 85% |
| Phase 2 | 4-6 | 0.80 | +0.05 | 中 | 70% |
| Phase 3 | 7-9 | 0.82+ | +0.02 | 高 | 50% |

**總體勝算**:
- 達到 0.75 (進入排行榜): 85%
- 達到 0.80 (進入前十): 60%
- 達到 0.82 (超越 GROZD): 35%

**關鍵變數**:
1. Task B 的實作是否正確（影響最大）
2. Ensemble 的多樣性（第二重要）
3. 是否能成功整合 ESM-2（bonus）

---

## 最後提醒

1. **不要追逐 Public LB**
   - 相信本地 CV
   - LODO-CV 最接近 Private LB

2. **保持程式碼可重現**
   - 固定隨機種子
   - 記錄所有超參數
   - 使用 Git 版本控制

3. **健康第一**
   - 不要連續熬夜
   - 每天保持 6+ 小時睡眠
   - 效率 > 工作時間

4. **享受過程**
   - 這是學習免疫組學 ML 的好機會
   - 即使沒拿第一，經驗也很寶貴
   - 記錄所有學習到的東西

---

**制定者**: Claude (Competition Master Agent)
**批准者**: 等待 User 確認
**版本**: 1.0
**下次更新**: Day 3 (檢查 Phase 1 成果後)
