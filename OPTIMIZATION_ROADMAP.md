# AIRR-ML-25 優化路線圖
**當前分數**: 0.66987 → **目標分數**: 0.82+
**時間**: 2025-12-08 至 2025-12-17 (9 天)
**每日提交限制**: 5 次/天

---

## 🎯 快速勝利策略 (Quick Wins - 1-2天)

### 1. Multi-scale k-mer Features (預期提升: +0.03-0.05)
**當前問題**: Dataset 8 使用 k=3，其他使用 k=4，不一致且未充分利用
**解決方案**:
- 對所有 datasets 統一使用 k=3,4,5 組合
- 使用 sparse matrix 節省記憶體
- GPU 加速 k-mer 計數（將來優化）

**實施步驟**:
```python
# 1. 修改 train_gpu_k3.py 支持多 k 值
# 2. 訓練 k=[3,4,5] 模型
# 3. 提交並比較分數
```

**預期時間**: 4-6 小時（訓練時間）
**風險**: 低 - 已有 k=3 和 k=4 經驗

---

### 2. V/J Gene Usage Features (預期提升: +0.02-0.04)
**當前問題**: 只使用序列特徵，忽略基因使用模式
**解決方案**:
- 提取 top-20 V gene usage frequencies
- 提取 top-20 J gene usage frequencies
- 加入 VJ pair combinations

**實施步驟**:
```python
# src/airr_ml25/features/vj_usage.py
def extract_vj_features(df):
    v_usage = df['v_call'].value_counts().head(20) / len(df)
    j_usage = df['j_call'].value_counts().head(20) / len(df)
    vj_pairs = df.groupby(['v_call', 'j_call']).size().head(50) / len(df)
    return pd.concat([v_usage, j_usage, vj_pairs])
```

**預期時間**: 2-3 小時
**風險**: 低 - 簡單特徵工程

---

### 3. Per-Dataset Model + Ensemble (預期提升: +0.04-0.07)
**當前問題**: 單一模型處理 8 個不同 datasets，可能欠擬合
**解決方案**:
- 為每個 dataset 訓練專門模型
- 使用 weighted ensemble 或 stacking

**實施步驟**:
```bash
# 1. 並行訓練 8 個模型
python scripts/train_per_dataset.py --dataset 1 &
python scripts/train_per_dataset.py --dataset 2 &
# ...
# 2. Ensemble predictions
python scripts/ensemble.py --method weighted_avg
```

**預期時間**: 8-12 小時（可並行）
**風險**: 中 - 需要調整 ensemble 權重

---

## 🚀 中期優化 (3-5天)

### 4. Clonality Metrics (預期提升: +0.02-0.03)
- Shannon entropy
- Gini coefficient
- D50 (diversity index)
- Simpson index

### 5. Hyperparameter Optimization (預期提升: +0.01-0.03)
- Grid search on XGBoost params
- Per-dataset hyperparameter tuning
- Early stopping with validation

### 6. Feature Selection & Engineering (預期提升: +0.02-0.04)
- Public clonotypes (shared across individuals)
- CDR3 length distribution features
- Amino acid composition patterns
- Remove low-importance features

---

## 🎲 高風險高回報 (5-7天)

### 7. Protein Language Model Embeddings (預期提升: +0.05-0.10)
**方法**: ESM-2 或 ProtBERT embeddings
**挑戰**:
- 計算成本高（需要 GPU）
- 可能需要降維
- Aggregation strategy 需要實驗

### 8. Deep Learning Models (預期提升: +0.03-0.08)
- Transformer with attention pooling
- BiLSTM for sequence encoding
- Graph Neural Networks

---

## 📊 實驗管理策略

### 每日提交計劃 (5 次/天)
**Day 1 (今天)**:
1. Multi-scale k-mers (k=3,4,5) - baseline
2. Multi-scale + V/J usage
3. (保留) 等待結果

**Day 2**:
1. Per-dataset models
2. Ensemble (weighted avg)
3. Ensemble (stacking)
4. Best from Day 1 + clonality
5. (保留) 實驗性嘗試

**Day 3-8**:
- 持續迭代最佳配置
- 嘗試深度學習方法
- Fine-tuning hyperparameters

**Day 9 (截止日)**:
- 提交最佳模型
- 保留 2 次做最後調整

---

## 🗂️ 目錄重構計劃

### Phase 1: 核心模組化 (今天完成)
```bash
# 創建新目錄結構
mkdir -p experiments/{configs,runs,results}
mkdir -p src/airr_ml25/{features,models,training,evaluation}

# 整合 k-mer 特徵提取
# 創建統一訓練腳本
```

### Phase 2: 實驗追蹤系統 (明天)
```bash
# 實現 YAML 配置系統
# 創建實驗記錄工具
# 遷移現有結果到新結構
```

---

## 🎯 優先級排序

**立即執行** (今天):
1. ✅ Multi-scale k-mers (k=3,4,5)
2. ✅ V/J gene usage features

**明天執行**:
3. Per-dataset models + ensemble
4. Clonality metrics

**本週執行**:
5. Hyperparameter optimization
6. Feature selection

**視情況執行**:
7. Protein LM embeddings (如果前面方法不夠)
8. Deep learning (最後手段)

---

## 📈 成功指標

| 階段 | 目標分數 | 實際分數 | 狀態 |
|------|---------|---------|------|
| Baseline | 0.66987 | 0.66987 | ✅ 完成 |
| Quick Win 1 | 0.70+ | - | ⏳ 進行中 |
| Quick Win 2 | 0.72+ | - | 📝 計劃中 |
| Mid-term | 0.75+ | - | 📝 計劃中 |
| Target | 0.82+ | - | 🎯 目標 |

---

## 🚨 風險管理

1. **Time Risk**: 訓練時間過長
   - 解決: 使用 GPU，並行訓練，縮小搜索空間

2. **Overfitting Risk**: Public leaderboard 過度優化
   - 解決: 使用 leave-one-dataset-out CV
   - 保留部分數據做 local validation

3. **Resource Risk**: GPU OOM
   - 解決: Batch processing，sparse features，k=3 fallback

---

**下一步**: 等待 3 個 agents 完成分析，整合建議後立即開始實施
