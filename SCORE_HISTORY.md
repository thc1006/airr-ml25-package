# 🏆 AIRR-ML-25 分數記錄

這個檔案追蹤所有提交的分數，以便分析每次改進的成效。

---

## 📊 提交歷史

### Baseline (原始 main.py, k=4)
- **日期**: 2025-12-08 (早期)
- **分數**: **0.66987**
- **方法**:
  - 單一 k-mer (k=4)
  - LogisticRegression (L1 正則化)
  - 僅使用 k-mer 特徵
- **Task A**: 預測問題 (20.7% 低信心預測)
- **Task B**: 使用正確的 LogReg 係數方法 ✓
- **備註**: Task B 實作正確，分數低的真正原因是特徵不足

---

### Enhanced v2 (Priority 1 特徵)
- **日期**: 2025-12-08 (09:40-09:57)
- **分數**: **待測試** (預期 0.75-0.80)
- **改進內容**:

  #### ✅ Priority 1 特徵（全部實施）
  1. **Multi-scale k-mers** (k=3,4,5)
     - 從單一 k=4 擴展到三種尺度
     - 提供不同粒度的序列模式

  2. **V/J gene usage patterns** (90+ 特徵)
     - Top 20 V genes
     - Top 20 J genes
     - Top 50 VJ pairs

  3. **Clonality metrics** (4 特徵)
     - Shannon entropy
     - Gini-Simpson index
     - D50 (diversity index)
     - Clonality score

  4. **CDR3 length statistics** (7 特徵)
     - Mean, std, median, quartiles
     - Skewness, kurtosis

- **技術細節**:
  - 特徵總數: ~50,000+ (vs 原始 ~160,000 k=4 only)
  - 處理速度: ~5.3 repertoires/second
  - 管線總時間: ~17 分鐘 (8 datasets)
  - 模型: LogisticRegression (L1, C=0.01, max_iter=1000)

- **檔案位置**: `./results_v2_20251208_094002/submissions.csv`
- **預期提升**: +0.08 ~ +0.13 (from 0.67 → 0.75-0.80)

---

## 📈 改進分析

### 從 Baseline → Enhanced v2

**主要改進來源**:
1. **Multi-scale k-mers**: 捕捉不同長度的序列模式 → +0.02-0.03
2. **V/J gene usage**: 加入免疫受體遺傳資訊 → +0.03-0.05 ⭐ 高影響
3. **Clonality metrics**: 量化 repertoire 多樣性 → +0.02-0.03
4. **CDR3 length features**: 序列長度分佈統計 → +0.01-0.02

**預期總提升**: +0.08 ~ +0.13

---

## 🎯 目標追蹤

| 里程碑 | 目標分數 | 當前狀態 | 備註 |
|--------|---------|---------|------|
| Baseline | 0.67 | ✅ 已達成 | 原始實作 |
| Priority 1 特徵 | 0.75-0.80 | 🔄 待驗證 | Enhanced v2 |
| Top 10 | 0.78 | ⏳ 進行中 | 需驗證分數 |
| Top 3 | 0.82 | ⏳ 下一步 | 可能需要 Priority 2 |
| 第 1 名 | 0.81364+ | ⏳ 終極目標 | 當前冠軍: GROZD (0.81364) |

---

## 📝 待測試改進 (Priority 2+)

### 🔧 Priority 2: 模型優化 (預期 +0.05-0.08)
- [ ] Ensemble Methods (XGBoost + LightGBM + CatBoost)
- [ ] Per-dataset models with ensemble
- [ ] Hyperparameter tuning (GridSearch/Optuna)

### 🎲 Priority 3: 高級技術 (預期 +0.04-0.10)
- [ ] ESM-2 Protein Language Model embeddings
- [ ] DeepRC Attention mechanisms
- [ ] Graph-based sequence similarity
- [ ] Public clonotypes database

---

## 🔬 實驗記錄

### Experiment 1: Enhanced Features (Priority 1)
- **開始時間**: 2025-12-08 09:40
- **完成時間**: 2025-12-08 09:57 (17 分鐘)
- **測試狀態**: ✅ 所有測試通過
- **管線狀態**: ✅ 成功完成 (8/8 datasets)
- **提交狀態**: ⏳ 待上傳 Kaggle

---

## 🚀 Priority 3: 深度學習冠軍衝刺

### Championship Deep Learning Pipeline (ESM-2 + Attention) - v2 (修復版)

#### 🔧 訓練歷史

**第一次嘗試** (2025-12-08 11:27 UTC)
- ❌ **失敗**: 在 Dataset 8 時崩潰
- **錯誤 1**: `TypeError: ReduceLROnPlateau.__init__() got an unexpected keyword argument 'verbose'`
- **錯誤 2**: `ValueError: Input contains NaN`

**修復措施** (2025-12-09 15:50 UTC)
- ✅ 移除 `ReduceLROnPlateau` 的 `verbose=True` 參數（PyTorch 版本相容性）
- ✅ 實作完整的 NaN 處理機制：
  - `extract_clonality_features()`: 使用 `dropna()` 並檢查 NaN/inf
  - `standardize_features()`: 在賦值前驗證 NaN/inf
  - `extract_features_from_repertoire()`: 將 NaN 替換為 0.0
- ✅ 記憶體優化：
  - ESM-2 batch_size: 32 → 16
  - DataLoader batch_size: 8 → 4
  - num_workers: 4 → 2
  - 混合精度訓練 (FP16/FP32)
  - 定期 GPU cache 清理

**第二次嘗試** (2025-12-09 15:50 UTC) ❌ **已停止 (16:05 UTC)**
- 運行 15 分鐘後停止以實作多核心優化
- Dataset 1 進度: 94% (375/400 repertoires)

**第三次嘗試 - 多核心優化版** (2025-12-09 16:09 UTC) ❌ **Dataset 8 崩潰**
- 在 Dataset 8 ESM-2 處理階段崩潰（約 48% 進度）

**第四次嘗試 - 自動監控版** (2025-12-09 22:54 UTC) ✅ **運行中**
- **開始時間**: 2025-12-09 22:54 UTC
- **狀態**: 🔄 **訓練中** - Phase 1: Dataset 1 ESM-2 編碼
- **監控系統**: `auto_watchdog.py` 運行中
  - 自動檢測崩潰並重啟（最多 5 次）
  - 每 60 秒監控一次
  - 日誌: `logs/watchdog.log`
- **訓練日誌**: `logs/auto_train_20251209_225446.log`
- **GPU 狀態**: 91% 使用率, 2986 MB, 60°C
- **預計完成**: 18-26 小時（8-fold CV）
- **目標分數**: **0.82+** → 擊敗 GROZD (0.81364) 奪冠

#### 架構設計
- **ESM-2**: 650M 參數蛋白質語言模型（Meta AI）
- **Attention**: Multi-head attention (4 heads) 聚合可變長度 repertoires
- **Hybrid Features**:
  - 深度學習 embeddings (1280-dim)
  - 傳統特徵 (~389-dim): V/J usage, VJ pairs, clonality, CDR3 length
- **訓練策略**: Leave-one-dataset-out CV（8 folds）

#### 訓練性能（即時）
- 🚀 **GPU 使用率**: 90% (RTX 5080 16GB)
- 💾 **GPU 記憶體**: 2.9 GB / 15.9 GB (18%)
- 🌡️ **GPU 溫度**: 59°C ← 更低更穩定
- ⚡ **處理速度**:
  - Traditional Features: **56 repertoires/秒** (105x 加速) 🔥
  - ESM-2 Embeddings: ~1.98 秒/repertoire
- 📝 **日誌檔案**: `logs/auto_train_20251209_160908.log`
- 🆔 **PID**: 3405016

#### 監控指令
```bash
# 查看訓練進度
tail -f ./logs/auto_train_20251209_225446.log

# 查看監控系統狀態
tail -f ./logs/watchdog.log

# 查看 GPU 使用狀況
watch -n 1 nvidia-smi

# 檢查進程狀態
pgrep -af "championship|watchdog"
```

#### 預期結果
- **Cross-Validation AUC**: 0.75-0.82 (目標 > 0.80)
- **Public Leaderboard**: 預期達到 Top 3（> 0.78）
- **Private Leaderboard**: 衝擊第 1 名（> 0.81364）

---

**最後更新**: 2025-12-09 22:56 UTC
**當前狀態**: Priority 3 深度學習訓練進行中（v4 自動監控版）
**自動化系統**: `auto_watchdog.py` 運行中，自動處理崩潰並重啟
**優化亮點**: Traditional features 處理加速 **105倍**（2.1s → 0.018s per repertoire）
**下一步行動**: 等待訓練完成（18-26 小時）→ 提交 Kaggle → 記錄分數
