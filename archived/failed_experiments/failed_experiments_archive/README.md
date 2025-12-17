# Failed Experiments Archive - AIRR-ML-25

**Created**: 2025-12-17
**Purpose**: 記錄所有失敗的方法，避免重蹈覆轍

---

## 總結：什麼有效，什麼無效

| 方法 | 結果 | 教訓 |
|------|------|------|
| XGBoost + LightGBM | **~0.95 CV AUC** | 樹模型對表格資料有效 |
| ESM-2 + MIL | 0.5322 | 蛋白質語言模型無效 |
| V5 Features + MLP | 0.5199 | 神經網路過擬合 |
| CNN + Attention | ~0.52 | 深度學習無效 |
| 所有 DL 方法 | ~0.50-0.53 | 不適合此任務 |

---

## 失敗實驗詳情

### 1. champion_v6_neural.py - V5 Features + MLP
- **日期**: 2025-12-17
- **方法**: 使用 V5 的特徵 + DeepMLP 神經網路
- **結果**:
  - Train AUC = 1.0000 (完全過擬合)
  - Val AUC = 0.5199 (比隨機還差)
- **日誌**: `logs/v6_neural.log`
- **教訓**: MLP 無法泛化到免疫序列資料

### 2. champion_maxed_out.py - ESM-2 + Gated Attention MIL
- **日期**: 2025-12-17
- **方法**: ESM-2 蛋白質語言模型 embeddings + MIL
- **結果**: Mean LODO AUC = 0.5322
- **日誌**: `logs/maxed_20251217_041155.log`, `logs/maxed_restart.log`
- **教訓**: ESM-2 對免疫序列不適用

### 3. champion_breakthrough.py - Transformer 方法
- **日期**: 2025-12-17
- **方法**: Transformer-based attention mechanism
- **結果**: 失敗
- **日誌**: `logs/breakthrough_20251217_034902.log`
- **教訓**: Transformer 架構對此任務無效

### 4. champion_breakthrough_dual_gpu.py - 雙 GPU Transformer
- **日期**: 2025-12-17
- **方法**: 雙 GPU 加速 Transformer
- **結果**: OOM 或低 AUC
- **日誌**: `logs/breakthrough_dual_20251217_035259.log`
- **教訓**: 增加計算資源無法解決模型問題

### 5. champion_cnn_attention_gpu1.py - CNN + Attention
- **日期**: 2025-12-17
- **方法**: CNN + Attention pooling
- **結果**: ~0.52 AUC
- **日誌**: `logs/cnn_attention_20251217_040100.log`
- **教訓**: CNN 無法捕捉序列模式

### 6. championship_dl.py - 基礎深度學習
- **日期**: 2025-12-16
- **方法**: 標準深度學習 pipeline
- **結果**: 低 AUC
- **教訓**: 深度學習基線也失敗

### 7. championship_dl_mini.py - 簡化深度學習
- **日期**: 2025-12-16
- **方法**: 簡化版深度學習
- **結果**: 低 AUC
- **教訓**: 簡化版同樣無效

### 8. ultimate_parallel_system.py - 並行訓練系統
- **日期**: 2025-12-16
- **方法**: 多進程並行訓練
- **結果**: 效率問題
- **教訓**: 複雜系統不保證好結果

### 9. ultimate_parallel_system_v2_cached.py - 緩存版本
- **日期**: 2025-12-16
- **方法**: 加入緩存的並行系統
- **結果**: 仍然低效
- **教訓**: 緩存無法解決根本問題

---

## 關鍵教訓

### 深度學習為何失敗：
1. **免疫序列資料特性**
   - 序列之間的模式不像自然語言
   - CDR3 序列多樣性太高
   - 疾病信號是統計性的，不是序列性的

2. **資料量不足**
   - 每個資料集只有 ~400 個 repertoire
   - 深度學習需要更多資料

3. **過擬合**
   - 神經網路容易記憶訓練資料
   - 跨資料集泛化是關鍵

### 什麼有效：
1. **XGBoost + LightGBM**
   - 樹模型對表格特徵有效
   - 內建正則化防止過擬合

2. **Public Clone Mining**
   - 找出疾病特異的序列
   - 統計方法更穩健

3. **K-mer + 物理化學特徵**
   - 捕捉序列統計特性
   - 不依賴序列順序

---

## 檔案清單

```
failed_experiments_archive/
├── README.md                          # 本文件
├── champion_v6_neural.py              # MLP 神經網路 (失敗)
├── champion_maxed_out.py              # ESM-2 + MIL (失敗)
├── champion_breakthrough.py           # Transformer (失敗)
├── champion_breakthrough_dual_gpu.py  # 雙 GPU Transformer (失敗)
├── champion_cnn_attention_gpu1.py     # CNN + Attention (失敗)
├── championship_dl.py                 # 深度學習基線 (失敗)
├── championship_dl_mini.py            # 簡化深度學習 (失敗)
├── ultimate_parallel_system.py        # 並行系統 v1 (效率問題)
├── ultimate_parallel_system_v2_cached.py  # 並行系統 v2 (效率問題)
├── v6_neural.log                      # V6 神經網路日誌
└── logs/                              # 所有相關日誌
    ├── breakthrough_*.log
    ├── cnn_attention_*.log
    ├── maxed_*.log
    └── v6_*.log
```

---

## 結論

**永遠不要再嘗試深度學習方法！**

使用 XGBoost + LightGBM + Public Clone Mining 是唯一有效的方法。
