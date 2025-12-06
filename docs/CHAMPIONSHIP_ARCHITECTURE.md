# AIRR-ML-25 Championship Architecture

> **目標**: 奪得第一名 (分數 > 0.82)
> **截止日期**: 2025年12月17日 (06:59 UTC)
> **當前領先者**: 0.81364 (GROZD team)

---

## 1. 研究發現摘要

### 1.1 關鍵技術洞察

| 技術 | 來源 | 預期提升 | 優先級 |
|------|------|----------|--------|
| **DeepRC + Modern Hopfield Networks** | NeurIPS 2020 | 高 | P0 |
| **ESM2/TCR-BERT 嵌入** | Nature 2023 | 高 | P0 |
| **Attention-based MIL** | 多篇論文 | 高 | P0 |
| **XGBoost+LightGBM+CatBoost 堆疊** | Kaggle 冠軍 | 中-高 | P1 |
| **Public Clonotypes / Meta-clonotypes** | eLife 2021 | 中 | P1 |
| **V/J Gene Usage Patterns** | 多篇論文 | 中 | P2 |

### 1.2 競賽關鍵洞察

1. **50% 分數來自真實世界數據** - 不能只針對合成數據優化
2. **需要泛化能力** - 模型必須學習可遷移的模式
3. **Task B 同樣重要** - 識別疾病相關序列是關鍵差異化因素

---

## 2. 冠軍架構設計

### 2.1 多層次特徵提取

```
輸入: 免疫庫 (數萬條 TCR 序列)
          │
          ▼
┌─────────────────────────────────────────────────────────┐
│                    Level 1: 序列級特徵                    │
├─────────────────────────────────────────────────────────┤
│  • K-mer 頻率 (k=3,4,5)                                  │
│  • CDR3 長度分佈                                          │
│  • 氨基酸組成                                             │
│  • 物理化學性質 (疏水性、電荷等)                           │
└─────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────┐
│                Level 2: 嵌入式特徵 (新增)                  │
├─────────────────────────────────────────────────────────┤
│  • ESM2 蛋白質語言模型嵌入 (650M 參數)                    │
│  • TCR-BERT 專用嵌入                                      │
│  • 自定義 CNN 序列編碼                                    │
└─────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────┐
│                Level 3: 免疫庫級特徵                      │
├─────────────────────────────────────────────────────────┤
│  • V/J 基因使用頻率                                       │
│  • VJ 配對組合                                           │
│  • 克隆多樣性指標 (Shannon, Gini, D50)                    │
│  • Public Clonotype 特徵                                 │
└─────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────┐
│              Level 4: Attention-based 聚合               │
├─────────────────────────────────────────────────────────┤
│  • DeepRC-style Modern Hopfield 注意力                   │
│  • 多頭注意力機制                                         │
│  • 加權序列聚合                                           │
└─────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────┐
│                Level 5: 集成分類器                        │
├─────────────────────────────────────────────────────────┤
│  Base Learners:                                          │
│    • XGBoost (GPU: device='cuda')                        │
│    • LightGBM (GPU: device='gpu')                        │
│    • CatBoost (GPU: task_type='GPU')                     │
│  Meta-Learner:                                           │
│    • Ridge Regression / Logistic Regression              │
└─────────────────────────────────────────────────────────┘
          │
          ▼
輸出: 疾病概率 + 重要序列排名
```

### 2.2 模組設計

```
src/
├── airr_ml25/                   # 核心套件
│   ├── __init__.py
│   ├── config.py                # 配置管理
│   ├── data.py                  # 數據載入
│   └── models/
│       ├── __init__.py
│       └── baseline_logreg.py   # 基線模型
│
├── features/                    # 特徵工程模組 (新增)
│   ├── __init__.py
│   ├── kmer_features.py         # K-mer 特徵
│   ├── gene_usage.py            # V/J 基因使用
│   ├── diversity_metrics.py     # 多樣性指標
│   ├── public_clonotypes.py     # 公共克隆型 (已存在)
│   └── physicochemical.py       # 物理化學性質
│
├── embeddings/                  # 嵌入模組 (新增)
│   ├── __init__.py
│   ├── esm2_encoder.py          # ESM2 蛋白質語言模型
│   ├── tcr_bert.py              # TCR-BERT
│   └── cnn_encoder.py           # CNN 序列編碼器
│
├── attention/                   # 注意力模組 (重構)
│   ├── __init__.py
│   ├── hopfield.py              # Modern Hopfield Networks
│   ├── attention_pool.py        # 注意力池化
│   └── deeprc.py                # DeepRC 完整實現
│
├── ensemble/                    # 集成模組 (新增)
│   ├── __init__.py
│   ├── stacking.py              # 堆疊集成
│   ├── blending.py              # 混合集成
│   └── hill_climbing.py         # 爬山優化權重
│
└── pipeline/                    # 管線模組 (新增)
    ├── __init__.py
    ├── champion_pipeline.py     # 冠軍管線
    └── submission.py            # 提交生成
```

---

## 3. GPU 加速策略

### 3.1 RAPIDS cuML 零代碼更改加速

```bash
# 設置環境變量
export LD_LIBRARY_PATH="/path/to/nvidia/libs:$LD_LIBRARY_PATH"

# 使用 cuML 加速器運行
python -m cuml.accel champion_pipeline.py
```

### 3.2 模型 GPU 配置

```python
# XGBoost GPU
xgb_params = {
    'device': 'cuda',
    'tree_method': 'hist',
}

# LightGBM GPU
lgb_params = {
    'device': 'gpu',
    'gpu_platform_id': 0,
    'gpu_device_id': 0,
}

# CatBoost GPU
catboost_params = {
    'task_type': 'GPU',
    'devices': '0',
}
```

### 3.3 PyTorch 模型 (ESM2, Attention)

```python
# 自動選擇設備
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 混合精度訓練
from torch.amp import autocast, GradScaler
scaler = GradScaler()
with autocast(device_type='cuda'):
    outputs = model(inputs)
```

---

## 4. 實施路線圖

### Phase 1: 基礎建設 (Day 1-2)
- [x] 清理專案結構
- [x] 完成策略研究
- [ ] 重構 src/ 模組結構
- [ ] 建立單元測試框架

### Phase 2: 特徵工程 (Day 3-4)
- [ ] 實現多尺度 k-mer 特徵
- [ ] 實現 V/J 基因使用特徵
- [ ] 實現多樣性指標
- [ ] 整合 public clonotype 特徵

### Phase 3: 深度學習 (Day 5-7)
- [ ] 整合 ESM2 嵌入
- [ ] 實現 DeepRC-style 注意力
- [ ] 訓練序列編碼器

### Phase 4: 集成優化 (Day 8-9)
- [ ] 實現 XGB+LGB+CatBoost 堆疊
- [ ] 爬山優化權重
- [ ] 交叉驗證調優

### Phase 5: 最終提交 (Day 10-11)
- [ ] 完整管線測試
- [ ] 生成提交文件
- [ ] 驗證格式合規
- [ ] 最終提交

---

## 5. 預期分數提升

| 組件 | 當前 | 預期提升 | 新分數 |
|------|------|----------|--------|
| 基線 (k-mer + LogReg) | 0.65 | - | 0.65 |
| + V/J 基因使用 | - | +0.02 | 0.67 |
| + 多樣性指標 | - | +0.02 | 0.69 |
| + Public Clonotypes | - | +0.03 | 0.72 |
| + ESM2 嵌入 | - | +0.04 | 0.76 |
| + DeepRC 注意力 | - | +0.03 | 0.79 |
| + 堆疊集成 | - | +0.03 | 0.82 |
| + 超參數優化 | - | +0.01 | **0.83** |

---

## 6. 風險與緩解

| 風險 | 可能性 | 影響 | 緩解策略 |
|------|--------|------|----------|
| ESM2 太慢 | 中 | 高 | 使用較小模型 (8M) 或預計算嵌入 |
| 過擬合公開榜 | 高 | 高 | 保留驗證集，交叉驗證 |
| GPU 記憶體不足 | 低 | 中 | 批次處理，梯度累積 |
| 時間不足 | 中 | 高 | 優先實現高價值特徵 |

---

*最後更新: 2025-12-06*
*版本: 1.0.0*
