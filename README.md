# AIRR-ML-25 競賽專案

> **目標**：贏得 AIRR-ML-25: Adaptive Immune Profiling Challenge 2025
> **截止日期**：2025-12-17 (UTC 06:59)
> **獎金**：第一名 US$5,000 + Nature Methods 共同作者

---

## TL;DR

```bash
# 1. 安裝依賴 (GPU 版本)
pip install -r requirements.txt

# 2. 下載資料
kaggle competitions download -c adaptive-immune-profiling-challenge-2025 -p ./data/
cd data && unzip adaptive-immune-profiling-challenge-2025.zip && cd ..

# 3. 訓練模型
python scripts/train_gpu.py

# 4. 提交結果
kaggle competitions submit -c adaptive-immune-profiling-challenge-2025 \
    -f outputs/submissions/submission.csv -m "GPU XGBoost k=3"
```

---

## 比賽任務

| 任務 | 說明 | 評估指標 |
|------|------|----------|
| **Task A** | 預測免疫狀態 (疾病/健康) | ROC-AUC |
| **Task B** | 辨識前 50,000 個疾病相關序列 | Jaccard 相似度 |

**提交格式**：404,213 列 (4,213 預測 + 8×50,000 序列)

---

## 專案結構

```
airr-ml25-package/
├── scripts/              # 訓練腳本
│   ├── train_gpu.py      # GPU XGBoost 訓練
│   ├── train_fast.py     # 快速訓練版本
│   └── validate.py       # 驗證提交格式
├── src/airr_ml25/        # 核心模組
│   ├── config.py         # 路徑設定
│   ├── data.py           # 資料讀取
│   ├── features.py       # 特徵工程 (k-mer, V/J usage)
│   └── submission.py     # 提交檔生成
├── outputs/              # 輸出目錄 (gitignored)
│   └── submissions/      # 提交檔案
├── data/                 # 資料集 (gitignored, ~20GB)
├── docs/                 # 文件
└── tests/                # 測試
```

---

## 硬體需求

| 項目 | 建議規格 |
|------|----------|
| GPU | NVIDIA RTX 3080+ (16GB VRAM) |
| RAM | 32GB+ |
| 儲存 | 50GB+ SSD |

---

## 當前進度

| 版本 | 分數 | 狀態 |
|------|------|------|
| Enhanced v1 | 0.65176 | 最高分 |
| GPU Champion | 0.63017 | - |
| Simple baseline | 0.62744 | - |
| Official baseline | 0.60601 | - |

**目標**：0.82+ (領先者 GROZD: 0.83623)

---

## 快速指令

```bash
# 檢查提交格式
python scripts/validate.py outputs/submissions/submission.csv

# 查看提交紀錄
kaggle competitions submissions -c adaptive-immune-profiling-challenge-2025

# 查看排行榜
kaggle competitions leaderboard -c adaptive-immune-profiling-challenge-2025 --show
```

---

## 授權

MIT License - 依比賽規則，得獎者需公開原始碼。
