# Data format notes（根據目前公開資訊的 working 假設）

> **重要聲明**：實際欄位名稱與檔案架構請以 Kaggle 官方 data explorer 為準，這裡只是一份方便 Claude  reasoning 的「共識草稿」。citeturn10search2turn10search4

## 1. 目錄結構（典型）

在 Kaggle 環境中，官方壓縮檔展開後通常長這樣：

```text
/kaggle/input/adaptive-immune-profiling-challenge-2025/
  ├── train_datasets/
  │   ├── STUDY_1/
  │   │   ├── metadata.csv
  │   │   ├── sequences.csv  或  airr.tsv
  │   │   └── ...
  │   ├── STUDY_2/
  │   └── ...
  ├── test_datasets/
  │   ├── STUDY_A/
  │   └── ...
  ├── sample_submissions.csv
  └── README + LICENSE 等
```

本專案的 `config.py` 只假設有 `train_datasets/` 與 `test_datasets/` 兩層，不綁定特定 study 名稱。

## 2. `metadata.csv`（train side）

每個 train dataset 一般都會有一個 `metadata.csv`，大致會包含：citeturn9search10

- `repertoire_id`：repertoire 的唯一 ID。
- `label` 或類似欄位：immune state（0/1 或 multi‑class）。
- 可能還有：
  - `patient_id`、`sample_id`、`timepoint` 等個案或時間相關欄位。
  - `dataset_id` 或 study 名稱。

> 實作建議：
> - 初期 baseline 只用 `repertoire_id` + `label`。
> - 之後再納入 `dataset_id`、`patient_id` 做 multi‑level 或 leave‑one‑group‑out 驗證。

## 3. 序列檔（`sequences.csv` / `airr.tsv`）

序列檔遵循 **AIRR Community data standard** 的精神，常見欄位包含：citeturn9search13

- `repertoire_id`
- `sequence_id`
- `junction_aa` 或 `cdr3_aa`：胺基酸序列。
- `v_call`、`j_call`：V / J gene annotation。
- `duplicate_count` 或 `count`：clonotype 出現次數。
- 其他品質與 alignment 欄位。

本專案中：

- `data.py` 預設從 `{dataset}/sequences.csv` 讀取（CSV）；若實際為 TSV，請改成 `sequences.tsv` 或 `airr.tsv` 並調整參數。
- `features.py` 預設將 `cdr3_aa` 當作序列欄位，必要時可以改成 `junction_aa`。

## 4. `sample_submissions.csv`（雙任務 schema）

`sample_submissions.csv` 同時包含任務 A 與任務 B 的欄位，常見設計為：citeturn10search0turn10search5

- 用某種 `row_type` 或 `task` 欄位區分 A/B。
- 任務 A 需要：
  - `repertoire_id`
  - `probability` 或類似欄位（浮點數）。
- 任務 B 需要：
  - `sequence_id` 或序列本身
  - importance / ranking 欄位。

具體名稱請在 Kaggle notebook 內用：

```python
import pandas as pd
sample = pd.read_csv("/kaggle/input/adaptive-immune-profiling-challenge-2025/sample_submissions.csv")
sample.head()
sample.columns
```

實際檢查後，再回頭更新這份文件與 `submission.py` 裡的 mapping 邏輯。

## 5. 對 Claude 的提醒

當你幫忙修改程式時，請：

1. **不要亂假設欄位名稱**：先建議使用者在 Kaggle 裡 `print(sample.columns)` 確認。
2. 將「schema 具體長相」寫回這一份 `data_format.md`，並同步修改：
   - `airr_ml25.data.load_sequences`
   - `airr_ml25.features.aggregate_repertoire_features`
   - `airr_ml25.models.*`
   - `airr_ml25.submission` 的輸出欄位。

這樣未來在 refactor 或加新模型時，就不用每次重新 reverse‑engineer 数据格式。
