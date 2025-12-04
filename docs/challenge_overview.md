# AIRR‑ML‑25: Adaptive Immune Profiling Challenge 2025 — 實作向總覽

> 官方比賽頁面：Kaggle「Adaptive Immune Profiling Challenge 2025」。比賽由 University of Oslo / GreiffLab 與 Kaggle 合作舉辦，聚焦於從 **adaptive immune receptor repertoire (AIRR)** 預測 immune state。citeturn9search3turn9search20

## 1. 問題設定（高層）

- 輸入：成千上萬條 T / B cell 受體序列（CDR3 等），以 **repertoire** 為單位分組。
- 任務 A：根據 repertoire 特徵，預測其 **immune state label**（例如感染 vs 控制）。
- 任務 B：從所有可見序列中選出一個排序清單，代表「與標籤最相關」的受體序列。

## 2. 資料切分與來源（摘要）

- 多個獨立 **train datasets / studies**，每個有自己的 metadata 與序列檔。citeturn10search2
- Test 部分可能包含與 train 來源不同的 cohort，考驗跨資料集泛化。
- 每個 repertoire 由一個或多個 sample 組成，sample 之間可能來自不同時間點 / 個體。

> 建議：一開始就刻意把「dataset/study ID」當作重要欄位，作為潛在 confounder 來看待。

## 3. 任務與評分

### 3.1 任務 A：Immune state prediction

- 對每個 repertoire 預測 **陽性標籤機率** 。
- 評分：多個資料集的 ROC AUC 加權平均。

### 3.2 任務 B：Important sequences

- 對於所有序列，輸出一個排序清單（固定長度，官方建議 50k）。citeturn9search10
- 每筆紀錄包含：
  - 序列 ID 或序列本身（依官方 schema）。
  - 對「與標籤相關程度」的分數或排序。
- 評分：根據與隱藏真值序列集合的 **Jaccard similarity** 等指標計算。

> Task B 本質比較像 feature discovery / motif mining，比純 ML classification 更偏「可解釋性」。

## 4. 提交檔案與程式碼要求

- 官方提供 `sample_submissions.csv` 作為 schema。citeturn10search0turn10search1
- 建議做法：
  1. 讀入 `sample_submissions.csv`。
  2. 以 `repertoire_id` 等共享欄位 merge 你的預測。
  3. 對於重要序列部分，填入你選出的序列及排序。
- 主辦單位要求得獎隊伍提供：
  - 完整可重現的程式碼（推薦使用官方 `ImmuneStatePredictor` 模板）。citeturn9search3
  - 模型與特徵的描述，尤其是如何產生重要序列。

## 5. 典型 baseline 思路

綜合官方與社群目前釋出的 baseline notebook，可以觀察到幾個共同方向：citeturn10search2turn10search3

1. 對序列做 **頻率 / k‑mer 表徵**，再在 repertoire 層級聚合。
2. 使用 PCA 或類似方法降維。
3. 以 XGBoost 或其他樹模型作為強 baseline。
4. 有些隊伍會將 **dataset ID 當作 feature**，或採用分 dataset 訓練再 ensemble。

本專案的 baseline 則選擇 **L1 logistic regression**，作為一個最簡潔、最容易解釋與擴充的起點。

## 6. 為什麼這份文件存在？

這份 `challenge_overview.md` 的目標是：

- 讓 Claude / 你 快速回想起比賽長相，而不用每次都重讀 Kaggle 頁面。
- 把「會影響程式架構與 feature 設計」的重點濃縮起來。
- 提醒自己：這不只是一場 leaderboard 競賽，也是一個理解 adaptive immunity 的實驗場。

真正的細節（欄位定義、禁止事項、時間表、分數計算細節）還是要以 Kaggle 官方頁面為準。
