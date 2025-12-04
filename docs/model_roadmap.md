# Model roadmap & experiment backlog（AIRR‑ML‑25）

這份文件是給「未來的你」與 Claude 看的 experiment log / backlog。  
建議結構：

- 第一段：目前 baseline 與已完成實驗。
- 第二段：短期（1–2 週）計畫。
- 第三段：中期（1–2 個月）計畫與 crazy ideas。

---

## 1. 目前 baseline（v0）

- **Feature**：repertoire 層級 aggregate
  - seq_count, total_count
  - CDR3 長度 mean / std / max
  - global top‑64 k‑mer counts
- **Model**：L1 logistic regression（per‑dataset 訓練）。
- **輸出**：repertoire‑level probability table（尚未完全對齊 sample_submissions schema）。

TODO（交給你和 Claude）：

- [ ] 在 Kaggle notebook 內實際跑通 `airr_ml25.submission`，確認能產生正確大小的概率表。
- [ ] 計算 simple ROC AUC（使用 metadata label）作 sanity check。

---

## 2. 短期計畫（1–2 週）

1. **資料 schema 確認與清洗**
   - [ ] 把實際的 `sample_submissions.csv` 欄位寫回 `docs/data_format.md`。
   - [ ] 寫一個 `validate_submission.py`（或 notebook cell），確保輸出：
     - 欄位齊全、型別正確，沒有 NaN。
     - 行數與官方 sample 一致。

2. **多資料集 baseline**
   - [ ] 增加「dataset ID one‑hot」feature，在單一模型內同時訓練所有 dataset。
   - [ ] 設計 k‑fold 與 leave‑one‑dataset‑out cross‑validation。

3. **Task B 首版 pipeline**
   - [ ] 根據 Task A 的模型，計算每個 repertoire 的 feature importance。
   - [ ] 經由 k‑mer / motif 回推「重要序列」候選清單，產生一份最簡單的序列排序結果。
   - [ ] 用 dummy score（例如 per dataset 自評分數）當作 proxy，測試整體流程是否穩定。

---

## 3. 中期計畫（1–2 個月）

1. **更豐富的 repertoire 表徵**
   - [ ] 加入多種 k 值的 k‑mer（k=3,4,5）。
   - [ ] V/J usage、VJ pair one‑hot、public clonotypes（出現在多個個體的序列）。
   - [ ] Clonality & diversity 指標（Shannon entropy, Gini, D50 等）。citeturn9search12turn9search14

2. **模型升級**
   - [ ] 試驗樹模型（XGBoost / LightGBM）。
   - [ ] 嘗試 simple NN / transformer on sequence embeddings（在本地或具 GPU 的機器上）。

3. **跨資料集穩健性與因果觀點**
   - [ ] 分析各 dataset label 分佈與技術差異（platform, sequencing depth, 年齡等）。
   - [ ] 探索 CausalAIRR 或類似資料集與方法，避免學到 confounders。citeturn9search5

---

## 4. Crazy ideas（視時間與 GPU 情況）

- 使用 pre‑trained protein LM（ESM, ProtBERT 等）對 CDR3 做 embedding，repertoire 層級再做 pooling。
- 建立 sequence‑to‑sequence graph，利用圖神經網路或 community detection 找 motifs。
- 跟免疫學 /醫學背景的夥伴一起看重要序列清單，挑出能連到病理機制的範例，寫成 short note。

---

## 5. 寫給 Claude 的小叮嚀

當使用者請你「幫忙設計下一個實驗」時：

1. 先閱讀這份 `model_roadmap.md`，避免重複做一樣的事情。
2. 主動：
   - 提醒目前哪些實驗缺少完整紀錄。
   - 建議如何把 notebook / script 整理成 reusable module。
3. 完成一個實驗後，幫忙在這份文件加上：
   - 簡短結論（1–3 行）。
   - 最重要的圖 / 表格描述。
   - 對下一步的具體建議。
