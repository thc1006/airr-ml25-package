# AIRR‑ML‑25 Kaggle Starter (Claude‑friendly)

這個專案是一個為 **AIRR‑ML‑25: Adaptive Immune Profiling Challenge 2025** 準備的「可被 Claude Code 馴化」的起手式骨架。

目標是：

1. 讓你可以 **在本地或 Kaggle notebook** 上，一鍵跑通 baseline。
2. 讓 Claude 能夠 **理解專案結構、讀寫程式與文件**，自動幫你加 feature / 換模型 / 生報告。
3. 預留鉤子給 **MCP 伺服器與 Agent Skills**，方便未來做更激進的自動化。

> 比賽官方網站與說明請見 Kaggle 與 UiO 官網。這裡只保留實作上最重要的摘錄。

---

## 1. 比賽簡述

### 任務

比賽要求同時完成兩個任務：

1. **疾病標籤預測（Task A）**：根據每個人 T cell/B cell 受體序列集合（即免疫受體庫 `repertoire`）預測個體是否為標籤陽性，例如患病或健康。
2. **關鍵序列識別（Task B）**：辨識那些對預測影響最大的受體序列（`junction_aa`, `v_call`, `j_call` 組合）並以重要性排序，輸出前 50,000 個最重要的序列。

### 資料

- 官方資料集包含 **8 個訓練資料集**與 **11 個測試資料集**
- 共 7,832 檔案，約 **19.94 GB**，檔案類型為 TSV/CSV
- 每個訓練資料夾內有 `metadata.csv`，描述每個檔案的 `repertoire_id`、檔名與 `label_positive` 欄位
- 可選欄位包括：`age`、`sex`、`race`、`sequencing_run_id`、HLA 基因等

### 檔案格式

每個 TSV 檔含有：
- `junction_aa`（氨基酸序列）
- `v_call`、`j_call`（V/J 基因區段）
- 可選的 `d_call` 和 `duplicate_count`

### 評估方式

- **Task A**：各資料集 **ROC AUC** 的加權平均
- **Task B**：使用 **Jaccard 相似度**衡量你選出來的重要序列與主辦方「真實重要序列」之間的重疊

### 提交格式

- 使用官方提供的 `sample_submissions.csv` 作為 schema
- 提交檔應包含 **4,213 筆測試預測** + 每個訓練資料集中 **50,000 列重要序列**（共 404,213 列）
- 不允許 `NaN`；缺失值要用 **`-999.0`** 填補

### 時程與獎勵

- 比賽：2025‑11‑05 開始，**2025‑12‑17** 最終提交截止
- 獎金：第一名 US$5,000、第二名 US$3,000、第三名 US$2,000
- 得獎者需**公開原始碼**
- 前 10 名有機會參與 **Nature Methods** 的科學論文撰寫

---

## 2. 目錄內容

這個壓縮檔解開後，包含：

| 檔案／資料夾 | 用途 |
|---|---|
| `CLAUDE.md` | 告訴 Claude：這個 repo 是幹嘛用的、有哪些工具可以用、遇到什麼事要優先幫你做什麼 |
| `requirements.txt` | baseline 所需的 Python 套件 |
| `main.py` | 簡易的 baseline 範例，示範 3‑mer k‑mer 特徵 + L1 邏輯迴歸 |
| `claude_mcp_config.json` | 示範如何宣告要啟動的 MCP 伺服器 |
| `install_skills.md` | 說明如何安裝 Claude 插件與 Agent Skills |

### `src/airr_ml25/` 模組

| 檔案 | 用途 |
|---|---|
| `config.py` | 集中處理資料路徑與資料集名稱列表 |
| `data.py` | 讀取 `metadata.csv` 與序列檔（TSV / CSV） |
| `features.py` | 簡單的 k‑mer + V/J 使用率 + repertoire 統計特徵 |
| `models/baseline_logreg.py` | L1 Logistic Regression baseline |
| `submission.py` | 把預測組合成 Kaggle 要求格式的輸出 |

### 其他資源

| 檔案／資料夾 | 用途 |
|---|---|
| `notebooks/00_quick_eda.py` | 以腳本形式寫的 EDA 範例，方便你轉成 notebook |
| `docs/challenge_overview.md` | 比賽規則、評分方式與時程精簡版 |
| `docs/data_format.md` | 訓練 / 測試目錄結構與欄位說明 |
| `docs/model_roadmap.md` | 從 baseline → 進階特徵 → 深度學習 → 因果 / 模擬的 roadmap |
| `docs/mcp_and_skills.md` | 如何把 MCP + Skills 串進這個 repo |
| `skills/airr-ml25-research/SKILL.md` | 客製 Skill，讓 Claude 在面對 AIRR‑ML‑25 任務時有更明確的 operating mode |

---

## 3. 安裝與環境

### 3.1 本地環境（建議）

1. 建立虛擬環境：
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   ```

2. 安裝套件：
   ```bash
   pip install -r requirements.txt
   ```

3. 下載 Kaggle 官方資料集：
   ```bash
   kaggle competitions download -c adaptive-immune-profiling-challenge-2025
   unzip adaptive-immune-profiling-challenge-2025.zip -d ./data
   ```

### 3.2 Kaggle Notebook

在 Kaggle code cell 中：

```python
!pip install -q -r /kaggle/working/requirements.txt
```

並把本專案內容 upload 到 `/kaggle/working` 或透過 Git 拉進來。

---

## 4. 使用步驟

### 訓練與預測

執行 `main.py`，並指派訓練集與測試集路徑：

```bash
python main.py --train_dir /path/to/train_datasets/train_dataset_1 \
               --test_dirs /path/to/test_datasets/test_dataset_1 \
               --out_dir /path/to/results --device cpu --n_jobs 4
```

或使用模組化 entrypoint：

```bash
python -m airr_ml25.submission \
  --train-root /kaggle/input/adaptive-immune-profiling-challenge-2025/train_datasets \
  --test-root  /kaggle/input/adaptive-immune-profiling-challenge-2025/test_datasets \
  --sample-sub /kaggle/input/adaptive-immune-profiling-challenge-2025/sample_submissions.csv \
  --out-path   /kaggle/working/submission.csv
```

程式會輸出：
- `<train_dataset_name>_test_predictions.tsv`
- `<train_dataset_name>_important_sequences.tsv`

可重複對不同資料集執行，最後使用官方 `concatenate_output_files` 函式合併為 `submissions.csv`。

---

## 5. 如何用 Claude Code 來 work

1. 在專案資料夾打開 Claude Code。

2. 請 Claude：
   - 先閱讀 `docs/challenge_overview.md` 和 `docs/data_format.md`
   - 檢查 `src/airr_ml25/` 裡每個模組的邏輯是否正確
   - 用 bash 工具執行 baseline：
     ```bash
     python -m airr_ml25.submission --help
     ```

3. 接著可以要求 Claude：
   - 增加新的 feature（例如 CDR3 長度分佈、k‑mer PCA、VJ 組合 one‑hot 等）
   - 加上 cross‑validation 與 seed 控管
   - 幫你寫實驗紀錄到 `docs/model_roadmap.md`

### 建議開發流程

1. **理解資料**：閱讀並分析官方資料描述，注意 `metadata.csv` 中的欄位與各個 TSV 檔案的結構
2. **使用模板**：依照 `ImmuneStatePredictor` 介面的要求實作 `fit`、`predict_proba` 以及 `identify_sequences` 等方法
3. **優化模型**：嘗試不同的表示方法（例如細胞序列嵌入、Transformer 模型），透過交叉驗證與公榜成績來迭代模型
4. **生成提交檔**：按照規範生成 `submissions.csv`，確保欄位順序與格式與官方範例一致

---

## 6. MCP 伺服器與 Agent Skills

### MCP 伺服器概覽

MCP（Model Context Protocol）允許大模型安全地存取外部工具與資料來源：

| 類別 | 伺服器 | 說明 |
|---|---|---|
| **核心／參考實作** | `everything`、`fetch`、`filesystem`、`git`、`memory`、`sequential‑thinking`、`time` | 讀寫檔案、抓取網頁、操作 Git 倉庫、持久記憶、時間與時區處理等 |
| **開發工具** | `git`、`github`、`gitlab`、`sentry` | 讀取或操作程式碼庫、Issue 追蹤或錯誤日誌 |
| **資料儲存** | `postgresql`、`sqlite`、`google-drive` | 只讀資料庫或雲端硬碟存取功能 |
| **網頁自動化** | `brave‑search`、`puppeteer` | 網路搜尋與瀏覽器自動化 |
| **生產力工具** | `slack`、`google‑maps` | 即時通訊與地圖查詢 |
| **AI / 其他** | `everart`、`aws‑kb‑retrieval` 等 | 圖像生成或知識庫檢索等專用服務 |

### 啟動 MCP 伺服器

```bash
# 以 TypeScript 版啟動 memory 伺服器
npx -y @modelcontextprotocol/server-memory

# 以 Python 版啟動 git 伺服器
pip install mcp-server-git
python -m mcp_server_git
```

### MCP 配置範例 (`claude_mcp_config.json`)

```json
{
  "mcpServers": {
    "memory": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-memory"]
    },
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "./"]
    },
    "git": {
      "command": "uvx",
      "args": ["mcp-server-git"],
      "env": {
        "GIT_CLONE_DIR": "./repos"
      }
    }
  }
}
```

### Agent Skills

**Skills 與 MCP 的差別**：Skills 提供模型內部的專業知識（例如文件編輯、數據分析流程），MCP 伺服器則讓模型能呼叫外部服務。

**安裝 skills**：

```
/plugin marketplace add anthropic‑agent‑skills
/plugin install document‑skills@anthropic‑agent‑skills
/plugin install example‑skills@anthropic‑agent‑skills
```

若要自訂技能，可在 `.claude/skills/` 下建立資料夾並撰寫 `SKILL.md`。

---

## 7. 進階資源（用來長期升級）

這個骨架有刻意為以下方向預留空間，未來可以一個一個導入：

- **immuneML 平台**：完整的 AIRR ML pipeline 與 YAML‑based spec，可作為特徵與模型設計的靈感來源
- **CompAIRR**：高速計算 repertoire 之間的序列重疊，用來挖共用 motif 或做 graph‑based feature
- **CausalAIRR 與相關研究**：幫助處理 batch effect、混雜因子（confounders）與資料集外泛化問題

這些不會一開始就全部包進 requirements，而是當你打算往該方向走時，再請 Claude 幫你把依賴與程式結構補齊。

---

## 8. 下一步建議

1. **確認 baseline 能跑完**：產生合法的 `submission.csv`
2. **列出實驗計畫**：在 `docs/model_roadmap.md` 裡列出短期與中期的實驗方向
3. **使用自訂 Skill**：用 `skills/airr-ml25-research/SKILL.md` 作為自訂 Skill，讓 Claude 在處理這個專案時有「預設腦袋」：
   - 優先考慮交叉資料集的穩健性，而不是單一資料集的分數
   - 有意識地避免 overfitting 與資料洩漏
   - 把「可解釋性」當作第一等公民，而不只是追 leaderboard

---

## 9. 聯絡與授權

本資料夾僅用於學術與教學示範，未包含官方資料集。請務必遵守 Kaggle 競賽規則及公開授權要求。
