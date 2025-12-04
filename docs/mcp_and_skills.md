# MCP & Skills integration notes

這份文件是專門寫給「未來在這個 repo 裡開啟 Claude Code 的你」以及 Claude 自己看的。

---

## 1. MCP 伺服器建議組合

在 `claude_mcp_config.json`（你可以放在專案根目錄或使用者全域設定）中，建議至少啟用：

- `filesystem`：讀寫本專案的檔案（必要）。
- `git`：查看 commit history / diff（選用）。
- `memory`：存 TODO / 實驗摘要（選用）。

在終端機裡，大致會是這種風格（請依照實際安裝方式調整）：

```bash
# 例：使用 node 版 MCP server
npx @modelcontextprotocol/server-filesystem --root . &
npx @modelcontextprotocol/server-git --root . &
npx @modelcontextprotocol/server-memory &
```

然後在 `claude_mcp_config.json` 中加入對應 entries，例如（示意）：

```jsonc
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["@modelcontextprotocol/server-filesystem", "--root", "."]
    },
    "git": {
      "command": "npx",
      "args": ["@modelcontextprotocol/server-git", "--root", "."]
    },
    "memory": {
      "command": "npx",
      "args": ["@modelcontextprotocol/server-memory"]
    }
  }
}
```

> 實際欄位名稱請以官方文件與你本機的設定為準。

---

## 2. Skills（插件）建議

### 2.1 本專案內建 Skill

- `skills/airr-ml25-research/SKILL.md`  
  讓 Claude 在處理 AIRR‑ML‑25 任務時，有較聚焦的思考模式（模型 / 特徵 / 因果）。

在 Claude Code 中：

1. 打開本專案。
2. 在 Command Palette / 設定中加入這個 Skill 所在路徑。
3. 啟用 Skill 後，再開始請 Claude 幫忙設計實驗。

### 2.2 可考慮安裝的通用 Skills / Plugins

你可以從官方或社群的 Skill 集合中挑選，例如：

- 文件處理 / 筆記類：
  - 能夠讀寫 Markdown / PDF / docx，適合整理 `docs/` 與論文筆記。
- 資料科學輔助：
  - 幫忙推導統計檢定、畫圖、產生報告 skeleton。
- 專案管理：
  - 用於整理 TODO、issue 與實驗排程。

因為具體清單會隨時間更新，建議你直接：

1. 在 Claude 的 Skill / Plugin marketplace 搜尋「data science」「research」「documentation」等關鍵字。
2. 先安裝少數幾個你確定會用到的，避免訊息過載。
3. 明確告訴 Claude「可以使用哪些 Skill」，降低混亂。

---

## 3. 寫給 Claude 的操作守則

當你在這個 repo 啟用 MCP + Skills 時，請遵守：

1. **最小權限**：只在必要時使用 `filesystem` 與 `git` 進行修改或檢視。
2. **可重現性優先**：
   - 對於安裝指令（例如新增 Python 套件），請把命令也寫進 `README.md` 或 `requirements.txt`。
3. **明確說明來源**：
   - 當你引用外部工具（immuneML, CompAIRR 等）時，要在 `docs/model_roadmap.md` 或相關文件中註記來源與使用方式。

這樣未來你（或別人）在另一台機器上重跑，就不會卡在「我到底當時是怎麼裝好這些東西的？」這種問題上。
