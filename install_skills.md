# 安裝 Claude 插件與 Agent Skills

本檔案示範如何在 Claude Code 專案中安裝官方提供的插件與技能集（Skills）。技能是模型內部的工作流程與教學檔案，而插件則是一系列打包好的技能，可直接在 Claude 中調用。您可以根據自身需求選擇適合的技能集，例如文件處理或範例技能。

## 1. 加入技能市集

在 Claude Code 編輯器的聊天區輸入以下指令，以將 Anthropics 官方技能倉庫加入技能市集：

```
/plugin marketplace add anthropic-agent-skills
```

該倉庫包含各類技能，包括：

- **document‑skills**：處理 Word、PDF、PowerPoint、Excel 等檔案，支援建立、修改和解析文檔。
- **example‑skills**：示範性技能，如單元測試生成、程式碼重構、故事生成等。

## 2. 安裝具體技能包

加入市集後，可透過下列指令安裝特定技能包：

```
/plugin install document-skills@anthropic-agent-skills
/plugin install example-skills@anthropic-agent-skills
```

安裝完成後，Claude 會根據對話內容自動載入並使用這些技能。例如，若安裝了 `document-skills`，可以直接要求：「請用 PDF 技能萃取 `path/to/file.pdf` 中的表單欄位」。Claude 會自動調用對應技能來完成任務【351813915382025†L28-L46】。

## 3. 自訂技能

若要建立自己的技能，可在專案中建立 `.claude/skills/<你的技能名稱>/` 目錄，並添加 `SKILL.md` 檔案。`SKILL.md` 需包含 YAML 前言，例如：

```markdown
---
name: my-skill-name
description: 簡要說明技能能做什麼、何時會使用此技能
---

# 技能標題

## 使用說明
提供明確的步驟與示範，讓 Claude 知道執行流程。

## 範例
- 範例 1
- 範例 2

## 規範
- 指導原則 1
- 指導原則 2
```

技能目錄中可放置額外程式（`scripts/`）或模板（`templates/`），在 `SKILL.md` 中引用即可。詳細語法請參考官方文件。

## 4. 注意事項

- Skills 主要用於指導 Claude 進行流程，不適合處理大型資料；外部資料存取應透過 MCP 伺服器完成【167322134020916†L55-L61】。
- 安裝技能或插件之後，請測試其行為是否符合預期，再在正式工作中使用。
- 如需限制技能存取特定工具，可在 `SKILL.md` 的 YAML 前言中加入 `allowed-tools` 欄位，列出允許呼叫的工具名稱。