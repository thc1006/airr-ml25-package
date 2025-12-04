# CLAUDE.md — AIRR‑ML‑25 Kaggle Project

## Project role

You are helping the user work on the **AIRR‑ML‑25: Adaptive Immune Profiling Challenge 2025** on Kaggle.  
Your main jobs:

1. Build and iterate ML models that:
   - (a) predict repertoire‑level immune state labels, and
   - (b) identify label‑associated receptor sequences.
2. Keep everything reproducible and competition‑safe (no leaking other people’s private code or data).
3. Automate as much of the workflow as possible with Claude Code (bash, Python, MCP servers, and Skills).

Assume the user is an expert in ML / telecom / neuroscience and comfortable with Python and Linux.

## Tools you should use

Inside this repo, prefer these tools in roughly this order:

- **bash / shell**  
  - File + directory inspection, Git operations, Kaggle CLI, running Python entrypoints.
- **python**  
  - Data loading, feature engineering, model training, cross‑validation, submission generation.
- **MCP servers** (configured in `claude_mcp_config.json`):
  - `filesystem`: read / write project files during refactors.
  - `git`: inspect history and diffs; never push without explicit user request.
  - `memory`: store high‑level plans, TODO lists, and experiment notes.
- **Skills**  
  - Use the custom `airr-ml25-research` Skill for domain‑specific reasoning and long‑term planning.
  - Use marketplace Skills like `document-skills` for report drafting and PDF / docx handling when relevant.

If a tool is missing or mis‑configured, propose concrete commands to install or fix it instead of guessing.

## Project layout

Treat the repo as having this canonical structure:

- `CLAUDE.md` — this file.
- `README.md` — quickstart and high‑level overview.
- `requirements.txt` — Python packages needed for local development.
- `src/airr_ml25/`
  - `config.py` — path handling and simple dataset config.
  - `data.py` — loaders for metadata and repertoire files.
  - `features.py` — feature extraction (k‑mers, V/J usage, simple stats).
  - `models/baseline_logreg.py` — L1‑regularized logistic regression baseline.
  - `submission.py` — helpers to turn predictions into a Kaggle‑compatible submission.
- `notebooks/`
  - `00_quick_eda.py` (or `.ipynb`) — EDA and sanity checks.
- `docs/`
  - `challenge_overview.md` — distilled version of competition rules and scoring.
  - `data_format.md` — notes about metadata columns and sequence files.
  - `model_roadmap.md` — research directions and experiment backlog.
  - `mcp_and_skills.md` — how to wire MCP and Skills into this project.
- `skills/airr-ml25-research/`
  - `SKILL.md` — custom Skill for AIRR‑ML‑25 reasoning.

When adding new files, keep them inside `src/`, `notebooks/`, or `docs/` unless the user explicitly asks otherwise.

## Coding style and practices

- Use **Python 3.10+** and type hints where helpful.
- Prefer **pandas** + **numpy** as the core data stack.
- For models:
  - Start with scikit‑learn (logistic regression, linear models, tree‑based methods).
  - When proposing heavier models (XGBoost, LightGBM, transformers), clearly label them as optional and update `requirements.txt` only when the user agrees.
- Follow these guidelines:
  - Keep functions small and composable; avoid giant monolithic scripts.
  - Add docstrings with argument and return type descriptions.
  - Avoid hard‑coding absolute paths; always parameterize dataset locations.

## Kaggle‑specific constraints

- Never assume access to **internet** inside Kaggle kernels.
- Always:
  - Expose a single main entrypoint script (e.g. `python -m airr_ml25.submission ...`).
  - Use **relative paths** rooted at `/kaggle/input/adaptive-immune-profiling-challenge-2025` (or a user‑passed root).
  - Avoid writing large intermediate files to disk unless necessary; prefer in‑memory pipelines.
- Do **not** copy‑paste other competitors’ code or proprietary model weights.  
  Summarize ideas in your own words and implement them cleanly from scratch.

## How to use this project with Claude Code

1. **Clone the repo** to the dev machine (or Kaggle notebook environment).
2. Open the folder in Claude Code.
3. Ask Claude to:
   - Inspect `docs/challenge_overview.md` and `docs/data_format.md`.
   - Run `pip install -r requirements.txt`.
   - Execute the baseline training flow:
     - `python -m airr_ml25.submission --train_dir ... --test_dir ... --out_path submissions.csv`
4. Iterate:
   - Add features and models in `src/airr_ml25/`.
   - Log experiments and insights into `docs/model_roadmap.md`.
   - Use the `airr-ml25-research` Skill for deeper domain reasoning.

Always keep the user in the loop before making any destructive change (deleting files, force‑pushing Git branches, etc.).
