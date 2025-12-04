# CLAUDE.md — AIRR-ML-25 Competition Champion Framework

> **Mission**: Win the AIRR-ML-25: Adaptive Immune Profiling Challenge 2025
> **Deadline**: December 17, 2025 (06:59 UTC)
> **Current Top Score**: 0.81364 (GROZD team)
> **Target**: Score > 0.82 to secure 1st place ($5,000 + Nature Methods authorship)

---

## 1. Competition Overview

### 1.1 Dual-Task Challenge

| Task | Description | Metric | Weight |
|------|-------------|--------|--------|
| **Task A** | Predict immune state (disease/healthy) for each repertoire | ROC-AUC | Part of weighted avg |
| **Task B** | Identify top 50,000 label-associated receptor sequences per dataset | Jaccard Similarity | Part of weighted avg |

### 1.2 Critical Numbers

- **8** training datasets, **11** test datasets
- **4,213** test repertoires to predict
- **404,213** total submission rows (4,213 predictions + 8×50,000 sequences)
- **19.94 GB** dataset size
- **5 submissions/day** limit

### 1.3 Data Format

```
train_datasets/
├── train_dataset_1/
│   ├── metadata.csv       # repertoire_id, filename, label_positive
│   └── *.tsv              # junction_aa, v_call, j_call, [d_call, templates]
└── train_dataset_{2-8}/

test_datasets/
├── test_dataset_1/
└── test_dataset_{2-11}/   # Some datasets have multiple test sets (e.g., 7_1, 7_2)
```

---

## 2. Project Architecture

### 2.1 Directory Structure

```
airr-ml25-package/
├── CLAUDE.md                    # This file - Claude's operating instructions
├── main.py                      # Standalone baseline trainer (official template)
├── requirements.txt             # Python dependencies
├── kaggle.json                  # API credentials (DO NOT COMMIT TO PUBLIC)
├── claude_mcp_config.json       # MCP server configuration
│
├── .claude/                     # Claude Code configuration
│   ├── settings.json            # Permissions and environment
│   ├── agents/                  # 15 specialized AI agents (see Section 12)
│   │   ├── competition-master.md
│   │   ├── data-scientist.md
│   │   ├── ml-engineer.md
│   │   └── ... (12 more)
│   └── skills/                  # Custom skills (official location)
│       └── airr-ml25-research/
│           └── SKILL.md
│
├── src/airr_ml25/               # Core Python package
│   ├── config.py                # Path handling, dataset configuration
│   ├── data.py                  # Data loaders for metadata and sequences
│   ├── features.py              # Feature extraction (k-mers, V/J usage)
│   ├── submission.py            # Submission file generation
│   └── models/
│       └── baseline_logreg.py   # L1-regularized logistic regression
│
├── notebooks/                   # EDA and experiments
│   └── 00_quick_eda.py
│
├── docs/                        # Documentation hub
│   ├── challenge_overview.md    # Competition rules digest
│   ├── data_format.md           # Column specifications
│   ├── model_roadmap.md         # Experiment backlog and results
│   └── mcp_and_skills.md        # Integration notes
│
├── craw/                        # Crawled competition info (8 .md files)
│   └── 01_Overview.md ~ 08_Submissions.md
│
├── data/                        # Dataset directory (gitignored, ~19GB)
│   ├── train_datasets/train_datasets/train_dataset_{1-8}/
│   ├── test_datasets/test_datasets/test_dataset_{1-8_3}/
│   └── sample_submissions.csv
│
└── example-baseline-predictor-using-code-template.ipynb
```

### 2.2 Key Entry Points

```bash
# Run main baseline
python main.py --train_dir ./data/train_datasets/train_dataset_1 \
               --test_dirs ./data/test_datasets/test_dataset_1 \
               --out_dir ./results --n_jobs 4

# Or use modular package
python -m airr_ml25.submission --train-root ./data/train_datasets \
                               --test-root ./data/test_datasets \
                               --out-path ./submission.csv
```

---

## 3. Operating Principles (For Claude)

### 3.1 Long-Running Agent Best Practices

Following Anthropic's [effective harnesses guide](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents):

1. **One Feature at a Time**: Never attempt multiple major changes simultaneously
2. **Incremental Progress**: Update `docs/model_roadmap.md` after each experiment
3. **Clean State Handoffs**: Leave code documented and tests passing
4. **Explicit Verification**: Always verify features work before marking complete

### 3.2 Tool Use Strategy

Following Anthropic's [advanced tool use guide](https://www.anthropic.com/engineering/advanced-tool-use):

1. **Parallel Execution**: When reading multiple files or running independent tasks, batch them
2. **Programmatic Orchestration**: For complex workflows, use Python scripts not manual tool calls
3. **Error Prevention**: Validate inputs before expensive operations

### 3.3 Priority Stack (Highest First)

1. **Reproducibility**: Fixed seeds, documented splits, version-pinned dependencies
2. **Cross-Dataset Generalization**: Leave-one-dataset-out validation > single-split CV
3. **Scientific Insight**: Interpretable features that map to biology
4. **Leaderboard Score**: Optimize only after 1-3 are satisfied

### 3.4 Anti-Patterns to Avoid

- Marking tasks complete without end-to-end testing
- Overfitting to public leaderboard (different from private leaderboard)
- Complex models before simple baselines are exhausted
- Hard-coding paths (use config.py and CLI arguments)

---

## 4. MCP Server Configuration

The project uses four MCP servers configured in `claude_mcp_config.json`:

```json
{
  "mcpServers": {
    "memory": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-memory"]
    },
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "--root", "."]
    },
    "git": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-git", "--root", "."]
    },
    "playwright": {
      "command": "npx",
      "args": ["-y", "@playwright/mcp@latest"]
    }
  }
}
```

### Usage Guidelines

- **memory**: Store experiment summaries, TODO lists, key insights across sessions
- **filesystem**: Read/write project files during refactoring
- **git**: Inspect history and diffs; **never push without explicit user request**
- **playwright**: Browser automation for web scraping, testing, and data collection

---

## 5. Winning Strategy Roadmap

### Phase 1: Foundation (Current)
- [x] Project structure setup
- [x] Baseline model (k-mer + L1 LogReg)
- [ ] Download and validate dataset
- [ ] Run baseline on all 8 datasets
- [ ] First submission to establish baseline score

### Phase 2: Feature Engineering
- [ ] Multi-scale k-mers (k=3,4,5)
- [ ] V/J gene usage patterns
- [ ] VJ pair combinations
- [ ] Clonality metrics (Shannon entropy, Gini, D50)
- [ ] Public clonotypes (shared across individuals)
- [ ] CDR3 length distribution statistics

### Phase 3: Model Enhancement
- [ ] XGBoost/LightGBM ensemble
- [ ] Per-dataset models with ensemble
- [ ] Dataset ID as feature (handle distribution shift)
- [ ] Stratified cross-validation with dataset awareness

### Phase 4: Sequence Identification (Task B)
- [ ] Feature importance from trained models
- [ ] K-mer to sequence mapping
- [ ] SHAP values for interpretability
- [ ] Motif clustering and deduplication

### Phase 5: Advanced Techniques (If Time Permits)
- [ ] Protein language model embeddings (ESM, ProtBERT)
- [ ] Attention-based aggregation
- [ ] Graph-based sequence similarity features

---

## 6. Submission Format Specification

### Required Output Structure

```csv
ID,dataset,label_positive_probability,junction_aa,v_call,j_call
rep_001,test_dataset_1,0.85,-999.0,-999.0,-999.0
...                                                        # 4,213 prediction rows
train_dataset_1_seq_top_1,train_dataset_1,-999.0,CASSLGQAY,TRBV20-1,TRBJ2-7
...                                                        # 50,000 per dataset × 8
```

- **Missing values**: Use `-999.0` (Kaggle rejects NaN)
- **Total rows**: Exactly 404,213
- **ID format**: repertoire_id for predictions, custom for sequences

---

## 7. Code Template Compliance

For prize eligibility and Nature Methods authorship, implement `ImmuneStatePredictor`:

```python
class ImmuneStatePredictor:
    def __init__(self, n_jobs: int = 1, device: str = 'cpu', **kwargs):
        pass

    def fit(self, train_dir_path: str) -> 'ImmuneStatePredictor':
        """Train on data in train_dir_path."""
        pass

    def predict_proba(self, test_dir_path: str) -> pd.DataFrame:
        """Return DataFrame with columns: ID, dataset, label_positive_probability, ..."""
        pass

    def identify_associated_sequences(self, train_dir_path: str, top_k: int = 50000) -> pd.DataFrame:
        """Return top_k important sequences."""
        pass
```

**Run interface**:
```bash
python3 -m submission.main --train_dir ... --test_dir ... --out_dir ... --n_jobs 4 --device cpu
```

---

## 8. Quick Commands

### Dataset Operations
```bash
# Download competition data
kaggle competitions download -c adaptive-immune-profiling-challenge-2025 -p ./data/

# Extract
cd data && unzip adaptive-immune-profiling-challenge-2025.zip && cd ..

# Check data structure
ls -la data/train_datasets/
ls -la data/test_datasets/
```

### Development
```bash
# Setup environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run EDA
python notebooks/00_quick_eda.py --train-root ./data/train_datasets

# Run baseline
python main.py --train_dir ./data/train_datasets/train_dataset_1 \
               --test_dirs ./data/test_datasets/test_dataset_1 \
               --out_dir ./results --n_jobs 4
```

### Submission
```bash
# Validate submission format
python -c "import pandas as pd; df = pd.read_csv('submissions.csv'); print(df.shape, df.columns.tolist())"

# Submit to Kaggle
kaggle competitions submit -c adaptive-immune-profiling-challenge-2025 \
                          -f submissions.csv -m "Experiment: description"
```

---

## 9. Session Startup Checklist

When starting a new session, Claude should:

1. **Read current state**:
   ```
   - Check docs/model_roadmap.md for experiment history
   - Run git log --oneline -10 for recent changes
   - Review any running background tasks
   ```

2. **Verify environment**:
   ```
   - Check data/ directory exists and has expected structure
   - Confirm Python environment is activated
   - Validate requirements are installed
   ```

3. **Select next task**:
   ```
   - Review roadmap for next priority item
   - Confirm understanding with user if ambiguous
   - Create focused TODO list for session
   ```

---

## 10. Key References

### Competition Resources
- [Official Overview](https://www.kaggle.com/competitions/adaptive-immune-profiling-challenge-2025/overview)
- [Code Template Repo](https://github.com/uio-bmi/predict-airr)
- [Pre-registered Protocol](https://github.com/uio-bmi/adaptive_immune_profiling_challenge_2025/blob/main/registered_report.pdf)

### Domain Knowledge
- [State-of-the-art in AIRR Mining](https://www.sciencedirect.com/science/article/pii/S2452310020300524)
- [Modern Hopfield Networks for Repertoires](https://doi.org/10.1101/2020.04.12.038158)
- [immuneML Platform](https://pmc.ncbi.nlm.nih.gov/articles/PMC10312379/)

### Top Community Notebooks
- [XGBoost Baseline](https://www.kaggle.com/code/bakuer30/air-ml25-xgboost) - 43 votes
- [XGBoost + PCA](https://www.kaggle.com/code/jirkaborovec/airr-ml-25-naive-baseline-with-xgboost-pca) - 28 votes
- [TabPFN Approach](https://www.kaggle.com/code/dkriuchkova/airrml25-tabpfn) - 4 votes

---

## 11. Safety and Compliance

- **No data leakage**: Never use test labels or post-deadline information
- **Open source requirement**: All winning code must be MIT licensed
- **Original work**: Summarize ideas, implement from scratch
- **Credentials security**: Never commit kaggle.json to public repos

---

## 12. Available AI Agents (.claude/agents/)

This project includes 15 specialized AI agents for different tasks. Invoke them proactively based on task type.

### 12.1 Competition & Strategy Agents

| Agent | Model | Purpose | When to Use |
|-------|-------|---------|-------------|
| **competition-master** | sonnet | Strategic orchestrator, risk management, submission planning | Daily progress review, submission strategy, agent coordination |
| **ensemble-optimizer** | sonnet | WBF parameter tuning, file size optimization, model diversity | Ensemble combination testing, submission file optimization |

### 12.2 Machine Learning Agents

| Agent | Model | Purpose | When to Use |
|-------|-------|---------|-------------|
| **data-scientist** | sonnet | Statistical analysis, ML modeling, EDA, business insights | Data analysis, predictive modeling, A/B testing, customer analytics |
| **ml-engineer** | sonnet | Production ML systems, PyTorch 2.x, model serving, feature engineering | Model deployment, inference optimization, ML infrastructure |
| **mlops-engineer** | sonnet | ML pipelines, MLflow, Kubeflow, experiment tracking | Pipeline automation, model registry, cloud ML platforms |
| **parallel-trainer** | sonnet | Multi-GPU orchestration, Docker container training, resource allocation | Mass training (30-50 models), parallel job management, GPU optimization |

### 12.3 Data & Engineering Agents

| Agent | Model | Purpose | When to Use |
|-------|-------|---------|-------------|
| **data-engineer** | sonnet | Data pipelines, Spark, dbt, Airflow, streaming architectures | ETL/ELT pipelines, data warehouse design, real-time streaming |

### 12.4 Code Quality Agents

| Agent | Model | Purpose | When to Use |
|-------|-------|---------|-------------|
| **code-reviewer** | sonnet | AI-powered code analysis, security vulnerabilities, performance review | Code quality assurance, security audit, PR review |
| **python-pro** | sonnet | Python 3.12+, async programming, uv, ruff, modern patterns | Python development, optimization, advanced patterns |
| **tdd-orchestrator** | sonnet | Test-driven development, red-green-refactor, test suite architecture | TDD implementation, test coverage, quality gates |

### 12.5 Web Framework Agents

| Agent | Model | Purpose | When to Use |
|-------|-------|---------|-------------|
| **django-pro** | sonnet | Django 5.x, async views, DRF, Celery, Django Channels | Django web apps, ORM optimization, Django patterns |
| **fastapi-pro** | sonnet | FastAPI, SQLAlchemy 2.0, Pydantic V2, async APIs | FastAPI microservices, WebSocket, API architecture |

### 12.6 Debugging & Operations Agents

| Agent | Model | Purpose | When to Use |
|-------|-------|---------|-------------|
| **debugger** | sonnet | Root cause analysis, error diagnosis, minimal fixes | Any errors, test failures, unexpected behavior |
| **error-detective** | haiku | Log parsing, stack trace analysis, error correlation | Log analysis, production error investigation |
| **context-manager** | haiku | Dynamic context management, vector databases, knowledge graphs | Multi-agent coordination, RAG implementation, memory systems |

### 12.7 Agent Invocation Examples

```bash
# For competition strategy
# Use: competition-master, ensemble-optimizer

# For feature engineering and modeling
# Use: data-scientist → ml-engineer → mlops-engineer

# For code quality
# Use: python-pro → code-reviewer → tdd-orchestrator

# For debugging
# Use: debugger (simple) or error-detective (complex logs)
```

### 12.8 Agent Collaboration Patterns

```
Feature Development:
  data-scientist (analyze) → ml-engineer (implement) → code-reviewer (review)

Model Training Pipeline:
  mlops-engineer (setup) → parallel-trainer (train) → ensemble-optimizer (optimize)

Debugging Workflow:
  error-detective (find patterns) → debugger (fix root cause) → tdd-orchestrator (add tests)
```

---

## 13. Skills (.claude/skills/)

### 13.1 Custom Skills

| Skill | Location | Purpose |
|-------|----------|---------|
| **airr-ml25-research** | `.claude/skills/airr-ml25-research/SKILL.md` | Domain-specific reasoning for AIRR-ML-25 competition |

### 13.2 Skills vs Agents

- **Skills**: Domain knowledge and reasoning frameworks (declarative)
- **Agents**: Action-oriented specialists for specific tasks (imperative)

Use skills when you need deep domain context; use agents when you need specialized task execution.

---

*Last Updated: 2025-12-04*
*Version: 2.1.0*
