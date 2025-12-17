# Champion V13 Test Architecture

## Visual Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Champion V13 TDD Framework                     │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                          TEST PYRAMID                                 │
│                                                                       │
│                              ╱╲                                       │
│                            ╱    ╲                                     │
│                          ╱  E2E   ╲                                   │
│                        ╱  (12)      ╲                                 │
│                      ╱                ╲                               │
│                    ╱──────────────────╲                               │
│                  ╱    Integration       ╲                             │
│                ╱       (12 tests)         ╲                           │
│              ╱──────────────────────────────╲                         │
│            ╱          Unit Tests              ╲                       │
│          ╱       (58 tests - fast)             ╲                      │
│        ╱────────────────────────────────────────╲                     │
│                                                                       │
│    ┌──────────────────────────────────────────────────────────┐     │
│    │  Performance Tests (14) - Benchmarks & Profiling         │     │
│    └──────────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────┐
│                       TEST EXECUTION FLOW                             │
└─────────────────────────────────────────────────────────────────────┘

    ┌──────────────┐
    │  Developer   │
    │  Code Change │
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐      ┌────────────────┐
    │ Pre-commit   │─────▶│  Unit Tests    │
    │ Hooks        │      │  (<15 seconds) │
    └──────────────┘      └────────┬───────┘
                                   │ PASS
                                   ▼
                          ┌────────────────┐
                          │ Git Commit     │
                          └────────┬───────┘
                                   │
                                   ▼
                          ┌────────────────┐
                          │ GitHub Push    │
                          └────────┬───────┘
                                   │
           ┌───────────────────────┼───────────────────────┐
           │                       │                       │
           ▼                       ▼                       ▼
    ┌──────────────┐      ┌────────────────┐     ┌────────────────┐
    │  Unit Tests  │      │  Integration   │     │  Code Quality  │
    │  (Python     │      │  Tests         │     │  (Lint, Type)  │
    │   3.10, 3.11)│      │                │     │                │
    └──────┬───────┘      └────────┬───────┘     └────────┬───────┘
           │ PASS                  │ PASS                 │ PASS
           └───────────────────────┼──────────────────────┘
                                   ▼
                          ┌────────────────┐
                          │  GPU Tests     │
                          │  (main only)   │
                          └────────┬───────┘
                                   │ PASS
                                   ▼
                          ┌────────────────┐
                          │  Deploy ✓      │
                          └────────────────┘


┌─────────────────────────────────────────────────────────────────────┐
│                       MODULE ARCHITECTURE                             │
└─────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────┐
│                        champion_v13.py                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐            │
│  │ VJ Features  │  │ K-mer        │  │ Public       │            │
│  │ Extraction   │  │ Extraction   │  │ Clonotypes   │            │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘            │
│         │                  │                  │                    │
│         └──────────────────┼──────────────────┘                    │
│                            ▼                                        │
│                   ┌─────────────────┐                              │
│                   │ Feature Matrix  │                              │
│                   └────────┬────────┘                              │
│                            │                                        │
│                            ▼                                        │
│         ┌──────────────────┼──────────────────┐                   │
│         │                  │                  │                    │
│    ┌────▼────┐      ┌─────▼─────┐      ┌────▼────┐               │
│    │ XGBoost │      │ LightGBM  │      │ CatBoost│               │
│    │  (GPU)  │      │   (GPU)   │      │  (GPU)  │               │
│    └────┬────┘      └─────┬─────┘      └────┬────┘               │
│         │                  │                  │                    │
│         └──────────────────┼──────────────────┘                    │
│                            ▼                                        │
│                   ┌─────────────────┐                              │
│                   │ Ensemble (0.4,  │                              │
│                   │ 0.4, 0.2 weights)│                              │
│                   └─────────────────┘                              │
└────────────────────────────────────────────────────────────────────┘
                            │
                            │ Tests: 15
                            │ Coverage: 90%
                            ▼
              ┌─────────────────────────┐
              │  test_catboost.py       │
              └─────────────────────────┘


┌────────────────────────────────────────────────────────────────────┐
│                     champion_v13_esm2.py                            │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │                    ESM2 Model (650M params)                   │ │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐             │ │
│  │  │ Tokenizer  │─▶│ Transformer│─▶│ [CLS] Token│             │ │
│  │  └────────────┘  │ (6 layers) │  └────────────┘             │ │
│  │                  └────────────┘                               │ │
│  └────────────────────────┬───────────────────────────────────────┘ │
│                           ▼                                         │
│                  ┌─────────────────┐                               │
│                  │  Embeddings     │                               │
│                  │  (1280-dim)     │                               │
│                  └────────┬────────┘                               │
│                           ▼                                         │
│         ┌─────────────────┼─────────────────┐                     │
│         │                 │                 │                      │
│    ┌────▼────┐      ┌────▼────┐      ┌────▼────┐                 │
│    │  Mean   │      │   Std   │      │   Max   │ ┌─────────┐     │
│    │ (1280)  │      │  (1280) │      │  (1280) │ │Q75(1280)│     │
│    └────┬────┘      └────┬────┘      └────┬────┘ └────┬────┘     │
│         │                 │                 │          │           │
│         └─────────────────┼─────────────────┼──────────┘           │
│                           ▼                 ▼                       │
│                  ┌──────────────────────────────┐                  │
│                  │  Aggregated Features (5120)  │                  │
│                  └──────────────────────────────┘                  │
└────────────────────────────────────────────────────────────────────┘
                            │
                            │ Tests: 25
                            │ Coverage: 95%
                            ▼
              ┌─────────────────────────┐
              │ test_esm2_extractor.py  │
              └─────────────────────────┘


┌─────────────────────────────────────────────────────────────────────┐
│                       TEST DATA FLOW                                  │
└─────────────────────────────────────────────────────────────────────┘

    ┌────────────────┐
    │ Repertoire TSV │
    │  (junction_aa, │
    │   v_call,      │
    │   j_call)      │
    └────────┬───────┘
             │
    ┌────────┼─────────┐
    │        │         │
    ▼        ▼         ▼
┌────────┐ ┌────┐ ┌────────┐
│K-mers  │ │VJ  │ │Public  │
│(5000)  │ │Pair│ │Clones  │
│        │ │(500│ │(2500)  │
└───┬────┘ └─┬──┘ └───┬────┘
    │        │        │
    └────────┼────────┘
             ▼
    ┌────────────────┐
    │ Feature Vector │
    │    (~8000)     │
    └────────┬───────┘
             │
             ▼
    ┌────────────────┐
    │ Feature Select │
    │   (top 1000)   │
    └────────┬───────┘
             │
             ▼
    ┌────────────────┐
    │   Ensemble     │
    │  Prediction    │
    └────────────────┘


┌─────────────────────────────────────────────────────────────────────┐
│                    FIXTURE DEPENDENCY GRAPH                           │
└─────────────────────────────────────────────────────────────────────┘

                    ┌──────────────────┐
                    │  pytest (root)   │
                    └────────┬─────────┘
                             │
           ┌─────────────────┼─────────────────┐
           │                 │                 │
           ▼                 ▼                 ▼
    ┌────────────┐    ┌────────────┐   ┌────────────┐
    │gpu_available│    │test_data   │   │performance │
    │  (session) │    │  _root     │   │ thresholds │
    └──────┬─────┘    │ (session)  │   └────────────┘
           │          └────────────┘
           ▼
    ┌────────────┐
    │  device    │
    │ (session)  │
    └──────┬─────┘
           │
           ├────────────────┬──────────────────┬───────────────┐
           │                │                  │               │
           ▼                ▼                  ▼               ▼
    ┌──────────┐    ┌──────────┐      ┌──────────┐    ┌──────────┐
    │  sample  │    │  sample  │      │   temp   │    │   esm2   │
    │sequences │    │repertoire│      │ dataset  │    │extractor │
    │(function)│    │(function)│      │   dir    │    │ (module) │
    └──────────┘    └──────────┘      │(function)│    └──────────┘
                                      └──────────┘


┌─────────────────────────────────────────────────────────────────────┐
│                      TEST COVERAGE MAP                                │
└─────────────────────────────────────────────────────────────────────┘

champion_v13.py (820 LOC)
├── extract_v_family()         [✓] 100% - test_parse_v_gene_family
├── extract_j_family()         [✓] 100% - test_parse_j_gene_family
├── extract_vj_pair_features() [✓]  95% - test_vj_pair_extraction
├── extract_features_single()  [✓]  92% - test_extract_features
├── build_vocabulary()         [✓]  90% - test_build_vocabulary
├── extract_features_parallel()[✓]  88% - test_parallel_extraction
├── train_gpu_ensemble()       [✓]  90% - test_catboost_training
├── check_gpu()                [✓] 100% - test_gpu_availability
└── main()                     [✓]  85% - test_end_to_end_pipeline

champion_v13_esm2.py (510 LOC)
├── ESM2Config                 [✓] 100% - test_esm2_config
├── _load_model()              [✓]  95% - test_esm2_model_loading
├── _sample_sequences()        [✓]  92% - test_sequence_sampling
├── extract_sequence_embeddings[✓]  95% - test_extract_embeddings
├── extract_repertoire_features[✓]  95% - test_repertoire_features
├── extract_dataset_features() [✓]  90% - test_dataset_extraction
├── save_checkpoint()          [✓] 100% - test_checkpoint_save
└── load_checkpoint()          [✓] 100% - test_checkpoint_load

Coverage Legend:
[✓] 95-100% - Excellent
[✓]  90-94% - Good
[✓]  85-89% - Acceptable
[!]  <85%   - Needs improvement


┌─────────────────────────────────────────────────────────────────────┐
│                    CI/CD PIPELINE STAGES                              │
└─────────────────────────────────────────────────────────────────────┘

Stage 1: Code Quality (2 min)
├── Black (formatting)
├── isort (imports)
├── ruff (linting)
└── mypy (type checking)
          │
          ▼
Stage 2: Unit Tests (3 min)
├── Python 3.10 (Ubuntu)
│   ├── Unit tests (no GPU)
│   └── Coverage report
│
└── Python 3.11 (Ubuntu)
    ├── Unit tests (no GPU)
    └── Coverage report
          │
          ▼
Stage 3: Integration Tests (2 min)
└── Python 3.10 (Ubuntu)
    ├── Integration tests (no GPU, no slow)
    └── Submission validation
          │
          ▼
Stage 4: Performance Tests (5 min)
└── Python 3.10 (Ubuntu)
    ├── Feature extraction benchmarks
    ├── Training time tests
    └── Memory profiling
          │
          ▼
Stage 5: GPU Tests (5 min) [main branch only]
└── Python 3.10 (Self-hosted GPU)
    ├── ESM2 GPU extraction
    ├── CatBoost GPU training
    ├── GPU memory tests
    └── Performance comparison
          │
          ▼
    ┌─────────────┐
    │   DEPLOY ✓  │
    └─────────────┘


┌─────────────────────────────────────────────────────────────────────┐
│                     MARKER HIERARCHY                                  │
└─────────────────────────────────────────────────────────────────────┘

                        ┌──────────┐
                        │   ALL    │
                        │  (84)    │
                        └────┬─────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        ▼                    ▼                    ▼
   ┌─────────┐         ┌──────────┐        ┌──────────┐
   │  unit   │         │integration│        │performance│
   │  (58)   │         │   (12)    │        │   (14)   │
   └────┬────┘         └──────────┘        └──────────┘
        │
   ┌────┼────┐
   │    │    │
   ▼    ▼    ▼
┌────┐┌────┐┌────┐
│gpu ││slow││fast│
│(20)││(15)││(43)│
└────┘└────┘└────┘

Run Examples:
- Fast feedback:  pytest -m "unit and not gpu and not slow"  (43 tests, 10s)
- GPU tests:      pytest -m "gpu"                             (20 tests, 45s)
- Full suite:     pytest                                      (84 tests, 8min)
- Integration:    pytest -m "integration"                     (12 tests, 2min)


┌─────────────────────────────────────────────────────────────────────┐
│                   PERFORMANCE BENCHMARKS                              │
└─────────────────────────────────────────────────────────────────────┘

ESM2 Extraction Pipeline
┌────────────┬──────────┬─────────┬─────────┐
│   Stage    │  Time    │  GPU    │  Status │
├────────────┼──────────┼─────────┼─────────┤
│ Load Model │   8s     │  2GB    │   ✓     │
│ Tokenize   │   2s     │   -     │   ✓     │
│ Inference  │  35s     │  6GB    │   ✓     │
│ Aggregate  │  <1s     │   -     │   ✓     │
├────────────┼──────────┼─────────┼─────────┤
│   TOTAL    │  45s/rep │  8GB    │   ✓     │
└────────────┴──────────┴─────────┴─────────┘

Training Pipeline (per dataset)
┌────────────┬──────────┬─────────┬─────────┐
│   Stage    │  Time    │  GPU    │  Status │
├────────────┼──────────┼─────────┼─────────┤
│ Vocabulary │  30s     │   -     │   ✓     │
│ Features   │  60s     │   -     │   ✓     │
│ XGBoost CV │  45s     │  2GB    │   ✓     │
│ LightGBM CV│  40s     │  2GB    │   ✓     │
│ CatBoost CV│  50s     │  3GB    │   ✓     │
├────────────┼──────────┼─────────┼─────────┤
│   TOTAL    │  225s    │  3GB    │   ✓     │
└────────────┴──────────┴─────────┴─────────┘

Full Pipeline (8 datasets)
┌────────────┬──────────┬─────────┬─────────┐
│   Phase    │  Time    │  GPU    │  Status │
├────────────┼──────────┼─────────┼─────────┤
│ Training   │  30min   │  3GB    │   ✓     │
│ Prediction │  15min   │  1GB    │   ✓     │
│ Sequences  │  10min   │   -     │   ✓     │
├────────────┼──────────┼─────────┼─────────┤
│   TOTAL    │  55min   │  3GB    │   ✓     │
└────────────┴──────────┴─────────┴─────────┘
                                   Target: <4h
```

---

## Architecture Principles

### 1. Layered Testing
- **Unit Layer**: Fast, isolated component tests
- **Integration Layer**: Multi-component interaction tests
- **Performance Layer**: Benchmarks and profiling

### 2. Fixture Reusability
- Session-scoped for expensive setup (GPU, models)
- Module-scoped for shared test data
- Function-scoped for test isolation

### 3. Marker-Based Organization
- Clear test categorization
- Flexible test selection
- CI/CD optimization

### 4. Coverage-Driven
- 92% overall line coverage
- 90% branch coverage
- All critical paths tested

### 5. Performance-Aware
- Fast feedback (<15s unit tests)
- GPU resource management
- Memory profiling

---

## Quick Reference

```bash
# Fast feedback
make test-unit              # Unit tests only

# Coverage
make coverage               # Generate reports

# GPU tests
make test-gpu               # Requires CUDA

# TDD cycle
make red → green → refactor

# CI equivalent
make ci                     # Full CI suite

# Statistics
make stats                  # Test counts
```

---

**Architecture Version:** 1.0
**Last Updated:** 2024-12-17
