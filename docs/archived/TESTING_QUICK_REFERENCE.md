# Champion V13 - Testing Quick Reference Card

## TL;DR - Essential Commands

```bash
# Fast feedback (10 seconds)
make test-unit

# Full test suite (8 minutes)
make test

# Coverage report
make coverage

# TDD cycle
make red → green → refactor

# Pre-commit checks
make precommit
```

---

## Test Categories

| Category | Command | Time | Tests |
|----------|---------|------|-------|
| **Unit (fast)** | `pytest -m "unit and not gpu and not slow"` | 10s | 43 |
| **Unit (all)** | `pytest -m unit` | 1min | 58 |
| **Integration** | `pytest -m integration` | 2min | 12 |
| **Performance** | `pytest -m performance` | 5min | 14 |
| **GPU** | `pytest -m gpu` | 45s | 20 |
| **Full Suite** | `pytest` | 8min | 84 |

---

## File Locations

```
tests/
├── test_esm2_extractor.py      # ESM2 tests (25)
├── test_vj_features.py         # VJ tests (18)
├── test_catboost.py            # CatBoost tests (15)
├── test_integration_v13.py     # Integration (12)
└── test_performance_v13.py     # Performance (14)

docs/
├── TDD_WORKFLOW.md             # Complete TDD guide
├── TEST_MATRIX.md              # Coverage matrix
└── TEST_ARCHITECTURE.md        # Architecture diagrams
```

---

## Makefile Targets

### Test Execution
```bash
make test           # All tests (no GPU, no slow)
make test-unit      # Unit tests only (fast)
make test-integration
make test-performance
make test-gpu       # Requires CUDA
make test-all       # Complete suite
```

### Coverage
```bash
make coverage       # HTML + terminal report
make cov-summary    # Quick summary
make cov-open       # Open HTML in browser
```

### TDD Workflow
```bash
make red            # Run tests (expect fail)
make green          # Run tests (expect pass)
make refactor       # Full tests + coverage
```

### Quality
```bash
make lint           # ruff linting
make format         # black + isort
make typecheck      # mypy type checking
make precommit      # All quality checks
```

### Utilities
```bash
make stats          # Test statistics
make clean          # Remove artifacts
make help           # Show all targets
```

---

## pytest Command Patterns

### Filter by Marker
```bash
# Unit tests only
pytest -m unit

# No GPU tests
pytest -m "not gpu"

# Integration + performance
pytest -m "integration or performance"

# Fast tests only
pytest -m "not slow and not gpu"
```

### Filter by Pattern
```bash
# Tests matching "esm2"
pytest -k "esm2"

# Tests NOT matching "slow"
pytest -k "not slow"

# Specific test file
pytest tests/test_esm2_extractor.py
```

### Useful Options
```bash
# Verbose output
pytest -v

# Very verbose
pytest -vv

# Show print statements
pytest -s

# Stop on first failure
pytest -x

# Drop into debugger
pytest --pdb

# Show local variables on failure
pytest -l

# Parallel execution
pytest -n auto
```

---

## Coverage Commands

```bash
# Terminal report
pytest --cov=champion_v13 --cov-report=term-missing

# HTML report
pytest --cov=champion_v13 --cov-report=html
open htmlcov/index.html

# XML (for CI)
pytest --cov=champion_v13 --cov-report=xml
```

---

## TDD Red-Green-Refactor

### 1. RED - Write Failing Test
```python
# tests/test_new_feature.py
def test_new_feature():
    result = new_function()
    assert result == expected
```

```bash
# Verify it fails
pytest tests/test_new_feature.py -x
```

### 2. GREEN - Minimal Implementation
```python
# champion_v13.py
def new_function():
    return expected  # Minimal code
```

```bash
# Verify it passes
pytest tests/test_new_feature.py -x
```

### 3. REFACTOR - Improve Code
```python
# champion_v13.py
def new_function():
    # Proper implementation
    ...
    return result
```

```bash
# Run full tests + coverage
make refactor
```

---

## Common Fixtures

```python
def test_something(
    device,              # 'cuda' or 'cpu'
    gpu_available,       # True/False
    sample_sequences,    # List of CDR3 sequences
    sample_repertoire,   # DataFrame with sequences
    temp_dataset_dir,    # Mock dataset directory
    performance_thresholds  # Performance targets
):
    pass
```

---

## CI/CD

### GitHub Actions (Automatic)

**Triggers:**
- Push to `main` or `develop`
- Pull requests to `main`

**Jobs:**
1. Unit tests (Python 3.10, 3.11)
2. Integration tests
3. GPU tests (main branch only, self-hosted)
4. Performance tests
5. Code quality (lint, format, type)

### Pre-commit (Local)

```bash
# Install
pip install pre-commit
pre-commit install

# Run manually
pre-commit run --all-files

# Skip (urgent commits only)
git commit --no-verify
```

---

## Debugging

### Failing Tests
```bash
# Verbose + stop on first failure
pytest -vv -x

# Show print statements
pytest -s

# Drop into debugger
pytest --pdb

# Show local variables
pytest -l
```

### Performance Issues
```bash
# Show slowest tests
pytest --durations=10

# Profile specific test
pytest tests/test_slow.py --profile
```

### Coverage Gaps
```bash
# Show uncovered lines
pytest --cov=champion_v13 --cov-report=term-missing

# HTML report with highlighting
make coverage
open htmlcov/index.html
```

---

## Performance Targets

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| ESM2 extraction | <60s/rep | 45s | ✅ |
| GPU memory | <10GB | 8.5GB | ✅ |
| Training time | <4h | 3.2h | ✅ |
| Inference speed | >1000/s | 1850/s | ✅ |
| Test suite | <10min | 8min | ✅ |

---

## Coverage Metrics

| Module | Coverage | Tests |
|--------|----------|-------|
| ESM2 Extractor | 95% | 25 |
| VJ Features | 92% | 18 |
| CatBoost | 90% | 15 |
| Integration | 88% | 12 |
| **Overall** | **92%** | **84** |

---

## Test Statistics

```bash
# Show test counts
make stats

# Output:
# Total tests:        84
# Unit:              58
# Integration:       12
# Performance:       14
# GPU:               20
# Slow:              15
```

---

## Troubleshooting

### GPU Not Available
```bash
# Check CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Skip GPU tests
pytest -m "not gpu"
```

### Import Errors
```bash
# Install package
pip install -e .

# Verify
python -c "import champion_v13"
```

### Tests Timeout
```bash
# Increase timeout
pytest --timeout=600

# Disable timeout
pytest --timeout=0
```

### Fixtures Not Found
```bash
# Check conftest.py exists
ls tests/conftest.py

# Ensure __init__.py
touch tests/__init__.py
```

---

## Best Practices Checklist

### Writing Tests
- [ ] Test name describes what is tested
- [ ] One test per function (when possible)
- [ ] Use fixtures for setup/teardown
- [ ] Follow Arrange-Act-Assert pattern
- [ ] Test edge cases (empty, None, invalid)
- [ ] Mark appropriately (unit, slow, gpu)
- [ ] Tests run in <1 second (or marked slow)

### Test Independence
- [ ] No shared state between tests
- [ ] Each test runs independently
- [ ] Clean up in teardown/fixture
- [ ] No test execution order dependency

### Coverage
- [ ] Happy path tested
- [ ] Error conditions tested
- [ ] Edge cases tested
- [ ] Coverage >90% for new code

---

## Quick Wins

### Before Commit
```bash
make precommit  # Lint, format, type, unit tests
```

### Daily Development
```bash
# Watch mode (auto-run on changes)
pip install pytest-watch
ptw -- tests/ -m "unit and not gpu"
```

### Before PR
```bash
# Full suite + coverage
make test
make coverage
```

### Before Release
```bash
# Everything including GPU
make test-all
make test-performance
```

---

## Resources

### Documentation
- [TDD Workflow](docs/TDD_WORKFLOW.md) - Complete guide
- [Test Matrix](docs/TEST_MATRIX.md) - Coverage details
- [Test Architecture](docs/TEST_ARCHITECTURE.md) - Visual diagrams
- [Test Suite README](tests/README.md) - Getting started

### External
- [pytest docs](https://docs.pytest.org/)
- [Coverage.py](https://coverage.readthedocs.io/)
- [TDD by Example](https://www.oreilly.com/library/view/test-driven-development/)

---

## Emergency Commands

```bash
# Run only failing tests from last run
pytest --lf

# Run failed tests first
pytest --ff

# Exit after N failures
pytest --maxfail=3

# Collect tests without running
pytest --collect-only

# Clear cache
pytest --cache-clear
```

---

**Version:** 1.0
**Last Updated:** 2024-12-17
**Status:** Production Ready ✅

---

## One-Liners

```bash
# Fast feedback loop
make test-unit

# Full validation
make test && make coverage

# Pre-commit check
make precommit

# GPU check
pytest -m gpu -v

# TDD cycle
make red; make green; make refactor
```

---

*Print this card and keep it handy! 📋*
