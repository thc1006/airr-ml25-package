# Champion V13 Test Suite

Comprehensive test-driven development (TDD) framework for the AIRR-ML-25 competition champion model.

## Quick Start

```bash
# Install dependencies
make install

# Run unit tests (fast feedback)
make test-unit

# Run all tests
make test

# Generate coverage report
make coverage
```

## Test Structure

```
tests/
├── README.md                    # This file
├── conftest.py                  # Shared fixtures and configuration
├── test_esm2_extractor.py      # ESM2 feature extractor tests (25 tests)
├── test_vj_features.py         # VJ gene features tests (18 tests)
├── test_catboost.py            # CatBoost integration tests (15 tests)
├── test_integration_v13.py     # End-to-end pipeline tests (12 tests)
├── test_performance_v13.py     # Performance benchmarks (14 tests)
└── fixtures/                   # Test data
    ├── sample_repertoire.tsv
    └── sample_metadata.csv
```

## Test Categories

### Unit Tests (58 tests, ~15 seconds)

Fast, focused tests for individual components.

```bash
# Run all unit tests
pytest -m unit

# Run without GPU tests
pytest -m "unit and not gpu"

# Run specific module
pytest tests/test_esm2_extractor.py -m unit
```

**Coverage:**
- ESM2 Feature Extractor: 95%
- VJ Features: 92%
- CatBoost Integration: 90%

### Integration Tests (12 tests, ~120 seconds)

Multi-component tests for end-to-end workflows.

```bash
# Run integration tests
pytest -m integration

# Skip slow tests
pytest -m "integration and not slow"
```

**Tests:**
- Complete pipeline execution
- Feature compatibility (ESM2 + VJ)
- Submission format validation
- Cross-validation workflow

### Performance Tests (14 tests, ~300 seconds)

Benchmarks for speed and resource usage.

```bash
# Run performance tests
pytest -m performance

# GPU performance tests
pytest -m "performance and gpu"

# Show timing details
pytest -m performance --durations=0
```

**Targets:**
- ESM2 extraction: <60s per repertoire
- GPU memory: <10GB peak
- Training: <4 hours total
- Inference: >1000 predictions/second

### GPU Tests (20 tests, ~45 seconds)

Tests requiring CUDA-enabled GPU.

```bash
# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"

# Run GPU tests
pytest -m gpu
```

## Test Markers

Use markers to filter tests:

| Marker | Count | Purpose |
|--------|-------|---------|
| `unit` | 58 | Fast unit tests |
| `integration` | 12 | Multi-component tests |
| `performance` | 14 | Benchmarks |
| `gpu` | 20 | Requires CUDA |
| `slow` | 15 | Takes >10 seconds |
| `requires_data` | 3 | Needs full dataset |

**Examples:**

```bash
# Fast tests only (no GPU, no slow)
pytest -m "unit and not gpu and not slow"

# All tests except GPU
pytest -m "not gpu"

# Integration + performance
pytest -m "integration or performance"
```

## TDD Workflow

### Red-Green-Refactor Cycle

```bash
# 1. RED: Write failing test
vim tests/test_new_feature.py

# Run to verify it fails
make red

# 2. GREEN: Implement minimal code
vim champion_v13.py

# Run to verify it passes
make green

# 3. REFACTOR: Improve code quality
vim champion_v13.py

# Run full tests + coverage
make refactor
```

See [docs/TDD_WORKFLOW.md](../docs/TDD_WORKFLOW.md) for detailed guidance.

## Coverage

### Generate Reports

```bash
# Terminal report
pytest --cov=champion_v13 --cov-report=term-missing

# HTML report
make coverage
open htmlcov/index.html

# XML report (for CI)
pytest --cov=champion_v13 --cov-report=xml
```

### Current Coverage

```
champion_v13.py          90%  (738/820 lines)
champion_v13_esm2.py     95%  (485/510 lines)
Overall                  92%  (1223/1330 lines)
```

**Target: >90% line coverage**

## Fixtures

### Available Fixtures

| Fixture | Scope | Description |
|---------|-------|-------------|
| `gpu_available` | session | GPU availability check |
| `device` | session | Device string ('cuda' or 'cpu') |
| `sample_sequences` | function | 8 CDR3 sequences |
| `sample_repertoire` | function | Mock repertoire DataFrame |
| `sample_metadata` | function | Mock metadata CSV |
| `temp_dataset_dir` | function | Full mock dataset directory |
| `esm2_extractor` | module | Shared ESM2 instance |
| `performance_thresholds` | function | Performance targets |

### Using Fixtures

```python
def test_something(sample_repertoire, device):
    """Test using fixtures."""
    extractor = ESM2FeatureExtractor(device=device)
    features = extractor.extract_repertoire_features(sample_repertoire)
    assert features.shape == (1280,)
```

## Common Tasks

### Run Specific Tests

```bash
# Single test
pytest tests/test_esm2_extractor.py::test_esm2_model_loading_gpu

# Test class
pytest tests/test_esm2_extractor.py::TestESM2Loading

# Pattern matching
pytest -k "esm2 and not gpu"
```

### Debug Failing Tests

```bash
# Verbose output
pytest -vv

# Show print statements
pytest -s

# Stop on first failure
pytest -x

# Drop into debugger
pytest --pdb

# Show local variables
pytest -l
```

### Parallel Execution

```bash
# Install plugin
pip install pytest-xdist

# Run in parallel (4 workers)
pytest -n 4

# Auto-detect CPU count
pytest -n auto
```

### Watch Mode

```bash
# Install plugin
pip install pytest-watch

# Auto-run on file changes
ptw

# Custom args
ptw -- tests/ -m unit
```

## CI/CD

### GitHub Actions

Tests run automatically on:
- Push to `main` or `develop`
- Pull requests to `main`

**Jobs:**
1. **Unit tests** (Python 3.10, 3.11)
2. **Integration tests** (Python 3.10)
3. **GPU tests** (self-hosted runner, main branch only)
4. **Performance tests** (Python 3.10)
5. **Code quality** (linting, type checking)

See [.github/workflows/test.yml](../.github/workflows/test.yml) for configuration.

### Pre-commit Hooks

```bash
# Install
pip install pre-commit
pre-commit install

# Run manually
pre-commit run --all-files

# Skip for urgent commits
git commit -m "message" --no-verify
```

## Performance Benchmarks

### Current Performance

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| ESM2 extraction | <60s/rep | 45s | ✓ |
| GPU memory | <10GB | 8.5GB | ✓ |
| Training time | <4h | 3.2h | ✓ |
| Inference speed | >1000/s | 1850/s | ✓ |

### Run Benchmarks

```bash
# All performance tests
make test-performance

# GPU benchmarks only
pytest -m "performance and gpu" --durations=0

# Compare with baseline
pytest -m performance --benchmark-compare
```

## Troubleshooting

### GPU Tests Failing

```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Check NVIDIA driver
nvidia-smi

# Skip GPU tests
pytest -m "not gpu"
```

### ImportError

```bash
# Install package in editable mode
pip install -e .

# Verify installation
python -c "import champion_v13; print('OK')"
```

### Timeout Errors

```bash
# Increase timeout (default: 300s)
pytest --timeout=600

# Disable timeout
pytest --timeout=0
```

### Fixtures Not Found

```bash
# Ensure conftest.py exists
ls tests/conftest.py

# Ensure tests/ has __init__.py
touch tests/__init__.py
```

## Advanced Testing

### Property-Based Testing

```bash
# Install Hypothesis
pip install hypothesis

# Run property tests
pytest -k "property"
```

### Mutation Testing

```bash
# Install mutmut
pip install mutmut

# Run mutation testing
make mutate

# View results
mutmut show
```

### Snapshot Testing

```bash
# Install plugin
pip install pytest-snapshot

# Update snapshots
pytest --snapshot-update
```

## Best Practices

### Writing Tests

1. **One assertion per test** (when possible)
2. **Descriptive test names** (`test_esm2_extraction_returns_correct_shape`)
3. **Use fixtures** for setup/teardown
4. **Arrange-Act-Assert** pattern
5. **Test edge cases** (empty, None, invalid)
6. **Mark appropriately** (unit, integration, slow, gpu)

### Test Independence

- Each test should run independently
- No shared state between tests
- Use fixtures, not globals
- Clean up in teardown

### Performance

- Unit tests should be <1 second
- Mark slow tests with `@pytest.mark.slow`
- Use mocks for external dependencies
- Avoid unnecessary setup

## Resources

- [TDD Workflow Guide](../docs/TDD_WORKFLOW.md) - Detailed TDD practices
- [Test Matrix](../docs/TEST_MATRIX.md) - Complete test coverage
- [pytest documentation](https://docs.pytest.org/)
- [Coverage.py docs](https://coverage.readthedocs.io/)

## Statistics

```bash
# Show test counts
make stats

# Show slowest tests
pytest --durations=10

# Show coverage summary
make cov-summary
```

**Current Stats:**
- Total tests: 84
- Total LOC: ~3,500
- Test/Prod ratio: 2.6:1
- Average test time: 5.7s
- Coverage: 92%

---

**Last Updated:** 2024-12-17
**Version:** 1.0
**Maintainer:** AIRR-ML-25 Team
