# TDD Workflow for Champion V13

## Red-Green-Refactor Cycle

### Overview

Test-Driven Development (TDD) follows a strict cycle:

1. **RED**: Write a failing test that defines desired functionality
2. **GREEN**: Write minimal code to make the test pass
3. **REFACTOR**: Improve code quality while keeping tests green

### Workflow Steps

#### 1. RED Phase: Write Failing Tests

```bash
# Create test file first
vim tests/test_new_feature.py
```

```python
# Example: Testing new ESM2 caching feature
def test_esm2_cache_saves_embeddings(temp_dir, device):
    """Test ESM2 cache saves embeddings correctly."""
    from champion_v13_esm2 import ESM2FeatureExtractor

    extractor = ESM2FeatureExtractor(device=device, cache_dir=temp_dir)
    sequence = 'CASSLAPGATNEKLFF'

    # First extraction
    embedding1 = extractor.extract_sequence_embedding(sequence)

    # Check cache file exists
    cache_files = list(temp_dir.glob("*.pkl"))
    assert len(cache_files) == 1, "Cache file not created"

    # Second extraction should use cache
    embedding2 = extractor.extract_sequence_embedding(sequence)
    np.testing.assert_array_equal(embedding1, embedding2)
```

Run test to verify it fails:

```bash
pytest tests/test_new_feature.py::test_esm2_cache_saves_embeddings -v
```

Expected output:
```
FAILED - ImportError: cannot import name 'ESM2FeatureExtractor'
```

#### 2. GREEN Phase: Minimal Implementation

Write just enough code to make the test pass:

```python
# champion_v13_esm2.py

class ESM2FeatureExtractor:
    def __init__(self, device='cuda', cache_dir=None):
        self.device = device
        self.cache_dir = cache_dir or Path('./cache')
        self.cache_dir.mkdir(exist_ok=True)

    def extract_sequence_embedding(self, sequence):
        # Check cache first
        cache_file = self.cache_dir / f"{hash(sequence)}.pkl"

        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                return pickle.load(f)

        # Extract embedding (simplified)
        embedding = np.random.randn(1280)  # Placeholder

        # Save to cache
        with open(cache_file, 'wb') as f:
            pickle.dump(embedding, f)

        return embedding
```

Run test again:

```bash
pytest tests/test_new_feature.py::test_esm2_cache_saves_embeddings -v
```

Expected output:
```
PASSED
```

#### 3. REFACTOR Phase: Improve Code Quality

Now improve the implementation:

```python
# champion_v13_esm2.py

class ESM2FeatureExtractor:
    def __init__(self, device='cuda', cache_dir=None):
        self.device = device
        self.cache_dir = cache_dir or Path('./cache')
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        self._cache = {}  # In-memory cache

    def _get_cache_key(self, sequence: str) -> str:
        """Generate cache key for sequence."""
        return hashlib.md5(sequence.encode()).hexdigest()

    def extract_sequence_embedding(self, sequence: str) -> np.ndarray:
        """Extract embedding with two-level caching."""
        cache_key = self._get_cache_key(sequence)

        # Check in-memory cache
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Check disk cache
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                embedding = pickle.load(f)
                self._cache[cache_key] = embedding  # Populate in-memory
                return embedding

        # Extract embedding (real implementation)
        embedding = self._extract_from_model(sequence)

        # Save to both caches
        self._cache[cache_key] = embedding
        with open(cache_file, 'wb') as f:
            pickle.dump(embedding, f)

        return embedding
```

Run all tests to ensure nothing broke:

```bash
pytest tests/ -v
```

---

## TDD Best Practices

### 1. Test Granularity

**DO**: Write focused, single-purpose tests

```python
# GOOD: Focused test
def test_vj_pair_frequency_is_normalized():
    """Test VJ pair frequencies sum to 1.0."""
    features = extract_vj_pairs(sample_df)
    total = sum(v for k, v in features.items() if k.startswith('vj_pair_'))
    assert abs(total - 1.0) < 1e-6

# GOOD: Another focused test
def test_vj_pair_handles_missing_v_call():
    """Test VJ extraction handles missing V calls."""
    df = pd.DataFrame({'v_call': [None, 'TRBV20-1'], 'j_call': ['TRBJ2-7', 'TRBJ2-7']})
    features = extract_vj_pairs(df)
    assert 'vj_pair_UNK_TRBJ2-7' in features
```

**DON'T**: Write multi-purpose tests

```python
# BAD: Tests too many things
def test_vj_features():
    """Test VJ features."""
    features = extract_vj_pairs(sample_df)
    assert len(features) > 0  # What does this test?
    assert 'vj_pair_TRBV20-1_TRBJ2-7' in features  # And this?
    assert sum(features.values()) == 1.0  # And this?
    # Too much in one test!
```

### 2. Test Independence

**DO**: Each test should be independent

```python
# GOOD: Independent tests
@pytest.fixture
def fresh_extractor():
    return ESM2FeatureExtractor()

def test_extraction_1(fresh_extractor):
    result = fresh_extractor.extract('SEQ1')
    assert result.shape == (1280,)

def test_extraction_2(fresh_extractor):
    result = fresh_extractor.extract('SEQ2')
    assert result.shape == (1280,)
```

**DON'T**: Tests that depend on execution order

```python
# BAD: Test order dependency
extractor = None

def test_create_extractor():
    global extractor
    extractor = ESM2FeatureExtractor()
    assert extractor is not None

def test_use_extractor():  # Depends on test_create_extractor!
    result = extractor.extract('SEQ')
    assert result is not None
```

### 3. Test Coverage

**DO**: Test edge cases and error conditions

```python
def test_empty_repertoire_raises_error():
    """Test empty repertoire raises informative error."""
    with pytest.raises(ValueError, match="empty repertoire"):
        extract_features(pd.DataFrame())

def test_invalid_vj_calls_handled_gracefully():
    """Test invalid V/J calls use 'UNK' placeholder."""
    df = pd.DataFrame({'v_call': ['INVALID'], 'j_call': ['ALSO_INVALID']})
    features = extract_vj_pairs(df)
    assert 'vj_pair_UNK_UNK' in features
```

**DON'T**: Only test happy paths

```python
# BAD: Missing edge cases
def test_extract_features():
    """Test feature extraction."""
    features = extract_features(valid_df)
    assert len(features) > 0
    # What about empty df? Invalid data? Missing columns?
```

### 4. Descriptive Test Names

**DO**: Use descriptive names

```python
def test_esm2_extraction_returns_correct_shape():
def test_vj_pairs_normalized_to_sum_one():
def test_catboost_uses_gpu_when_available():
def test_empty_sequences_filtered_before_extraction():
```

**DON'T**: Use vague names

```python
def test_extraction():
def test_features():
def test_model():
def test_it_works():
```

### 5. Arrange-Act-Assert Pattern

```python
def test_ensemble_weights_sum_to_one():
    # ARRANGE: Set up test data
    config = V13Config(
        weight_xgb=0.4,
        weight_lgb=0.4,
        weight_cb=0.2
    )

    # ACT: Execute the code under test
    total_weight = config.weight_xgb + config.weight_lgb + config.weight_cb

    # ASSERT: Verify expected outcome
    assert abs(total_weight - 1.0) < 1e-6
```

---

## Running Tests

### Basic Commands

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_esm2_extractor.py

# Run specific test
pytest tests/test_esm2_extractor.py::test_esm2_model_loading_gpu

# Run tests matching pattern
pytest -k "esm2"

# Run with verbose output
pytest -v

# Run with coverage
pytest --cov=champion_v13 --cov-report=html
```

### Filtering by Markers

```bash
# Run only unit tests (fast)
pytest -m unit

# Skip slow tests
pytest -m "not slow"

# Skip GPU tests (for CPU-only machines)
pytest -m "not gpu"

# Run only integration tests
pytest -m integration

# Run performance benchmarks
pytest -m performance
```

### Parallel Execution

```bash
# Install pytest-xdist
pip install pytest-xdist

# Run tests in parallel (4 workers)
pytest -n 4

# Run tests in parallel (auto-detect CPU count)
pytest -n auto
```

### Watch Mode

```bash
# Install pytest-watch
pip install pytest-watch

# Auto-run tests on file changes
ptw
```

---

## Test Organization

### Directory Structure

```
tests/
├── __init__.py
├── conftest.py                  # Shared fixtures
├── test_esm2_extractor.py      # ESM2 unit tests
├── test_vj_features.py         # VJ features unit tests
├── test_catboost.py            # CatBoost unit tests
├── test_integration_v13.py     # Integration tests
├── test_performance_v13.py     # Performance benchmarks
└── fixtures/                   # Test data
    ├── sample_repertoire.tsv
    └── sample_metadata.csv
```

### Fixture Hierarchy

```python
# conftest.py

@pytest.fixture(scope="session")  # Once per test session
def gpu_available():
    return torch.cuda.is_available()

@pytest.fixture(scope="module")  # Once per test module
def esm2_model():
    return ESM2FeatureExtractor()

@pytest.fixture(scope="function")  # Once per test (default)
def temp_dataset():
    dataset = create_temp_dataset()
    yield dataset
    cleanup_dataset(dataset)
```

---

## Continuous Integration

### GitHub Actions Workflow

See `.github/workflows/test.yml` for the complete CI/CD pipeline.

Key features:
- Automated testing on push/PR
- Matrix testing (Python 3.10, 3.11)
- GPU testing support
- Coverage reporting
- Test result annotations

### Pre-commit Hooks

```bash
# Install pre-commit
pip install pre-commit

# Setup hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

---

## Debugging Failing Tests

### 1. Use pytest's debugging features

```bash
# Show local variables on failure
pytest -l

# Drop into debugger on failure
pytest --pdb

# Drop into debugger on first failure
pytest -x --pdb

# Show print statements
pytest -s
```

### 2. Isolate the failing test

```bash
# Run only the failing test
pytest tests/test_file.py::test_name -v

# Run with maximum verbosity
pytest tests/test_file.py::test_name -vv
```

### 3. Add temporary debugging

```python
def test_something():
    result = function_under_test()

    # Temporary debugging
    import pdb; pdb.set_trace()  # Breakpoint
    print(f"DEBUG: result = {result}")  # Print debugging

    assert result == expected
```

---

## Coverage Targets

### Current Coverage

```bash
# Generate coverage report
pytest --cov=champion_v13 --cov-report=term-missing

# Generate HTML report
pytest --cov=champion_v13 --cov-report=html
open htmlcov/index.html
```

### Coverage Goals

- **Overall**: >90% line coverage
- **Critical paths**: 100% coverage
  - Feature extraction
  - Model training
  - Prediction pipeline
  - Submission generation

### Coverage Exclusions

```python
# pragma: no cover - for debugging code
if DEBUG:  # pragma: no cover
    print(f"Debug: {value}")

# pragma: no cover - for abstract methods
def abstract_method():  # pragma: no cover
    raise NotImplementedError
```

---

## Advanced Testing Techniques

### Property-Based Testing

```python
from hypothesis import given, strategies as st

@given(
    sequences=st.lists(
        st.text(alphabet='ACDEFGHIKLMNPQRSTVWY', min_size=5, max_size=20),
        min_size=1,
        max_size=100
    )
)
def test_vj_frequencies_always_positive(sequences):
    """Property: VJ frequencies are always >= 0."""
    df = pd.DataFrame({'v_call': ['TRBV20-1'] * len(sequences), 'j_call': ['TRBJ2-7'] * len(sequences)})
    features = extract_vj_pairs(df)

    for value in features.values():
        assert value >= 0
```

### Mutation Testing

```bash
# Install mutmut
pip install mutmut

# Run mutation testing
mutmut run

# Show results
mutmut results
```

### Snapshot Testing

```python
import pytest

def test_submission_format_matches_snapshot(snapshot):
    """Test submission format hasn't changed."""
    submission = generate_submission()
    snapshot.assert_match(submission.to_csv(index=False), 'submission.csv')
```

---

## Test Metrics

### Execution Time

```bash
# Show slowest 10 tests
pytest --durations=10

# Show all test durations
pytest --durations=0
```

### Test Distribution

```bash
# Count tests by marker
pytest --collect-only -q | grep "test session"
```

Current distribution:
- Unit tests: ~80 tests
- Integration tests: ~15 tests
- Performance tests: ~10 tests
- GPU tests: ~20 tests

---

## Troubleshooting

### Common Issues

#### GPU tests failing on CPU-only machine

```bash
# Skip GPU tests
pytest -m "not gpu"
```

#### Tests timing out

```bash
# Increase timeout
pytest --timeout=600
```

#### Import errors

```bash
# Ensure package is installed in editable mode
pip install -e .
```

#### Fixture not found

```bash
# Check conftest.py is in correct location
# Ensure __init__.py exists in tests/
```

---

## Quick Reference

### Essential Commands

```bash
# Fast feedback loop (unit tests only)
pytest -m unit -x

# Full test suite
pytest

# CI-equivalent run
pytest --cov=champion_v13 --cov-report=html -m "not slow"

# Performance benchmarks
pytest -m performance --durations=0

# Update snapshots
pytest --snapshot-update
```

### Test Writing Checklist

- [ ] Test is focused and tests one thing
- [ ] Test is independent (no shared state)
- [ ] Test name describes what is tested
- [ ] Test includes arrange-act-assert sections
- [ ] Test includes edge cases
- [ ] Test includes error cases
- [ ] Test has appropriate marker (unit/integration/slow/gpu)
- [ ] Test runs in <1 second (or marked as slow)
- [ ] Test has clear assertion messages

---

## Resources

- [pytest documentation](https://docs.pytest.org/)
- [TDD by Example - Kent Beck](https://www.oreilly.com/library/view/test-driven-development/0321146530/)
- [Python Testing with pytest](https://pragprog.com/titles/bopytest/python-testing-with-pytest/)
- [Hypothesis documentation](https://hypothesis.readthedocs.io/)

---

*Last Updated: 2024-12-17*
*Champion V13 TDD Framework v1.0*
