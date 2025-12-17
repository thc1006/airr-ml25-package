# Makefile for Champion V13 Testing

.PHONY: help test test-unit test-integration test-performance test-gpu coverage clean install

help:
	@echo "Champion V13 Test Suite"
	@echo "======================="
	@echo ""
	@echo "Available targets:"
	@echo "  make test              - Run all tests (unit + integration, no GPU)"
	@echo "  make test-unit         - Run unit tests only (fast)"
	@echo "  make test-integration  - Run integration tests"
	@echo "  make test-performance  - Run performance benchmarks"
	@echo "  make test-gpu          - Run GPU tests (requires CUDA)"
	@echo "  make test-all          - Run complete test suite (including GPU)"
	@echo "  make coverage          - Generate coverage report"
	@echo "  make clean             - Clean test artifacts"
	@echo "  make install           - Install dependencies"
	@echo ""

# Install dependencies
install:
	pip install -r requirements.txt
	pip install pytest pytest-cov pytest-timeout pytest-xdist

# Run all non-GPU tests
test:
	pytest tests/ -m "not gpu and not slow" -v

# Run unit tests only (fast feedback)
test-unit:
	pytest tests/ -m "unit and not gpu" -v -x

# Run integration tests
test-integration:
	pytest tests/ -m "integration and not gpu" -v

# Run performance benchmarks
test-performance:
	pytest tests/ -m "performance and not gpu" --durations=0 -v

# Run GPU tests
test-gpu:
	@echo "Checking GPU availability..."
	@python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'" || \
		(echo "ERROR: CUDA not available. Skipping GPU tests." && exit 1)
	pytest tests/ -m "gpu" -v

# Run complete test suite
test-all:
	pytest tests/ -v

# Generate coverage report
coverage:
	pytest tests/ -m "not gpu and not slow" \
		--cov=champion_v13 \
		--cov=champion_v13_esm2 \
		--cov-report=html \
		--cov-report=term-missing \
		--cov-report=xml
	@echo ""
	@echo "Coverage report generated in htmlcov/index.html"

# Watch mode (requires pytest-watch)
watch:
	@which ptw > /dev/null || (echo "Installing pytest-watch..." && pip install pytest-watch)
	ptw -- tests/ -m "unit and not gpu"

# Parallel execution
test-parallel:
	pytest tests/ -m "unit and not gpu" -n auto -v

# Clean test artifacts
clean:
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -f .coverage coverage.xml
	rm -rf __pycache__
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# TDD workflow helpers
red:
	@echo "RED phase: Run tests (should fail)"
	pytest tests/ -m "unit and not gpu" -x -v --tb=short

green:
	@echo "GREEN phase: Run tests (should pass)"
	pytest tests/ -m "unit and not gpu" -x -v --tb=short

refactor:
	@echo "REFACTOR phase: Run all tests + coverage"
	pytest tests/ -m "not gpu and not slow" --cov=champion_v13 --cov-report=term-missing -v

# Quick feedback (fastest tests only)
quick:
	pytest tests/ -m "unit and not gpu and not slow" -x

# Full CI equivalent
ci:
	@echo "Running CI-equivalent test suite..."
	pytest tests/ -m "not gpu" \
		--cov=champion_v13 \
		--cov-report=xml \
		--cov-report=term-missing \
		-v

# Debug mode (verbose, show prints, stop on first failure)
debug:
	pytest tests/ -m "unit and not gpu" -vv -s -x --pdb

# Show test statistics
stats:
	@echo "Test Statistics"
	@echo "==============="
	@echo ""
	@echo "Total tests:"
	@pytest --collect-only -q tests/ | tail -n 1
	@echo ""
	@echo "By marker:"
	@echo "  Unit:        $$(pytest --collect-only -q -m unit tests/ | tail -n 1)"
	@echo "  Integration: $$(pytest --collect-only -q -m integration tests/ | tail -n 1)"
	@echo "  Performance: $$(pytest --collect-only -q -m performance tests/ | tail -n 1)"
	@echo "  GPU:         $$(pytest --collect-only -q -m gpu tests/ | tail -n 1)"
	@echo "  Slow:        $$(pytest --collect-only -q -m slow tests/ | tail -n 1)"

# Update requirements
freeze:
	pip freeze > requirements-frozen.txt
	@echo "Frozen requirements saved to requirements-frozen.txt"

# Lint and format code
lint:
	@which ruff > /dev/null || (echo "Installing ruff..." && pip install ruff)
	ruff check .

format:
	@which black > /dev/null || (echo "Installing black..." && pip install black)
	@which isort > /dev/null || (echo "Installing isort..." && pip install isort)
	black .
	isort .

# Type checking
typecheck:
	@which mypy > /dev/null || (echo "Installing mypy..." && pip install mypy)
	mypy champion_v13*.py --ignore-missing-imports

# Pre-commit checks
precommit: format lint typecheck test-unit
	@echo ""
	@echo "✓ All pre-commit checks passed!"

# Show coverage summary
cov-summary:
	@coverage report --skip-empty --show-missing 2>/dev/null || \
		echo "No coverage data. Run 'make coverage' first."

# Open coverage report in browser
cov-open:
	@test -f htmlcov/index.html || (echo "No coverage report. Run 'make coverage' first." && exit 1)
	@which xdg-open > /dev/null && xdg-open htmlcov/index.html || \
		which open > /dev/null && open htmlcov/index.html || \
		echo "Coverage report at: htmlcov/index.html"

# Benchmarking (requires pytest-benchmark)
benchmark:
	@which pytest-benchmark > /dev/null || pip install pytest-benchmark
	pytest tests/ -m "performance" --benchmark-only

# Mutation testing (requires mutmut)
mutate:
	@which mutmut > /dev/null || pip install mutmut
	mutmut run

# Property-based testing only
property:
	pytest tests/ -k "property" -v
