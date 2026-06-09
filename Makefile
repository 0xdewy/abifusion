.PHONY: help install install-dev test test-coverage lint format format-check typecheck check clean

SRC := abifusion

help:
	@echo "abifusion - Development Commands"
	@echo ""
	@echo "  install      Install package in development mode"
	@echo "  install-dev  Install with development dependencies"
	@echo "  test         Run tests"
	@echo "  test-coverage  Run tests with coverage report"
	@echo "  lint         Run ruff linting checks"
	@echo "  format       Format code with black"
	@echo "  typecheck    Run mypy type checking"
	@echo "  check        format-check + lint + typecheck"
	@echo "  clean        Remove build/test artifacts"

install:
	uv pip install -e .

install-dev:
	uv pip install -e ".[dev]"

test:
	uv run pytest tests/ -v

test-coverage:
	uv run pytest tests/ --cov=$(SRC) --cov-report=html

lint:
	uv run ruff check $(SRC) tests scripts

format:
	uv run black $(SRC) tests scripts

format-check:
	uv run black --check $(SRC) tests scripts

typecheck:
	uv run mypy $(SRC)

check: format-check lint typecheck
	@echo "All checks passed!"

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache .ruff_cache .mypy_cache htmlcov/ .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
