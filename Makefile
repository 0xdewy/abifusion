.PHONY: install install-dev test lint format typecheck clean help

# Default target
help:
	@echo "ABI Reconstructor - Development Commands"
	@echo ""
	@echo "install     Install package in development mode"
	@echo "install-dev Install with development dependencies"
	@echo "test        Run tests"
	@echo "lint        Run linting checks"
	@echo "format      Format code with black"
	@echo "typecheck   Run type checking with mypy"
	@echo "clean       Clean build artifacts"
	@echo ""

# Installation (using uv)
install:
	uv pip install -e .

install-dev:
	uv pip install -e ".[dev]"

# Testing
test:
	uv run pytest tests/ -v

test-coverage:
	uv run pytest tests/ --cov=abi_reconstructor --cov-report=html

# Code quality
lint:
	uv run ruff check src/ tests/ scripts/

format:
	uv run black src/ tests/ scripts/

format-check:
	uv run black --check src/ tests/ scripts/

typecheck:
	uv run mypy src/

# Cleanup
clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache .ruff_cache .mypy_cache
	rm -rf htmlcov/ .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

# Development
dev: install-dev
	@echo "Development environment ready"

# Website (separate)
website-install:
	cd website/backend && pip install -r requirements.txt

website-run:
	cd website/backend && uvicorn main:app --reload

# Training
train:
	@echo "Use: python scripts/train.py or ./scripts/shell/train.sh"

# Quick check
check: format-check lint typecheck
	@echo "All checks passed!"