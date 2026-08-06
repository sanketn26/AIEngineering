# Makefile for AIEngineering Poetry Project

.PHONY: help setup install lint format test run clean all docs docs-serve docs-build

# Default target
all: setup lint format test

help:
	@echo "Available targets:"
	@echo "  all         Run setup, lint, format, and test"
	@echo "  setup       Setup the project environment (configure Poetry + install deps)"
	@echo "  install     Install/update dependencies using Poetry"
	@echo "  lint        Run flake8 on src/ and tests/"
	@echo "  format      Format code with black and sort imports with isort"
	@echo "  test        Run pytest on tests/"
	@echo "  run         Run example script"
	@echo "  docs        Install docs deps into .venv-docs"
	@echo "  docs-serve  Live MkDocs server (http://127.0.0.1:8000)"
	@echo "  docs-build  Strict static build into site/"
	@echo "  clean       Remove .venv and Python cache files"

setup:
	@echo "Setting up AIEngineering project..."
	poetry config virtualenvs.in-project true
	@# Prefer 3.11–3.13; core modules are stdlib-only
	@command -v python3.11 >/dev/null && poetry env use python3.11 || true
	poetry install --with dev
	@echo "✅ Core + dev installed. For stock/data track: poetry install -E track-data"
	@echo "✅ Run tests: make test"

install:
	@echo "Installing/updating dependencies..."
	poetry install --with dev

lint:
	@echo "Running linting with flake8..."
	poetry run flake8 src/ tests/

format:
	@echo "Formatting code with black and isort..."
	poetry run black src/ tests/
	poetry run isort src/ tests/

test:
	@echo "Running tests with pytest..."
	poetry run pytest tests/ -v

run:
	@echo "Running example script..."
	poetry run python src/example.py

docs:
	@echo "Installing documentation dependencies..."
	python3 -m venv .venv-docs
	.venv-docs/bin/pip install -U pip
	.venv-docs/bin/pip install -r requirements-docs.txt
	@echo "✅ Docs env ready (.venv-docs). Run: make docs-serve"

docs-serve: docs
	.venv-docs/bin/mkdocs serve

docs-build: docs
	.venv-docs/bin/mkdocs build --strict
	@echo "✅ Built site/ for GitHub Pages"

clean:
	@echo "Cleaning up cache files and virtual environment..."
	rm -rf .venv .venv-docs site __pycache__ src/__pycache__ tests/__pycache__ .pytest_cache
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "✅ Cleanup complete!"
