# Makefile for AIEngineering Poetry Project

.PHONY: help setup install lint format test run clean all

# Default target
all: setup lint format test

help:
	@echo "Available targets:"
	@echo "  all       Run setup, lint, format, and test"
	@echo "  setup     Setup the project environment (configure Poetry + install deps)"
	@echo "  install   Install/update dependencies using Poetry"
	@echo "  lint      Run flake8 on src/ and tests/"
	@echo "  format    Format code with black and sort imports with isort"
	@echo "  test      Run pytest on tests/"
	@echo "  run       Run example script"
	@echo "  clean     Remove .venv and Python cache files"

setup:
	@echo "Setting up AIEngineering project..."
	poetry config virtualenvs.in-project true
	poetry install
	@echo "✅ Project setup complete! Virtual environment created at .venv/"

install:
	@echo "Installing/updating dependencies..."
	poetry install

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

clean:
	@echo "Cleaning up cache files and virtual environment..."
	rm -rf .venv __pycache__ src/__pycache__ tests/__pycache__ .pytest_cache
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "✅ Cleanup complete!"
