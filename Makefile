# Makefile for MOEA/D ARM Project

.PHONY: help install install-dev test validate clean lint format check

# Default target
help:
	@echo "Available targets:"
	@echo "  make install       - Install production dependencies"
	@echo "  make install-dev   - Install all dependencies (dev + prod)"
	@echo "  make validate      - Run quick validation tests"
	@echo "  make test          - Run full test suite"
	@echo "  make lint          - Check code quality (radon, ruff)"
	@echo "  make format        - Auto-format code (black)"
	@echo "  make check         - Run all checks (validate + lint + test)"
	@echo "  make clean         - Remove generated files"

# Install production dependencies
install:
	pip install numpy pandas pymoo matplotlib seaborn

# Install all dependencies (including dev tools)
install-dev:
	pip install -r requirements.txt

# Quick validation of new architecture
validate:
	python validate_refactoring.py

# Run full test suite (when available)
test:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term

# Lint code
lint:
	@echo "Running Radon complexity check..."
	radon cc src/ -a -nb
	@echo "\nRunning Radon maintainability index..."
	radon mi src/ -s
	@echo "\nRunning Ruff linter..."
	ruff check src/

# Format code
format:
	black src/ tests/ --line-length 100

# Type checking
typecheck:
	mypy src/ --ignore-missing-imports

# Run all checks
check: validate lint test

# Clean generated files
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	rm -rf .pytest_cache .coverage htmlcov .mypy_cache .ruff_cache
	@echo "Cleaned generated files"

# Run legacy system
run-legacy:
	python main.py

# Run new system (when available)
run-new:
	python main_v2.py

# Generate documentation
docs:
	@echo "Documentation targets:"
	@echo "  README.md      - Project overview"
	@echo "  MIGRATION.md   - Refactoring roadmap"
	@echo "  .github/copilot-instructions.md - AI agent guide"
