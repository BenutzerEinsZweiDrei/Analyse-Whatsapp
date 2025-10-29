.PHONY: help install install-dev test lint format clean run

help:
	@echo "WhatsApp Analyzer - Development Commands"
	@echo ""
	@echo "Available targets:"
	@echo "  install       Install production dependencies"
	@echo "  install-dev   Install development dependencies"
	@echo "  test          Run tests with pytest"
	@echo "  lint          Run linters (ruff, black --check, isort --check)"
	@echo "  format        Auto-format code with black and isort"
	@echo "  clean         Remove build artifacts and caches"
	@echo "  run           Start Streamlit app"

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt
	pip install pytest pytest-cov black ruff isort

test:
	python -m pytest tests/ -v --cov=app --cov-report=term-missing

lint:
	ruff check . --extend-exclude="local_profile_generator.py"
	black --check . --extend-exclude="local_profile_generator.py"
	isort --check-only --profile black . --skip local_profile_generator.py

format:
	black . --extend-exclude="local_profile_generator.py"
	isort . --profile black --skip local_profile_generator.py
	ruff check . --fix --extend-exclude="local_profile_generator.py"

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf build/ dist/ .coverage htmlcov/ coverage.xml

run:
	streamlit run streamlit_app.py
