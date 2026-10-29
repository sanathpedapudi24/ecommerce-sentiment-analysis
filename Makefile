.PHONY: help setup data eda train evaluate promote batch-score export serve test lint format all clean

PYTHON ?= python

help:
	@echo Available commands:
	@echo   make setup        - Install dependencies and NLTK data
	@echo   make data         - Load, clean, split, and ingest raw data into SQLite
	@echo   make eda          - Generate exploratory data analysis charts
	@echo   make train        - Train baseline and candidate models, select best, evaluate
	@echo   make evaluate     - Evaluate active model and output evaluation metrics
	@echo   make promote      - Promote a model version (e.g. make promote VERSION=v1.0.0)
	@echo   make batch-score  - Score all reviews in the database with active model
	@echo   make export       - Export database views to reports/exports/*.csv
	@echo   make serve        - Run Flask REST API
	@echo   make test         - Run pytest suite with coverage gate (>= 70%)
	@echo   make lint         - Run ruff lint check and black format check
	@echo   make format       - Format code with black and ruff
	@echo   make all          - Run complete pipeline (data -> eda -> train -> batch-score -> export)

setup:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('punkt'); nltk.download('punkt_tab')"

data:
	$(PYTHON) -m src.cli data

eda:
	$(PYTHON) -m src.cli eda

train:
	$(PYTHON) -m src.cli train $(if $(VERSION),--version $(VERSION),)

evaluate:
	$(PYTHON) -m src.cli evaluate

promote:
	$(PYTHON) -m src.cli promote --version $(VERSION)

batch-score:
	$(PYTHON) -m src.cli batch-score

export:
	$(PYTHON) -m src.cli export

serve:
	$(PYTHON) -m src.cli serve

test:
	$(PYTHON) -m pytest --cov=src --cov-fail-under=70 tests/

lint:
	$(PYTHON) -m ruff check src tests
	$(PYTHON) -m ruff format --check src tests

format:
	$(PYTHON) -m ruff format src tests
	$(PYTHON) -m ruff check --fix src tests

all:
	$(PYTHON) -m src.cli all

clean:
	$(PYTHON) -c "import shutil, os; [shutil.rmtree(p, ignore_errors=True) for p in ['.pytest_cache', '.ruff_cache', 'htmlcov']]; os.remove('.coverage') if os.path.exists('.coverage') else None"
