<#
.SYNOPSIS
    Cross-platform PowerShell automation helper equivalent to Makefile targets.
.EXAMPLE
    .\make.ps1 test
    .\make.ps1 all
#>
param(
    [Parameter(Position=0)]
    [string]$Target = "help",
    [Parameter(Position=1)]
    [string]$Version = ""
)

$Python = "python"

switch ($Target) {
    "setup" {
        & $Python -m pip install --upgrade pip
        & $Python -m pip install -r requirements.txt
        & $Python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('punkt'); nltk.download('punkt_tab')"
    }
    "data" {
        & $Python -m src.cli data
    }
    "eda" {
        & $Python -m src.cli eda
    }
    "train" {
        if ($Version) {
            & $Python -m src.cli train --version $Version
        } else {
            & $Python -m src.cli train
        }
    }
    "evaluate" {
        & $Python -m src.cli evaluate
    }
    "promote" {
        if (-not $Version) {
            Write-Error "Please specify Version: .\make.ps1 promote v1.0.0"
            exit 1
        }
        & $Python -m src.cli promote --version $Version
    }
    "batch-score" {
        & $Python -m src.cli batch-score
    }
    "export" {
        & $Python -m src.cli export
    }
    "serve" {
        & $Python -m src.cli serve
    }
    "test" {
        & $Python -m pytest --cov=src --cov-fail-under=70 tests/
    }
    "lint" {
        & $Python -m ruff check src tests
        & $Python -m ruff format --check src tests
    }
    "format" {
        & $Python -m ruff format src tests
        & $Python -m ruff check --fix src tests
    }
    "all" {
        & $Python -m src.cli all
    }
    "clean" {
        & $Python -c "import shutil, os; [shutil.rmtree(p, ignore_errors=True) for p in ['.pytest_cache', '.ruff_cache', 'htmlcov']]; os.remove('.coverage') if os.path.exists('.coverage') else None"
    }
    default {
        Write-Host "Available targets:"
        Write-Host "  .\make.ps1 setup        - Install dependencies & NLTK datasets"
        Write-Host "  .\make.ps1 data         - Ingest, clean, split raw data to SQLite"
        Write-Host "  .\make.ps1 eda          - Generate EDA charts"
        Write-Host "  .\make.ps1 train        - Train baseline and candidate models"
        Write-Host "  .\make.ps1 evaluate     - Evaluate active model"
        Write-Host "  .\make.ps1 promote <ver>- Promote model version to active"
        Write-Host "  .\make.ps1 batch-score  - Score all reviews in DB"
        Write-Host "  .\make.ps1 export       - Export views to reports/exports/*.csv"
        Write-Host "  .\make.ps1 serve        - Run Flask API server"
        Write-Host "  .\make.ps1 test         - Run pytest suite with coverage gate"
        Write-Host "  .\make.ps1 lint         - Run ruff and black check"
        Write-Host "  .\make.ps1 format       - Format code with black and ruff"
        Write-Host "  .\make.ps1 all          - Run complete pipeline"
    }
}
