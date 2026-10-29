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

# Every target runs inside the uv-managed .venv
function Py { uv run python @args }

switch ($Target) {
    "setup" {
        uv sync
        Py -m nltk.downloader stopwords wordnet punkt punkt_tab
    }
    "data" {
        Py -m src.cli data
    }
    "eda" {
        Py -m src.cli eda
    }
    "train" {
        if ($Version) {
            Py -m src.cli train --version $Version
        } else {
            Py -m src.cli train
        }
    }
    "evaluate" {
        Py -m src.cli evaluate
    }
    "promote" {
        if (-not $Version) {
            Write-Error "Please specify Version: .\make.ps1 promote v1.0.0"
            exit 1
        }
        Py -m src.cli promote --version $Version
    }
    "batch-score" {
        Py -m src.cli batch-score
    }
    "export" {
        Py -m src.cli export
    }
    "serve" {
        Py -m src.cli serve
    }
    "test" {
        Py -m pytest --cov=src --cov-fail-under=70 tests/
    }
    "lint" {
        Py -m ruff check src tests
        Py -m ruff format --check src tests
    }
    "format" {
        Py -m ruff format src tests
        Py -m ruff check --fix src tests
    }
    "all" {
        Py -m src.cli all
    }
    "clean" {
        Py -c "import shutil, os; [shutil.rmtree(p, ignore_errors=True) for p in ['.pytest_cache', '.ruff_cache', 'htmlcov']]; os.remove('.coverage') if os.path.exists('.coverage') else None"
    }
    default {
        Write-Host "Available targets:"
        Write-Host "  .\make.ps1 setup        - Create uv environment & NLTK datasets"
        Write-Host "  .\make.ps1 data         - Ingest, clean, split raw data to SQLite"
        Write-Host "  .\make.ps1 eda          - Generate EDA charts"
        Write-Host "  .\make.ps1 train        - Train baseline and candidate models"
        Write-Host "  .\make.ps1 evaluate     - Evaluate active model"
        Write-Host "  .\make.ps1 promote <ver>- Promote model version to active"
        Write-Host "  .\make.ps1 batch-score  - Score all reviews in DB"
        Write-Host "  .\make.ps1 export       - Export views to reports/exports/*.csv"
        Write-Host "  .\make.ps1 serve        - Run Flask API server"
        Write-Host "  .\make.ps1 test         - Run pytest suite with coverage gate"
        Write-Host "  .\make.ps1 lint         - Run ruff lint and format checks"
        Write-Host "  .\make.ps1 format       - Format code with ruff"
        Write-Host "  .\make.ps1 all          - Run complete pipeline"
    }
}
