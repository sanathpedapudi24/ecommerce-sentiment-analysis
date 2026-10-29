"""Customer Sentiment Analysis & Reporting package."""

import os

# Figures are only ever saved to PNG, so default to the headless backend (no Tk needed for the
# CLI, tests, or the API server). Jupyter sets its own inline backend, which takes precedence.
os.environ.setdefault("MPLBACKEND", "Agg")

__version__ = "1.0.0"
