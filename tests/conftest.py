"""Pytest configuration and environment fixtures."""

import os
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest

# Add project root to sys.path so 'import src...' works across all tests
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Point src.config at a throwaway SQLite DB before it is imported, so test runs never write
# into data/sentiment.db (whose views feed the Power BI exports).
os.environ.pop("DATABASE_URL", None)
os.environ["DB_PATH"] = str(Path(tempfile.mkdtemp()) / "test_sentiment.db")


@pytest.fixture(scope="session")
def seeded_db(tmp_path_factory):
    """Ingest a small sample covering every label and split, and register v1.0.0 as active.

    Lets DB-backed tests run without the raw Kaggle CSV (which is gitignored).
    """
    from src.data.clean import clean_data
    from src.data.split import make_stratified_split
    from src.db.repository import ingest_reviews, register_model_run

    texts = {
        5: "Love this dress, fits perfectly and the fabric is soft",
        3: "It is okay, nothing special but decent for the price",
        1: "Terrible quality, it ripped after one wash and I returned it",
    }
    rows = [
        {
            "Clothing ID": i,
            "Age": 25 + i % 40,
            "Title": None,
            "Review Text": texts[rating],
            "Rating": rating,
            "Recommended IND": int(rating > 3),
            "Positive Feedback Count": 0,
            "Division Name": "General",
            "Department Name": "Dresses",
            "Class Name": "Dresses",
        }
        for i, rating in enumerate([5, 3, 1] * 20)
    ]
    df, _ = clean_data(pd.DataFrame(rows))
    df, _ = make_stratified_split(df, save_path=tmp_path_factory.mktemp("split") / "ids.json")
    ingest_reviews(df)
    register_model_run("v1.0.0", "Tuned MultinomialNB", {}, {}, "sha256:test", is_active=1)
