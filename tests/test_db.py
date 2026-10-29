"""Unit tests for SQLite schema, CHECK constraints, idempotency, and SQL views."""

import pandas as pd
import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.exc import IntegrityError

from src.db.repository import (
    ingest_reviews,
    init_db,
    register_model_run,
)


@pytest.fixture
def test_engine(tmp_path):
    """Create an isolated in-memory or temporary SQLite database."""
    db_file = tmp_path / "test_sentiment.db"
    engine = create_engine(
        f"sqlite:///{db_file.as_posix()}", connect_args={"check_same_thread": False}
    )
    init_db(engine)
    return engine


def test_schema_creates_tables(test_engine):
    """Verify tables exist in the initialized database."""
    with test_engine.connect() as conn:
        tables = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'")).fetchall()
        table_names = [t[0] for t in tables]
        for expected in ["reviews", "model_runs", "predictions", "term_stats"]:
            assert expected in table_names


def test_check_constraints_reject_bad_rating(test_engine):
    """Verify CHECK constraint rejects Rating outside 1-5."""
    with pytest.raises(IntegrityError), test_engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO reviews (review_text, rating, true_label)
                VALUES ('Bad rating test', 6, 'positive')
            """)
        )


def test_check_constraints_reject_bad_label(test_engine):
    """Verify CHECK constraint rejects invalid true_label."""
    with pytest.raises(IntegrityError), test_engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO reviews (review_text, rating, true_label)
                VALUES ('Bad label test', 4, 'super_happy')
            """)
        )


def test_idempotent_ingestion(test_engine):
    """Verify re-ingesting identical data does not create duplicate rows."""
    df = pd.DataFrame(
        [
            {
                "Clothing ID": 100,
                "Age": 30,
                "Title": "Nice item",
                "Review Text": "Lovely skirt, great quality",
                "Rating": 5,
                "true_label": "positive",
                "Recommended IND": 1,
                "Positive Feedback Count": 2,
                "Division Name": "General",
                "Department Name": "Bottoms",
                "Class Name": "Skirts",
                "split": "train",
            }
        ]
    )

    inserted1, skipped1 = ingest_reviews(df, test_engine)
    assert inserted1 == 1
    assert skipped1 == 0

    inserted2, skipped2 = ingest_reviews(df, test_engine)
    assert inserted2 == 0
    assert skipped2 == 1

    with test_engine.connect() as conn:
        count = conn.execute(text("SELECT COUNT(*) FROM reviews")).scalar()
        assert count == 1


def test_only_one_active_model(test_engine):
    """Verify uniqueness of is_active = 1 in model_runs."""
    register_model_run(
        model_version="v1.0.0",
        algorithm="LogisticRegression",
        params={},
        metrics={"test_accuracy": 0.82, "test_macro_f1": 0.65},
        data_hash="abc",
        is_active=1,
        engine=test_engine,
    )
    register_model_run(
        model_version="v2.0.0",
        algorithm="RandomForest",
        params={},
        metrics={"test_accuracy": 0.83, "test_macro_f1": 0.66},
        data_hash="abc",
        is_active=1,
        engine=test_engine,
    )

    with test_engine.connect() as conn:
        active_models = conn.execute(
            text("SELECT model_version FROM model_runs WHERE is_active = 1")
        ).fetchall()
        assert len(active_models) == 1
        assert active_models[0][0] == "v2.0.0"


def test_views_return_expected_columns(test_engine):
    """Verify views return required columns."""
    with test_engine.connect() as conn:
        overview_cols = [
            c[1] for c in conn.execute(text("PRAGMA table_info(vw_sentiment_overview)")).fetchall()
        ]
        assert set(overview_cols) == {"label", "review_count", "pct"}

        dept_cols = [
            c[1]
            for c in conn.execute(text("PRAGMA table_info(vw_sentiment_by_department)")).fetchall()
        ]
        assert set(dept_cols) == {
            "department_name",
            "class_name",
            "predicted_label",
            "review_count",
            "avg_rating",
        }
