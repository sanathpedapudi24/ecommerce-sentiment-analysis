"""Unit tests for data cleaning, label mapping, and schema validation."""

import pandas as pd

from src.data.clean import build_model_input_text, clean_data, derive_labels
from src.data.load import validate_schema


def test_derive_labels():
    """Verify rating to sentiment mapping: 1,2 -> negative, 3 -> neutral, 4,5 -> positive."""
    df = pd.DataFrame({"Rating": [1, 2, 3, 4, 5]})
    df_labelled = derive_labels(df)
    expected_labels = ["negative", "negative", "neutral", "positive", "positive"]
    expected_ids = [0, 0, 1, 2, 2]
    assert df_labelled["true_label"].tolist() == expected_labels
    assert df_labelled["class_id"].tolist() == expected_ids


def test_clean_data_drops_empty_and_null_text():
    """Verify rows with null or whitespace-only Review Text are dropped."""
    data = {
        "Clothing ID": [101, 102, 103, 104, 105],
        "Age": [30, 35, 40, 25, 50],
        "Title": ["Good", "Bad", None, "Okay", "Super"],
        "Review Text": ["Loved the fabric", "   ", None, "nan", "Great dress"],
        "Rating": [5, 1, 2, 3, 5],
    }
    df = pd.DataFrame(data)
    cleaned_df, metrics = clean_data(df)
    assert len(cleaned_df) == 2
    assert metrics["dropped_empty_text"] == 3
    assert set(cleaned_df["Clothing ID"]) == {101, 105}


def test_clean_data_removes_exact_duplicates():
    """Verify exact duplicates on deduplication columns are removed."""
    row = {
        "Clothing ID": 200,
        "Age": 28,
        "Title": "Nice",
        "Review Text": "Comfortable and pretty",
        "Rating": 4,
    }
    df = pd.DataFrame([row, row, row])
    cleaned_df, metrics = clean_data(df)
    assert len(cleaned_df) == 1
    assert metrics["dropped_duplicates"] == 2


def test_clean_data_rejects_ratings_outside_one_to_five():
    """Verify rows with ratings outside [1, 5] are dropped."""
    data = {
        "Clothing ID": [301, 302, 303, 304],
        "Age": [30, 30, 30, 30],
        "Title": ["T1", "T2", "T3", "T4"],
        "Review Text": ["Text one", "Text two", "Text three", "Text four"],
        "Rating": [0, 3, 5, 6],
    }
    df = pd.DataFrame(data)
    cleaned_df, metrics = clean_data(df)
    assert len(cleaned_df) == 2
    assert metrics["dropped_invalid_ratings"] == 2
    assert set(cleaned_df["Rating"]) == {3, 5}


def test_build_model_input_text():
    """Verify combination of Title and Review Text."""
    df = pd.DataFrame(
        {
            "Title": ["Gorgeous blouse", None, "   ", "Classic"],
            "Review Text": [
                "Fits nicely",
                "Soft material",
                "Too small",
                "Timeless style",
            ],
        }
    )
    res = build_model_input_text(df).tolist()
    assert res[0] == "Gorgeous blouse Fits nicely"
    assert res[1] == "Soft material"
    assert res[2] == "Too small"
    assert res[3] == "Classic Timeless style"


def test_validate_schema_reports_errors():
    """Verify validate_schema identifies missing columns and bad ratings."""
    df_bad = pd.DataFrame({"Clothing ID": [1], "Rating": [10]})
    is_valid, errors = validate_schema(df_bad)
    assert not is_valid
    assert any("Missing required columns" in e for e in errors)
    assert any("Rating outside [1, 5]" in e for e in errors)
