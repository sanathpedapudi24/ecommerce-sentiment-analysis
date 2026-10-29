"""Raw data ingestion and schema validation."""

from pathlib import Path

import pandas as pd

from src.config import RAW_DATA_PATH

EXPECTED_COLUMNS: list[str] = [
    "Clothing ID",
    "Age",
    "Title",
    "Review Text",
    "Rating",
    "Recommended IND",
    "Positive Feedback Count",
    "Division Name",
    "Department Name",
    "Class Name",
]


def validate_schema(df: pd.DataFrame) -> tuple[bool, list[str]]:
    """Validate that the dataframe contains required columns and proper types.

    Returns:
        (is_valid, error_messages)
    """
    errors: list[str] = []
    missing_cols = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")

    if "Rating" in df.columns:
        invalid_ratings = df[~df["Rating"].isna() & ~df["Rating"].isin([1, 2, 3, 4, 5])]
        if len(invalid_ratings) > 0:
            errors.append(f"Found {len(invalid_ratings)} rows with Rating outside [1, 5]")

    if "Review Text" in df.columns:
        null_count = df["Review Text"].isna().sum()
        if null_count > 0:
            errors.append(f"Found {null_count} rows with missing Review Text")

    return (len(errors) == 0, errors)


def load_raw_data(filepath: str | Path | None = None) -> pd.DataFrame:
    """Load the raw CSV dataset and return as a pandas DataFrame."""
    path = Path(filepath) if filepath else RAW_DATA_PATH
    if not path.exists():
        raise FileNotFoundError(f"Raw data file not found at: {path}")

    df = pd.read_csv(path)
    # Drop index column if present (e.g. Unnamed: 0)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    return df


def get_data_summary(df: pd.DataFrame) -> dict[str, int | float | dict]:
    """Generate high-level data summary statistics."""
    summary = {
        "total_rows": len(df),
        "columns": list(df.columns),
        "missing_values": df.isnull().sum().to_dict(),
        "ratings_distribution": df["Rating"].value_counts().to_dict()
        if "Rating" in df.columns
        else {},
    }
    return summary
