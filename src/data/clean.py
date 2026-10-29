"""Data cleaning, label derivation, and data quality reporting."""

from pathlib import Path

import pandas as pd

from src.config import DATA_QUALITY_PATH, LABEL_MAP, LABEL_TO_ID


def derive_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Map numerical Rating (1-5) to sentiment label ('negative', 'neutral', 'positive') and class_id."""
    df = df.copy()
    df["true_label"] = df["Rating"].map(LABEL_MAP)
    df["class_id"] = df["true_label"].map(LABEL_TO_ID)
    return df


def build_model_input_text(df: pd.DataFrame) -> pd.Series:
    """Construct model input text by combining Title and Review Text.

    Title is omitted when null or whitespace-only. Handles both raw and DB column names.
    """
    title_col = "Title" if "Title" in df.columns else "title"
    review_col = "Review Text" if "Review Text" in df.columns else "review_text"

    title = (
        df[title_col].fillna("").astype(str).str.strip()
        if title_col in df.columns
        else pd.Series([""] * len(df), index=df.index)
    )
    review = (
        df[review_col].fillna("").astype(str).str.strip()
        if review_col in df.columns
        else pd.Series([""] * len(df), index=df.index)
    )

    combined = title.where(title == "", title + " ") + review
    return combined.str.strip()


def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int | float | dict]]:
    """Apply the cleaning rules specified in DATA.md:

    1. Drop rows with null or whitespace-only review text.
    2. Drop exact duplicate rows on (Clothing ID, Age, Title, Review Text, Rating).
    3. Drop ratings outside 1-5.
    4. Derive labels and input text.
    5. Returns (cleaned_df, quality_metrics)
    """
    df = df.copy()
    initial_count = len(df)
    metrics: dict[str, int | float | dict] = {
        "initial_rows": initial_count,
        "initial_null_reviews": int(df["Review Text"].isna().sum()),
    }

    # Normalize whitespace in Review Text
    if "Review Text" in df.columns:
        df["Review Text"] = (
            df["Review Text"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
        )
        # Mark string representations of empty or 'nan' as null
        df.loc[df["Review Text"].isin(["", "nan", "None"]), "Review Text"] = None

    # Rule 1: Drop rows with null or whitespace-only review text
    df_after_text = df.dropna(subset=["Review Text"]).copy()
    dropped_empty_text = initial_count - len(df_after_text)
    metrics["dropped_empty_text"] = dropped_empty_text
    metrics["rows_after_empty_text_filter"] = len(df_after_text)

    # Rule 2: Drop exact duplicates on (Clothing ID, Age, Title, Review Text, Rating)
    dedup_cols = ["Clothing ID", "Age", "Title", "Review Text", "Rating"]
    existing_dedup_cols = [c for c in dedup_cols if c in df_after_text.columns]
    df_dedup = df_after_text.drop_duplicates(subset=existing_dedup_cols).copy()
    dropped_duplicates = len(df_after_text) - len(df_dedup)
    metrics["dropped_duplicates"] = dropped_duplicates
    metrics["rows_after_dedup"] = len(df_dedup)

    # Rule 3: Drop ratings outside 1-5
    df_valid_ratings = df_dedup[df_dedup["Rating"].isin([1, 2, 3, 4, 5])].copy()
    dropped_invalid_ratings = len(df_dedup) - len(df_valid_ratings)
    metrics["dropped_invalid_ratings"] = dropped_invalid_ratings
    metrics["final_rows"] = len(df_valid_ratings)

    # Rule 4: Derive labels
    df_cleaned = derive_labels(df_valid_ratings)

    # Rule 5: Derive model input text
    df_cleaned["input_text"] = build_model_input_text(df_cleaned)

    # Label distribution & null rates
    metrics["label_distribution"] = df_cleaned["true_label"].value_counts().to_dict()
    metrics["label_pct"] = (
        (df_cleaned["true_label"].value_counts(normalize=True) * 100).round(2).to_dict()
    )
    metrics["null_rates"] = (df_cleaned.isnull().mean() * 100).round(2).to_dict()

    return df_cleaned, metrics


def generate_data_quality_report(
    metrics: dict[str, int | float | dict], output_path: str | Path | None = None
) -> str:
    """Generate reports/data_quality.md markdown documentation."""
    out = Path(output_path) if output_path else DATA_QUALITY_PATH
    out.parent.mkdir(parents=True, exist_ok=True)

    report = [
        "# Data Quality & Ingestion Report",
        "",
        "## 1. Cleaning Pipeline Summary",
        "",
        "| Cleaning Step | Rows Dropped | Rows Remaining |",
        "|---|---|---|",
        f"| Raw ingestion | 0 | {metrics['initial_rows']:,} |",
        f"| Null / empty review text | {metrics['dropped_empty_text']:,} | {metrics['rows_after_empty_text_filter']:,} |",
        f"| Exact duplicate reviews | {metrics['dropped_duplicates']:,} | {metrics['rows_after_dedup']:,} |",
        f"| Invalid ratings (outside 1–5) | {metrics['dropped_invalid_ratings']:,} | {metrics['final_rows']:,} |",
        "",
        "## 2. Sentiment Label Distribution",
        "",
        "| Sentiment Label | Rating Range | Count | Share (%) |",
        "|---|---|---|---|",
    ]

    dist = metrics.get("label_distribution", {})
    pct = metrics.get("label_pct", {})
    rating_ranges = {"negative": "1–2", "neutral": "3", "positive": "4–5"}

    for label in ["positive", "neutral", "negative"]:
        count = dist.get(label, 0)
        percentage = pct.get(label, 0.0)
        rr = rating_ranges.get(label, "")
        report.append(f"| **{label.capitalize()}** | {rr} | {count:,} | {percentage:.2f}% |")

    report.extend(
        [
            "",
            "## 3. Missing Value Rates (Final Dataset)",
            "",
            "| Column | Null Rate (%) |",
            "|---|---|",
        ]
    )

    null_rates = metrics.get("null_rates", {})
    for col, rate in null_rates.items():
        report.append(f"| `{col}` | {rate:.2f}% |")

    report.extend(
        [
            "",
            "## 4. Integrity and Compliance",
            "",
            "- **Preservation of Raw Data**: Raw reviews stored unchanged in the SQLite database.",
            "- **Privacy**: No PII retained beyond age, rating, department/class categories.",
            "- **Deduplication Key**: `(Clothing ID, Age, Title, Review Text, Rating)`.",
            "",
        ]
    )

    content = "\n".join(report)
    out.write_text(content, encoding="utf-8")
    return content
