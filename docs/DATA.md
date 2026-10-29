# Data

## Source

Default dataset: **Women's E-Commerce Clothing Reviews** (public, ~23.5k reviews). Any e-commerce review dataset with free text and a 1–5 rating may be substituted if it provides the columns below.

Place the file at `data/raw/reviews.csv`. Raw data is never modified.

## Expected columns

| Column | Type | Notes |
|---|---|---|
| Clothing ID | int | Product identifier |
| Age | int | Reviewer age |
| Title | str | Optional, may be null |
| Review Text | str | Required, rows with null/empty text are dropped |
| Rating | int | 1–5, required |
| Recommended IND | int | 0/1 |
| Positive Feedback Count | int | ≥ 0 |
| Division Name | str | |
| Department Name | str | |
| Class Name | str | |

## Label definition

| Rating | Label | Class id |
|---|---|---|
| 1–2 | negative | 0 |
| 3 | neutral | 1 |
| 4–5 | positive | 2 |

Model input text = `Title + " " + Review Text` (title omitted when null).

## Cleaning rules

1. Drop rows with null or whitespace-only review text.
2. Drop exact duplicate rows on (Clothing ID, Age, Title, Review Text, Rating).
3. Drop ratings outside 1–5.
4. Keep the original text in the database; cleaned text is produced inside the pipeline only.

## Splits

| Split | Share | Purpose |
|---|---|---|
| Train | 70% | Fitting and cross-validation |
| Validation | 15% | Model selection |
| Test | 15% | Final report, touched once |

Stratified on label, `random_state=42`. Split indices saved to `data/processed/split_ids.json`.

## Class imbalance

Expect positive ≫ neutral/negative. Handle with `class_weight="balanced"` (LR, RF) and macro-F1 as the selection metric. Do not oversample before splitting.

## Data quality report

`make data` writes `reports/data_quality.md`: row counts before/after each cleaning rule, null rates, label distribution.
