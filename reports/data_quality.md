# Data Quality & Ingestion Report

## 1. Cleaning Pipeline Summary

| Cleaning Step | Rows Dropped | Rows Remaining |
|---|---|---|
| Raw ingestion | 0 | 23,486 |
| Null / empty review text | 845 | 22,641 |
| Exact duplicate reviews | 1 | 22,640 |
| Invalid ratings (outside 1–5) | 0 | 22,640 |

## 2. Sentiment Label Distribution

| Sentiment Label | Rating Range | Count | Share (%) |
|---|---|---|---|
| **Positive** | 4–5 | 17,447 | 77.06% |
| **Neutral** | 3 | 2,823 | 12.47% |
| **Negative** | 1–2 | 2,370 | 10.47% |

## 3. Missing Value Rates (Final Dataset)

| Column | Null Rate (%) |
|---|---|
| `Clothing ID` | 0.00% |
| `Age` | 0.00% |
| `Title` | 13.10% |
| `Review Text` | 0.00% |
| `Rating` | 0.00% |
| `Recommended IND` | 0.00% |
| `Positive Feedback Count` | 0.00% |
| `Division Name` | 0.06% |
| `Department Name` | 0.06% |
| `Class Name` | 0.06% |
| `true_label` | 0.00% |
| `class_id` | 0.00% |
| `input_text` | 0.00% |

## 4. Integrity and Compliance

- **Preservation of Raw Data**: Raw reviews stored unchanged in the SQLite database.
- **Privacy**: No PII retained beyond age, rating, department/class categories.
- **Deduplication Key**: `(Clothing ID, Age, Title, Review Text, Rating)`.
