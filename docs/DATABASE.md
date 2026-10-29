# Database

Engine: **SQLite** (`data/sentiment.db`) via SQLAlchemy. Schema uses ANSI SQL types so it ports to PostgreSQL unchanged.

## Tables

```sql
CREATE TABLE reviews (
    review_id               INTEGER PRIMARY KEY AUTOINCREMENT,
    clothing_id             INTEGER,
    age                     INTEGER CHECK (age BETWEEN 10 AND 110),
    title                   TEXT,
    review_text             TEXT NOT NULL,
    rating                  INTEGER NOT NULL CHECK (rating BETWEEN 1 AND 5),
    true_label              TEXT NOT NULL CHECK (true_label IN ('negative','neutral','positive')),
    recommended             INTEGER CHECK (recommended IN (0,1)),
    positive_feedback_count INTEGER CHECK (positive_feedback_count >= 0),
    division_name           TEXT,
    department_name         TEXT,
    class_name              TEXT,
    split                   TEXT CHECK (split IN ('train','val','test')),
    ingested_at             TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE model_runs (
    model_version  TEXT PRIMARY KEY,
    algorithm      TEXT NOT NULL,
    params_json    TEXT NOT NULL,
    metrics_json   TEXT NOT NULL,
    data_hash      TEXT NOT NULL,
    git_commit     TEXT,
    trained_at     TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    is_active      INTEGER NOT NULL DEFAULT 0 CHECK (is_active IN (0,1))
);

CREATE TABLE predictions (
    prediction_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    review_id        INTEGER REFERENCES reviews(review_id),
    input_text       TEXT NOT NULL,
    predicted_label  TEXT NOT NULL CHECK (predicted_label IN ('negative','neutral','positive')),
    confidence       REAL NOT NULL CHECK (confidence BETWEEN 0 AND 1),
    prob_negative    REAL NOT NULL,
    prob_neutral     REAL NOT NULL,
    prob_positive    REAL NOT NULL,
    model_version    TEXT NOT NULL REFERENCES model_runs(model_version),
    source           TEXT NOT NULL CHECK (source IN ('batch','api')),
    created_at       TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_pred_review  ON predictions(review_id);
CREATE INDEX idx_pred_version ON predictions(model_version);
CREATE INDEX idx_rev_dept     ON reviews(department_name);
CREATE UNIQUE INDEX idx_one_active ON model_runs(is_active) WHERE is_active = 1;
```

`review_id` is NULL for API predictions on free text.

## Views (dashboard sources)

| View | Columns | Use |
|---|---|---|
| `vw_sentiment_overview` | label, review_count, pct | KPI cards, donut |
| `vw_sentiment_by_department` | department_name, class_name, predicted_label, review_count, avg_rating | Category drill-down |
| `vw_rating_vs_predicted` | rating, predicted_label, review_count | Heatmap |
| `vw_negative_reviews` | review_id, department_name, class_name, title, review_text, confidence | Table of reviews needing attention |
| `vw_model_performance` | model_version, algorithm, accuracy, macro_f1, trained_at, is_active | Model monitoring |
| `vw_api_usage` | date, label, request_count, avg_confidence | Monitoring |
| `vw_top_terms` | label, term, weight, rank | Text insights (table written by training job) |

`vw_*` over base tables are plain `CREATE VIEW`; `vw_top_terms` is backed by a table `term_stats(model_version, label, term, weight, rank)`.

## Rules

- Schema created by `src/db/schema.sql`; migrations are numbered files in `src/db/migrations/`.
- All writes go through `src/db/repository.py`; all queries are parameterised.
- Ingestion is idempotent: re-running `make data` does not duplicate rows (deduplicate on content hash).
- Only one `model_runs` row may have `is_active = 1`.
