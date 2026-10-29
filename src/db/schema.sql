-- Customer Sentiment Analysis & Reporting SQL Schema (SQLite / ANSI SQL)

CREATE TABLE IF NOT EXISTS reviews (
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
    content_hash            TEXT UNIQUE,
    ingested_at             TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS model_runs (
    model_version  TEXT PRIMARY KEY,
    algorithm      TEXT NOT NULL,
    params_json    TEXT NOT NULL,
    metrics_json   TEXT NOT NULL,
    data_hash      TEXT NOT NULL,
    git_commit     TEXT,
    trained_at     TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    is_active      INTEGER NOT NULL DEFAULT 0 CHECK (is_active IN (0,1))
);

CREATE TABLE IF NOT EXISTS predictions (
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

CREATE TABLE IF NOT EXISTS term_stats (
    stat_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    model_version  TEXT NOT NULL REFERENCES model_runs(model_version),
    label          TEXT NOT NULL CHECK (label IN ('negative','neutral','positive')),
    term           TEXT NOT NULL,
    weight         REAL NOT NULL,
    rank           INTEGER NOT NULL
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_pred_review  ON predictions(review_id);
CREATE INDEX IF NOT EXISTS idx_pred_version ON predictions(model_version);
CREATE INDEX IF NOT EXISTS idx_rev_dept     ON reviews(department_name);
CREATE UNIQUE INDEX IF NOT EXISTS idx_one_active ON model_runs(is_active) WHERE is_active = 1;

-- Views for Dashboard and Reporting
DROP VIEW IF EXISTS vw_sentiment_overview;
CREATE VIEW vw_sentiment_overview AS
SELECT 
    true_label AS label,
    COUNT(*) AS review_count,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM reviews), 2) AS pct
FROM reviews
GROUP BY true_label;

DROP VIEW IF EXISTS vw_sentiment_by_department;
CREATE VIEW vw_sentiment_by_department AS
SELECT 
    COALESCE(r.department_name, 'Unknown') AS department_name,
    COALESCE(r.class_name, 'Unknown') AS class_name,
    COALESCE(p.predicted_label, r.true_label) AS predicted_label,
    COUNT(*) AS review_count,
    ROUND(AVG(r.rating), 2) AS avg_rating
FROM reviews r
LEFT JOIN predictions p ON r.review_id = p.review_id AND p.source = 'batch'
    AND p.model_version = (SELECT model_version FROM model_runs WHERE is_active = 1)
GROUP BY r.department_name, r.class_name, COALESCE(p.predicted_label, r.true_label);

DROP VIEW IF EXISTS vw_rating_vs_predicted;
CREATE VIEW vw_rating_vs_predicted AS
SELECT 
    r.rating,
    COALESCE(p.predicted_label, r.true_label) AS predicted_label,
    COUNT(*) AS review_count
FROM reviews r
LEFT JOIN predictions p ON r.review_id = p.review_id AND p.source = 'batch'
    AND p.model_version = (SELECT model_version FROM model_runs WHERE is_active = 1)
GROUP BY r.rating, COALESCE(p.predicted_label, r.true_label);

DROP VIEW IF EXISTS vw_negative_reviews;
CREATE VIEW vw_negative_reviews AS
SELECT 
    r.review_id,
    COALESCE(r.department_name, 'Unknown') AS department_name,
    COALESCE(r.class_name, 'Unknown') AS class_name,
    r.title,
    r.review_text,
    COALESCE(p.confidence, 1.0) AS confidence
FROM reviews r
JOIN predictions p ON r.review_id = p.review_id AND p.source = 'batch'
    AND p.model_version = (SELECT model_version FROM model_runs WHERE is_active = 1)
WHERE p.predicted_label = 'negative'
ORDER BY p.confidence ASC;

DROP VIEW IF EXISTS vw_model_performance;
CREATE VIEW vw_model_performance AS
SELECT 
    model_version,
    algorithm,
    COALESCE(ROUND(json_extract(metrics_json, '$.test_accuracy'), 4), 0.0) AS accuracy,
    COALESCE(ROUND(json_extract(metrics_json, '$.test_macro_f1'), 4), 0.0) AS macro_f1,
    trained_at,
    is_active
FROM model_runs;

DROP VIEW IF EXISTS vw_api_usage;
CREATE VIEW vw_api_usage AS
SELECT 
    date(created_at) AS date,
    predicted_label AS label,
    COUNT(*) AS request_count,
    ROUND(AVG(confidence), 4) AS avg_confidence
FROM predictions
WHERE source = 'api'
GROUP BY date(created_at), predicted_label;

DROP VIEW IF EXISTS vw_top_terms;
CREATE VIEW vw_top_terms AS
SELECT 
    ts.label,
    ts.term,
    ts.weight,
    ts.rank
FROM term_stats ts
JOIN model_runs mr ON ts.model_version = mr.model_version
WHERE mr.is_active = 1
ORDER BY ts.label, ts.rank ASC;
