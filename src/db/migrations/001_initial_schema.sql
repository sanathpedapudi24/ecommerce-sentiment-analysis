-- Migration 001: Initial schema creation
-- Mirrors src/db/schema.sql

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

CREATE INDEX IF NOT EXISTS idx_pred_review  ON predictions(review_id);
CREATE INDEX IF NOT EXISTS idx_pred_version ON predictions(model_version);
CREATE INDEX IF NOT EXISTS idx_rev_dept     ON reviews(department_name);
CREATE UNIQUE INDEX IF NOT EXISTS idx_one_active ON model_runs(is_active) WHERE is_active = 1;
