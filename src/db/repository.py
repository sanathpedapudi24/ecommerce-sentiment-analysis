"""Database repository layer providing parameterized access to SQLite via SQLAlchemy."""

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from src.config import DATABASE_URL, DB_PATH, EXPORTS_DIR

_ENGINE: Engine | None = None


def get_engine(db_url: str | None = None) -> Engine:
    """Get or create singleton SQLAlchemy Engine."""
    global _ENGINE
    url = db_url or DATABASE_URL
    if _ENGINE is None or db_url is not None:
        # SQLite connection setup
        if url.startswith("sqlite"):
            # Ensure parent directory exists
            DB_PATH.parent.mkdir(parents=True, exist_ok=True)
            engine = create_engine(
                url,
                connect_args={"check_same_thread": False},
                pool_pre_ping=True,
            )
        else:
            engine = create_engine(url, pool_pre_ping=True)
        if db_url is None:
            _ENGINE = engine
        return engine
    return _ENGINE


def init_db(engine: Engine | None = None) -> None:
    """Initialize database tables, indexes, and views from schema.sql."""
    eng = engine or get_engine()
    schema_path = Path(__file__).parent / "schema.sql"
    sql_script = schema_path.read_text(encoding="utf-8")

    # Split by statements
    with eng.begin() as conn:
        for statement in sql_script.split(";"):
            stmt = statement.strip()
            if stmt:
                conn.execute(text(stmt))


def compute_content_hash(row: pd.Series | dict[str, Any]) -> str:
    """Compute SHA-256 fingerprint for idempotency."""
    key = f"{row.get('Clothing ID')}_{row.get('Age')}_{row.get('Title')}_{row.get('Review Text')}_{row.get('Rating')}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def ingest_reviews(df: pd.DataFrame, engine: Engine | None = None) -> tuple[int, int]:
    """Idempotently ingest reviews into the reviews table.

    Returns (inserted_count, skipped_count).
    """
    eng = engine or get_engine()
    init_db(eng)

    df = df.copy()
    if "content_hash" not in df.columns:
        df["content_hash"] = df.apply(compute_content_hash, axis=1)

    records = []
    for _, row in df.iterrows():
        records.append(
            {
                "clothing_id": int(row["Clothing ID"])
                if pd.notna(row.get("Clothing ID"))
                else None,
                "age": int(row["Age"]) if pd.notna(row.get("Age")) else None,
                "title": str(row["Title"]) if pd.notna(row.get("Title")) else None,
                "review_text": str(row["Review Text"]),
                "rating": int(row["Rating"]),
                "true_label": str(row["true_label"]),
                "recommended": int(row["Recommended IND"])
                if pd.notna(row.get("Recommended IND"))
                else None,
                "positive_feedback_count": int(row["Positive Feedback Count"])
                if pd.notna(row.get("Positive Feedback Count"))
                else 0,
                "division_name": str(row["Division Name"])
                if pd.notna(row.get("Division Name"))
                else None,
                "department_name": str(row["Department Name"])
                if pd.notna(row.get("Department Name"))
                else None,
                "class_name": str(row["Class Name"]) if pd.notna(row.get("Class Name")) else None,
                "split": str(row.get("split", "train")),
                "content_hash": str(row["content_hash"]),
            }
        )

    insert_sql = text("""
        INSERT OR IGNORE INTO reviews (
            clothing_id, age, title, review_text, rating, true_label,
            recommended, positive_feedback_count, division_name,
            department_name, class_name, split, content_hash
        ) VALUES (
            :clothing_id, :age, :title, :review_text, :rating, :true_label,
            :recommended, :positive_feedback_count, :division_name,
            :department_name, :class_name, :split, :content_hash
        )
    """)

    with eng.begin() as conn:
        before_count = conn.execute(text("SELECT COUNT(*) FROM reviews")).scalar() or 0
        conn.execute(insert_sql, records)
        after_count = conn.execute(text("SELECT COUNT(*) FROM reviews")).scalar() or 0

    inserted = after_count - before_count
    skipped = len(records) - inserted
    return inserted, skipped


def register_model_run(
    model_version: str,
    algorithm: str,
    params: dict[str, Any],
    metrics: dict[str, Any],
    data_hash: str,
    git_commit: str | None = None,
    is_active: int = 0,
    engine: Engine | None = None,
) -> None:
    """Insert or replace record into model_runs table."""
    eng = engine or get_engine()
    init_db(eng)

    stmt = text("""
        INSERT OR REPLACE INTO model_runs (
            model_version, algorithm, params_json, metrics_json, data_hash, git_commit, is_active
        ) VALUES (
            :model_version, :algorithm, :params_json, :metrics_json, :data_hash, :git_commit, :is_active
        )
    """)

    with eng.begin() as conn:
        if is_active == 1:
            conn.execute(text("UPDATE model_runs SET is_active = 0 WHERE is_active = 1"))
        conn.execute(
            stmt,
            {
                "model_version": model_version,
                "algorithm": algorithm,
                "params_json": json.dumps(params),
                "metrics_json": json.dumps(metrics),
                "data_hash": data_hash,
                "git_commit": git_commit or "unknown",
                "is_active": is_active,
            },
        )


def set_active_model(model_version: str, engine: Engine | None = None) -> bool:
    """Set the given model_version as active (is_active=1) and others to 0."""
    eng = engine or get_engine()
    with eng.begin() as conn:
        exists = conn.execute(
            text("SELECT COUNT(*) FROM model_runs WHERE model_version = :ver"),
            {"ver": model_version},
        ).scalar()
        if not exists:
            return False

        conn.execute(text("UPDATE model_runs SET is_active = 0 WHERE is_active = 1"))
        conn.execute(
            text("UPDATE model_runs SET is_active = 1 WHERE model_version = :ver"),
            {"ver": model_version},
        )
    return True


def get_active_model_run(engine: Engine | None = None) -> dict[str, Any] | None:
    """Retrieve metadata for the currently active model."""
    eng = engine or get_engine()
    init_db(eng)
    with eng.connect() as conn:
        res = (
            conn.execute(
                text("""
            SELECT model_version, algorithm, params_json, metrics_json, data_hash, git_commit, trained_at, is_active
            FROM model_runs
            WHERE is_active = 1
            LIMIT 1
        """)
            )
            .mappings()
            .first()
        )

        if not res:
            # Fallback to the latest model if none explicitly marked active
            res = (
                conn.execute(
                    text("""
                SELECT model_version, algorithm, params_json, metrics_json, data_hash, git_commit, trained_at, is_active
                FROM model_runs
                ORDER BY trained_at DESC
                LIMIT 1
            """)
                )
                .mappings()
                .first()
            )

        if res:
            d = dict(res)
            d["params"] = json.loads(d["params_json"])
            d["metrics"] = json.loads(d["metrics_json"])
            return d
    return None


def insert_prediction(
    input_text: str,
    predicted_label: str,
    confidence: float,
    probabilities: dict[str, float],
    model_version: str,
    source: str = "api",
    review_id: int | None = None,
    engine: Engine | None = None,
) -> int:
    """Insert a single prediction into the predictions table. Returns new prediction_id."""
    eng = engine or get_engine()
    stmt = text("""
        INSERT INTO predictions (
            review_id, input_text, predicted_label, confidence,
            prob_negative, prob_neutral, prob_positive,
            model_version, source
        ) VALUES (
            :review_id, :input_text, :predicted_label, :confidence,
            :prob_negative, :prob_neutral, :prob_positive,
            :model_version, :source
        )
    """)

    with eng.begin() as conn:
        res = conn.execute(
            stmt,
            {
                "review_id": review_id,
                "input_text": input_text,
                "predicted_label": predicted_label,
                "confidence": float(confidence),
                "prob_negative": float(probabilities.get("negative", 0.0)),
                "prob_neutral": float(probabilities.get("neutral", 0.0)),
                "prob_positive": float(probabilities.get("positive", 0.0)),
                "model_version": model_version,
                "source": source,
            },
        )
        return res.lastrowid or 0


def insert_predictions_batch(records: list[dict[str, Any]], engine: Engine | None = None) -> int:
    """Insert multiple predictions in a single transaction."""
    eng = engine or get_engine()
    stmt = text("""
        INSERT INTO predictions (
            review_id, input_text, predicted_label, confidence,
            prob_negative, prob_neutral, prob_positive,
            model_version, source
        ) VALUES (
            :review_id, :input_text, :predicted_label, :confidence,
            :prob_negative, :prob_neutral, :prob_positive,
            :model_version, :source
        )
    """)
    with eng.begin() as conn:
        conn.execute(stmt, records)
    return len(records)


def save_term_stats(
    model_version: str,
    term_records: list[dict[str, Any]],
    engine: Engine | None = None,
) -> None:
    """Save top terms statistics to term_stats table."""
    eng = engine or get_engine()
    init_db(eng)
    with eng.begin() as conn:
        conn.execute(
            text("DELETE FROM term_stats WHERE model_version = :ver"),
            {"ver": model_version},
        )
        stmt = text("""
            INSERT INTO term_stats (model_version, label, term, weight, rank)
            VALUES (:model_version, :label, :term, :weight, :rank)
        """)
        formatted = [
            {
                "model_version": model_version,
                "label": r["label"],
                "term": r["term"],
                "weight": float(r["weight"]),
                "rank": int(r["rank"]),
            }
            for r in term_records
        ]
        conn.execute(stmt, formatted)


def fetch_reviews(
    label: str | None = None,
    department: str | None = None,
    limit: int = 50,
    offset: int = 0,
    engine: Engine | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """Query reviews with optional filters and pagination."""
    eng = engine or get_engine()
    init_db(eng)

    query_parts = [
        "FROM reviews r LEFT JOIN predictions p ON r.review_id = p.review_id AND p.source = 'batch'"
    ]
    where_clauses = []
    params: dict[str, Any] = {"limit": min(limit, 500), "offset": max(offset, 0)}

    if label:
        where_clauses.append("COALESCE(p.predicted_label, r.true_label) = :label")
        params["label"] = label
    if department:
        where_clauses.append("r.department_name = :dept")
        params["dept"] = department

    where_sql = (" WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

    count_sql = f"SELECT COUNT(*) {query_parts[0]} {where_sql}"
    select_sql = f"""
        SELECT
            r.review_id, r.clothing_id, r.age, r.title, r.review_text, r.rating,
            r.true_label, r.recommended, r.positive_feedback_count,
            r.division_name, r.department_name, r.class_name, r.split,
            COALESCE(p.predicted_label, r.true_label) AS predicted_label,
            p.confidence
        {query_parts[0]}
        {where_sql}
        ORDER BY r.review_id ASC
        LIMIT :limit OFFSET :offset
    """

    with eng.connect() as conn:
        total = conn.execute(text(count_sql), params).scalar() or 0
        rows = conn.execute(text(select_sql), params).mappings().all()
        items = [dict(row) for row in rows]

    return items, total


def get_stats_summary(engine: Engine | None = None) -> dict[str, Any]:
    """Compute aggregate stats for GET /stats endpoint."""
    eng = engine or get_engine()
    init_db(eng)

    with eng.connect() as conn:
        total_reviews = conn.execute(text("SELECT COUNT(*) FROM reviews")).scalar() or 0

        # By label
        label_rows = conn.execute(
            text("""
            SELECT true_label, COUNT(*) AS count
            FROM reviews
            GROUP BY true_label
        """)
        ).fetchall()
        by_label = {"negative": 0, "neutral": 0, "positive": 0}
        for lbl, count in label_rows:
            if lbl in by_label:
                by_label[lbl] = int(count)

        # By department
        dept_rows = conn.execute(
            text("""
            SELECT COALESCE(department_name, 'Unknown') AS dept, COUNT(*) AS count, ROUND(AVG(rating), 2) AS avg_rating
            FROM reviews
            GROUP BY department_name
            ORDER BY count DESC
        """)
        ).fetchall()
        by_dept = [
            {"department": r[0], "count": int(r[1]), "avg_rating": float(r[2] or 0.0)}
            for r in dept_rows
        ]

    return {
        "total_reviews": total_reviews,
        "by_label": by_label,
        "by_department": by_dept,
    }


def export_views_to_csv(
    export_dir: str | Path | None = None, engine: Engine | None = None
) -> list[Path]:
    """Export all vw_* views into CSV files for Power BI ingestion."""
    eng = engine or get_engine()
    init_db(eng)
    out_dir = Path(export_dir) if export_dir else EXPORTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    views = [
        "vw_sentiment_overview",
        "vw_sentiment_by_department",
        "vw_rating_vs_predicted",
        "vw_negative_reviews",
        "vw_model_performance",
        "vw_api_usage",
        "vw_top_terms",
    ]

    exported_files: list[Path] = []
    with eng.connect() as conn:
        for v in views:
            df = pd.read_sql_query(f"SELECT * FROM {v}", conn)
            # Power BI friendly naming (e.g. overview.csv or vw_sentiment_overview.csv)
            file_name = f"{v}.csv"
            out_file = out_dir / file_name
            df.to_csv(out_file, index=False)
            exported_files.append(out_file)

            # Also create standard alias names matching DASHBOARD.md table names
            alias_map = {
                "vw_sentiment_overview": "overview.csv",
                "vw_sentiment_by_department": "by_department.csv",
                "vw_rating_vs_predicted": "rating_vs_predicted.csv",
                "vw_negative_reviews": "negative_reviews.csv",
                "vw_model_performance": "model_performance.csv",
                "vw_api_usage": "api_usage.csv",
                "vw_top_terms": "top_terms.csv",
            }
            if v in alias_map:
                alias_file = out_dir / alias_map[v]
                df.to_csv(alias_file, index=False)
                exported_files.append(alias_file)

    return exported_files
