"""Flask REST API application for customer review sentiment classification."""

import logging
import time
from typing import Any

from flask import Flask, Response, g, jsonify, request
from sqlalchemy import text
from werkzeug.exceptions import BadRequest

from src.api.schemas import (
    make_error_response,
    validate_batch_payload,
    validate_predict_payload,
)
from src.config import API_HOST, API_PORT, CLASSES, REQUEST_SIZE_LIMIT
from src.db.repository import (
    fetch_reviews,
    get_active_model_run,
    get_dashboard_data,
    get_engine,
    get_stats_summary,
    init_db,
)
from src.models.predict import ModelService

# Configure structured logging (no review text logged)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("sentiment_api")


def create_app(test_config: dict[str, Any] | None = None) -> Flask:
    """Application factory for Flask sentiment prediction service."""
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = REQUEST_SIZE_LIMIT

    if test_config:
        app.config.update(test_config)

    # Initialize DB connection and check health
    engine = get_engine()
    init_db(engine)
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as e:
        logger.critical(f"Database connection failed: {e}")
        raise RuntimeError(f"Database startup check failed: {e}")

    # Load active model service
    model_service: ModelService | None = None
    try:
        model_service = ModelService()
        logger.info(f"Loaded active model version: {model_service.version}")
    except Exception as e:
        logger.warning(f"Could not load active model at startup: {e}")
        if not app.config.get("TESTING"):
            # In production, fail-fast as required by API.md
            raise RuntimeError(f"Model service startup check failed: {e}")

    # Attach model_service to app for request handlers
    app.model_service = model_service

    # Middleware: Request timing and structured logging (ETH-02: no review text)
    @app.before_request
    def start_timer():
        g.start_time = time.time()

    @app.after_request
    def log_and_header(response: Response) -> Response:
        duration_ms = (time.time() - getattr(g, "start_time", time.time())) * 1000
        # Add X-Model-Version header
        current_version = (
            getattr(app.model_service, "version", "unknown") if app.model_service else "unknown"
        )
        response.headers["X-Model-Version"] = current_version

        # Structured request logging
        logger.info(
            f"method={request.method} path={request.path} status={response.status_code} "
            f"duration_ms={duration_ms:.2f}"
        )
        return response

    # ------------------ Routes ------------------

    @app.route("/", methods=["GET"])
    def index():
        """GET / -> single-page UI that scores reviews through /predict and /predict/batch."""
        return app.send_static_file("index.html")

    @app.route("/dashboard", methods=["GET"])
    def dashboard():
        """GET /dashboard -> 4-page reporting dashboard (same views as the Power BI exports)."""
        return app.send_static_file("dashboard.html")

    @app.route("/dashboard/data", methods=["GET"])
    def dashboard_data():
        """GET /dashboard/data -> all reporting views as JSON, read live from SQLite."""
        return jsonify(get_dashboard_data(engine)), 200

    @app.route("/health", methods=["GET"])
    def health():
        """GET /health -> 200 {"status":"ok","model_version":"v1.0.0","db":"ok"}"""
        db_status = "ok"
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
        except Exception:
            db_status = "error"

        current_ver = app.model_service.version if app.model_service else "none"
        is_healthy = db_status == "ok" and app.model_service is not None

        status_code = 200 if is_healthy else 500
        return jsonify(
            {
                "status": "ok" if is_healthy else "degraded",
                "model_version": current_ver,
                "db": db_status,
            }
        ), status_code

    @app.route("/model/info", methods=["GET"])
    def model_info():
        """GET /model/info -> metadata and metrics of the currently active model."""
        active_run = get_active_model_run(engine)
        if not active_run:
            return jsonify(
                {
                    "model_version": getattr(app.model_service, "version", "unknown"),
                    "algorithm": "Unknown",
                    "trained_at": "unknown",
                    "metrics": {"accuracy": 0.0, "macro_f1": 0.0},
                    "labels": CLASSES,
                }
            ), 200

        metrics = active_run.get("metrics", {})
        accuracy = metrics.get("test_accuracy", metrics.get("accuracy", 0.0))
        macro_f1 = metrics.get("test_macro_f1", metrics.get("macro_f1", 0.0))

        return jsonify(
            {
                "model_version": active_run["model_version"],
                "algorithm": active_run["algorithm"],
                "trained_at": active_run["trained_at"],
                "metrics": {
                    "accuracy": round(float(accuracy), 4),
                    "macro_f1": round(float(macro_f1), 4),
                },
                "labels": CLASSES,
            }
        ), 200

    @app.route("/predict", methods=["POST"])
    def predict():
        """POST /predict -> Single review sentiment classification."""
        if not request.is_json:
            return make_error_response("bad_request", "Request body must be valid JSON", 400)

        try:
            data = request.get_json(silent=True)
            if data is None:
                return make_error_response("bad_request", "Invalid JSON syntax", 400)
        except BadRequest:
            return make_error_response("bad_request", "Invalid JSON body", 400)

        is_valid, error_msg = validate_predict_payload(data)
        if not is_valid:
            return make_error_response("validation_error", error_msg or "Invalid payload", 422)

        if not app.model_service:
            return make_error_response("service_unavailable", "Model service is not loaded", 500)

        try:
            result = app.model_service.predict_single(data["text"], source="api", persist=True)
            return jsonify(result), 200
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return make_error_response("internal_error", "Inference failure", 500)

    @app.route("/predict/batch", methods=["POST"])
    def predict_batch():
        """POST /predict/batch -> Batch review prediction (1 to 100 texts)."""
        if not request.is_json:
            return make_error_response("bad_request", "Request body must be valid JSON", 400)

        try:
            data = request.get_json(silent=True)
            if data is None:
                return make_error_response("bad_request", "Invalid JSON syntax", 400)
        except BadRequest:
            return make_error_response("bad_request", "Invalid JSON body", 400)

        is_valid, error_msg = validate_batch_payload(data)
        if not is_valid:
            return make_error_response("validation_error", error_msg or "Invalid payload", 422)

        if not app.model_service:
            return make_error_response("service_unavailable", "Model service is not loaded", 500)

        try:
            result = app.model_service.predict_batch(data["texts"], source="api", persist=True)
            return jsonify(result), 200
        except Exception as e:
            logger.error(f"Batch prediction error: {e}")
            return make_error_response("internal_error", "Batch inference failure", 500)

    @app.route("/reviews", methods=["GET"])
    def list_reviews():
        """GET /reviews -> Filterable paginated list of reviews with predicted labels."""
        label = request.args.get("label")
        dept = request.args.get("department")

        try:
            limit = int(request.args.get("limit", 50))
        except ValueError:
            limit = 50
        limit = max(1, min(limit, 500))  # Capped at 500

        try:
            offset = int(request.args.get("offset", 0))
        except ValueError:
            offset = 0
        offset = max(0, offset)

        items, total = fetch_reviews(
            label=label, department=dept, limit=limit, offset=offset, engine=engine
        )
        return jsonify(
            {
                "items": items,
                "total": total,
                "limit": limit,
                "offset": offset,
            }
        ), 200

    @app.route("/stats", methods=["GET"])
    def stats():
        """GET /stats -> Aggregate review sentiment and department metrics."""
        summary = get_stats_summary(engine)
        return jsonify(summary), 200

    # ------------------ Error Handlers ------------------

    @app.errorhandler(400)
    def handle_bad_request(e):
        return make_error_response(
            "bad_request",
            str(e.description if hasattr(e, "description") else "Bad request"),
            400,
        )

    @app.errorhandler(404)
    def handle_not_found(e):
        return make_error_response(
            "not_found", "The requested resource was not found on this server", 404
        )

    @app.errorhandler(405)
    def handle_method_not_allowed(e):
        return make_error_response(
            "method_not_allowed", "The method is not allowed for the requested URL", 405
        )

    @app.errorhandler(413)
    def handle_payload_too_large(e):
        return make_error_response("payload_too_large", "Request payload exceeds 1MB limit", 413)

    @app.errorhandler(422)
    def handle_unprocessable_entity(e):
        return make_error_response("validation_error", "Validation failed", 422)

    @app.errorhandler(500)
    def handle_internal_server_error(e):
        return make_error_response("internal_error", "An unexpected server error occurred", 500)

    return app


app = None


def get_wsgi_app():
    """Entry point for Gunicorn / WSGI servers."""
    global app
    if app is None:
        app = create_app()
    return app


if __name__ == "__main__":
    server_app = create_app()
    server_app.run(host=API_HOST, port=API_PORT, debug=False)
