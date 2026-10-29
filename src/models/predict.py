"""Prediction service for single, batch, and database review scoring."""

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sqlalchemy import text

from src.config import CLASSES, DEFAULT_MODEL_VERSION, MODELS_DIR
from src.db.repository import (
    get_active_model_run,
    get_engine,
    init_db,
    insert_prediction,
    insert_predictions_batch,
)


def load_active_model(
    models_dir: str | Path | None = None,
) -> tuple[Pipeline, str, dict[str, Any]]:
    """Find active model version from database and load its joblib artifact."""
    eng = get_engine()
    init_db(eng)
    active_run = get_active_model_run(eng)

    if active_run:
        version = active_run["model_version"]
    else:
        version = DEFAULT_MODEL_VERSION

    base_dir = Path(models_dir) if models_dir else MODELS_DIR
    model_path = base_dir / version / "model.joblib"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at: {model_path}. Please run 'make train' first."
        )

    pipeline: Pipeline = joblib.load(model_path)
    metadata = active_run if active_run else {"model_version": version, "algorithm": "Unknown"}
    return pipeline, version, metadata


class ModelService:
    """Encapsulates active model loading, inference, probability formatting, and SQL persistence."""

    def __init__(self, model_pipeline: Pipeline | None = None, version: str | None = None):
        if model_pipeline is not None and version is not None:
            self.pipeline = model_pipeline
            self.version = version
            self.metadata = {"model_version": version}
        else:
            self.pipeline, self.version, self.metadata = load_active_model()

        # Determine class ordering from classifier
        classifier = self.pipeline.named_steps.get("classifier") or self.pipeline.named_steps.get(
            "clf"
        )
        if hasattr(classifier, "classes_"):
            self.classes = list(classifier.classes_)
        else:
            self.classes = CLASSES

    def predict_single(
        self,
        text_content: str,
        source: str = "api",
        review_id: int | None = None,
        persist: bool = True,
    ) -> dict[str, Any]:
        """Score a single input text and optionally persist to SQL."""
        text_str = str(text_content).strip()
        preds = self.pipeline.predict([text_str])
        probs = self.pipeline.predict_proba([text_str])[0]

        predicted_label = str(preds[0])
        confidence = float(np.max(probs))

        probabilities: dict[str, float] = {}
        for cls_name, prob in zip(self.classes, probs):
            probabilities[str(cls_name)] = round(float(prob), 4)

        # Ensure all standard classes present
        for c in CLASSES:
            if c not in probabilities:
                probabilities[c] = 0.0

        # Guarantee probabilities sum to 1.0 (NFR-03, acceptance criterion 3)
        prob_sum = sum(probabilities.values())
        if prob_sum > 0:
            probabilities = {k: round(v / prob_sum, 4) for k, v in probabilities.items()}

        if persist:
            insert_prediction(
                input_text=text_str,
                predicted_label=predicted_label,
                confidence=confidence,
                probabilities=probabilities,
                model_version=self.version,
                source=source,
                review_id=review_id,
            )

        return {
            "label": predicted_label,
            "confidence": round(confidence, 4),
            "probabilities": probabilities,
            "model_version": self.version,
        }

    def predict_batch(
        self,
        texts: list[str],
        source: str = "api",
        persist: bool = True,
    ) -> dict[str, Any]:
        """Score an array of up to 100 texts in batch and optionally persist to SQL."""
        if not texts or len(texts) == 0:
            raise ValueError("Input texts array cannot be empty.")
        if len(texts) > 100:
            raise ValueError(f"Batch size {len(texts)} exceeds maximum allowed limit of 100.")

        clean_texts = [str(t).strip() for t in texts]
        preds = self.pipeline.predict(clean_texts)
        probs_all = self.pipeline.predict_proba(clean_texts)

        results = []
        db_records = []

        for i, text_str in enumerate(clean_texts):
            label = str(preds[i])
            probs = probs_all[i]
            conf = float(np.max(probs))

            probabilities = {str(c): round(float(p), 4) for c, p in zip(self.classes, probs)}
            for c in CLASSES:
                if c not in probabilities:
                    probabilities[c] = 0.0

            prob_sum = sum(probabilities.values())
            if prob_sum > 0:
                probabilities = {k: round(v / prob_sum, 4) for k, v in probabilities.items()}

            pred_obj = {
                "label": label,
                "confidence": round(conf, 4),
                "probabilities": probabilities,
                "model_version": self.version,
            }
            results.append(pred_obj)

            if persist:
                db_records.append(
                    {
                        "review_id": None,
                        "input_text": text_str,
                        "predicted_label": label,
                        "confidence": conf,
                        "prob_negative": probabilities.get("negative", 0.0),
                        "prob_neutral": probabilities.get("neutral", 0.0),
                        "prob_positive": probabilities.get("positive", 0.0),
                        "model_version": self.version,
                        "source": source,
                    }
                )

        if persist and db_records:
            insert_predictions_batch(db_records)

        return {
            "predictions": results,
            "count": len(results),
        }

    def batch_score_all_stored_reviews(self, chunk_size: int = 1000) -> int:
        """FR-14: Score all reviews stored in reviews table and persist predictions with source='batch'."""
        eng = get_engine()
        init_db(eng)

        with eng.connect() as conn:
            reviews_df = pd.read_sql_query(
                "SELECT review_id, title, review_text FROM reviews ORDER BY review_id ASC",
                conn,
            )

        if len(reviews_df) == 0:
            print("No reviews found in database to batch score.")
            return 0

        title = reviews_df["title"].fillna("").astype(str).str.strip()
        rev = reviews_df["review_text"].fillna("").astype(str).str.strip()
        input_texts = (title.where(title == "", title + " ") + rev).tolist()
        review_ids = reviews_df["review_id"].tolist()

        print(f"Batch scoring {len(input_texts)} stored reviews with model {self.version}...")

        # Delete existing batch predictions for this model to maintain idempotency
        with eng.begin() as conn:
            conn.execute(
                text("DELETE FROM predictions WHERE source = 'batch' AND model_version = :ver"),
                {"ver": self.version},
            )

        total_scored = 0
        for start_idx in range(0, len(input_texts), chunk_size):
            end_idx = min(start_idx + chunk_size, len(input_texts))
            batch_texts = input_texts[start_idx:end_idx]
            batch_ids = review_ids[start_idx:end_idx]

            preds = self.pipeline.predict(batch_texts)
            probs_all = self.pipeline.predict_proba(batch_texts)

            records = []
            for i, (rid, txt) in enumerate(zip(batch_ids, batch_texts)):
                probs = probs_all[i]
                probabilities = {str(c): float(p) for c, p in zip(self.classes, probs)}
                conf = float(np.max(probs))

                records.append(
                    {
                        "review_id": int(rid),
                        "input_text": txt,
                        "predicted_label": str(preds[i]),
                        "confidence": conf,
                        "prob_negative": float(probabilities.get("negative", 0.0)),
                        "prob_neutral": float(probabilities.get("neutral", 0.0)),
                        "prob_positive": float(probabilities.get("positive", 0.0)),
                        "model_version": self.version,
                        "source": "batch",
                    }
                )

            insert_predictions_batch(records, eng)
            total_scored += len(records)
            print(f"  Scored {total_scored}/{len(input_texts)} reviews...")

        print(f"Batch scoring completed! {total_scored} reviews persisted to predictions table.")
        return total_scored
