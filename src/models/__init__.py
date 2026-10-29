"""Model training, tuning, evaluation, and prediction pipeline."""

from .evaluate import (
    compute_metrics,
    discover_negative_themes,
    extract_top_terms_per_class,
    generate_evaluation_report,
    plot_confusion_matrix,
    run_error_analysis,
    run_ethics_evaluation,
)
from .predict import ModelService, load_active_model
from .train import build_pipeline, train_all_models, train_and_select_model
from .tune import tune_candidate_models

__all__ = [
    "ModelService",
    "build_pipeline",
    "compute_metrics",
    "discover_negative_themes",
    "extract_top_terms_per_class",
    "generate_evaluation_report",
    "load_active_model",
    "plot_confusion_matrix",
    "run_error_analysis",
    "run_ethics_evaluation",
    "train_all_models",
    "train_and_select_model",
    "tune_candidate_models",
]
