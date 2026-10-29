"""Comprehensive test suite covering data splitting, eda, model evaluation, reporting, and CLI commands."""

import numpy as np
import pandas as pd

from src.cli import main
from src.config import RAW_DATA_PATH
from src.data.clean import generate_data_quality_report
from src.data.eda import generate_eda_figures
from src.data.load import get_data_summary
from src.data.split import load_split_ids, make_stratified_split
from src.db.repository import (
    export_views_to_csv,
    fetch_reviews,
    get_engine,
    get_stats_summary,
    init_db,
)
from src.models.evaluate import (
    compute_metrics,
    discover_negative_themes,
    generate_evaluation_report,
    plot_confusion_matrix,
    run_error_analysis,
    run_ethics_evaluation,
)
from src.models.predict import ModelService
from src.models.train import compute_file_sha256
from src.models.tune import tune_logistic_regression, tune_multinomial_nb


def test_split_and_load_split_ids(tmp_path):
    """Test make_stratified_split and load_split_ids."""
    df = pd.DataFrame(
        {
            "Clothing ID": list(range(100)),
            "Age": [30] * 100,
            "Title": ["Title"] * 100,
            "Review Text": ["Good product"] * 100,
            "Rating": [5] * 70 + [3] * 15 + [1] * 15,
            "true_label": ["positive"] * 70 + ["neutral"] * 15 + ["negative"] * 15,
        }
    )
    save_json = tmp_path / "split_ids.json"
    df_split, _split_dict = make_stratified_split(df, save_path=save_json)

    assert "split" in df_split.columns
    assert set(df_split["split"]) == {"train", "val", "test"}
    assert save_json.exists()

    loaded = load_split_ids(save_json)
    assert "train" in loaded
    assert "val" in loaded
    assert "test" in loaded


def test_eda_figures_generation(tmp_path):
    """Test generate_eda_figures saves all 4 charts."""
    df = pd.DataFrame(
        {
            "Clothing ID": [1, 2, 3],
            "Age": [25, 35, 45],
            "Title": ["Nice", "Bad", "Okay"],
            "Review Text": ["Love this dress", "Shrank after wash", "Average quality"],
            "Rating": [5, 1, 3],
            "true_label": ["positive", "negative", "neutral"],
            "Department Name": ["Dresses", "Tops", "Dresses"],
        }
    )
    generate_eda_figures(df, output_dir=tmp_path)
    assert (tmp_path / "label_distribution.png").exists()
    assert (tmp_path / "text_length_distribution.png").exists()
    assert (tmp_path / "rating_by_department.png").exists()
    assert (tmp_path / "top_terms_per_class.png").exists()


def test_data_summary_and_quality_report(tmp_path):
    """Test data summary and data quality report generation."""
    df = pd.DataFrame(
        {
            "Clothing ID": [10, 20],
            "Rating": [5, 2],
            "Review Text": ["Great", "Bad"],
        }
    )
    summary = get_data_summary(df)
    assert summary["total_rows"] == 2
    assert "Rating" in df.columns

    metrics = {
        "initial_rows": 100,
        "dropped_empty_text": 5,
        "rows_after_empty_text_filter": 95,
        "dropped_duplicates": 2,
        "rows_after_dedup": 93,
        "dropped_invalid_ratings": 0,
        "final_rows": 93,
        "label_distribution": {"positive": 70, "neutral": 13, "negative": 10},
        "label_pct": {"positive": 75.27, "neutral": 13.98, "negative": 10.75},
        "null_rates": {"Clothing ID": 0.0, "Title": 10.0},
    }
    report_file = tmp_path / "data_quality.md"
    rep_content = generate_data_quality_report(metrics, output_path=report_file)
    assert report_file.exists()
    assert "Data Quality & Ingestion Report" in rep_content


def test_model_evaluation_helpers(tmp_path):
    """Test plot_confusion_matrix, run_error_analysis, run_ethics_evaluation, themes, and report."""
    cm = [[0.8, 0.1, 0.1], [0.2, 0.6, 0.2], [0.05, 0.05, 0.9]]
    cm_img = tmp_path / "cm.png"
    plot_confusion_matrix(cm, cm_img)
    assert cm_img.exists()

    df_test = pd.DataFrame(
        {
            "review_id": [1, 2, 3],
            "Clothing ID": [10, 20, 30],
            "Age": [25, 40, 60],
            "Department Name": ["Tops", "Dresses", "Tops"],
            "Class Name": ["Knits", "Casual", "Knits"],
            "Title": ["T1", "T2", "T3"],
            "Review Text": ["Text 1", "Text 2", "Text 3"],
        }
    )
    y_true = ["positive", "neutral", "negative"]
    y_pred = ["negative", "neutral", "positive"]
    probs = [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]]

    run_error_analysis(
        df_test, y_true, y_pred, np.array(probs), output_path=tmp_path / "errors.csv"
    )
    assert (tmp_path / "errors.csv").exists()

    ethics = run_ethics_evaluation(df_test, y_true, y_pred)
    assert "overall_macro_f1" in ethics

    neg_texts = [
        "cheap fabric tore immediately",
        "shrank in wash awful",
        "hideous fit terrible style",
        "very bad material disappointed",
        "poor quality zipper broke",
        "scratchy material returned",
        "too tight and small",
        "horrible seams loose threads",
        "not as pictured ugly color",
        "thin see-through fabric bad",
        "shrunk two sizes completely ruined",
        "waste of money fell apart",
        "color faded after wash",
        "defective hem line crooked",
        "cheap button popped off",
        "wrinkled terrible feel",
        "itchy awful fit",
        "frayed edges poor workmanship",
        "extremely disappointed with quality",
        "disaster purchase hated it",
    ]
    discover_negative_themes(neg_texts, k_range=range(2, 4), output_path=tmp_path / "themes.md")
    assert (tmp_path / "themes.md").exists()

    base_m = compute_metrics(["positive", "negative"], ["positive", "negative"])
    fin_m = compute_metrics(["positive", "negative"], ["positive", "negative"])
    generate_evaluation_report(
        baseline_metrics=base_m,
        final_metrics=fin_m,
        candidate_summary=[
            {
                "name": "M1",
                "accuracy": 0.8,
                "macro_f1": 0.7,
                "weighted_f1": 0.8,
                "notes": "",
            }
        ],
        ethics_metrics=ethics,
        top_terms=[{"label": "positive", "term": "love", "weight": 1.5, "rank": 1}],
        output_path=tmp_path / "eval.md",
    )
    assert (tmp_path / "eval.md").exists()


def test_repository_export_and_stats(tmp_path):
    """Test fetch_reviews, get_stats_summary, and export_views_to_csv."""
    eng = get_engine()
    init_db(eng)

    summary = get_stats_summary(eng)
    assert "total_reviews" in summary
    assert "by_label" in summary

    items, total = fetch_reviews(limit=10, offset=0, engine=eng)
    assert isinstance(items, list)
    assert total >= 0

    exported = export_views_to_csv(export_dir=tmp_path, engine=eng)
    assert len(exported) > 0


def test_tuning_helpers():
    """Test hyperparameter tuning modules on toy data with repeated words."""
    X = [
        "love this great shirt dress",
        "fantastic fit perfect great dress",
        "terrible awful bad quality fabric",
        "cheap fabric shrank bad hated it",
        "okay shirt nothing special fabric",
        "average blouse moderate great style",
        "love this great dress shirt",
        "fantastic dress fit great quality",
        "terrible bad fabric quality awful",
        "cheap fabric shrank bad shirt",
        "okay great dress shirt fabric",
        "average great blouse shirt quality",
    ]
    y = [
        "positive",
        "positive",
        "negative",
        "negative",
        "neutral",
        "neutral",
        "positive",
        "positive",
        "negative",
        "negative",
        "neutral",
        "neutral",
    ]

    _, mnb_params = tune_multinomial_nb(X, y, cv_folds=2)
    assert "alpha" in mnb_params

    _, lr_params = tune_logistic_regression(X, y, n_iter=2, cv_folds=2)
    assert "C" in lr_params


def test_batch_scoring_service():
    """Test batch_score_all_stored_reviews in ModelService."""
    service = ModelService()
    # Batch scoring a subset of reviews
    scored = service.batch_score_all_stored_reviews(chunk_size=10000)
    assert scored > 0


def test_cli_subcommands(tmp_path):
    """Test invoking CLI subcommands."""
    # Test promote
    res = main(["promote", "--version", "v1.0.0"])
    assert res == 0

    # Test evaluate
    res_eval = main(["evaluate"])
    assert res_eval == 0

    # Test export
    res_exp = main(["export", "--output-dir", str(tmp_path)])
    assert res_exp == 0

    # Test eda
    res_eda = main(["eda"])
    assert res_eda == 0

    # Test hash computation
    h = compute_file_sha256(RAW_DATA_PATH)
    assert h.startswith("sha256:")
