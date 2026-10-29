"""Model training pipeline, candidate comparison, serialization, and artifact management."""

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from src.config import (
    DEFAULT_MODEL_VERSION,
    MODELS_DIR,
    RAW_DATA_PATH,
    SEED,
)
from src.db.repository import (
    get_engine,
    init_db,
    register_model_run,
    save_term_stats,
    set_active_model,
)
from src.features.text import TextPreprocessor
from src.models.evaluate import (
    compute_metrics,
    discover_negative_themes,
    extract_top_terms_per_class,
    generate_evaluation_report,
    plot_confusion_matrix,
    run_error_analysis,
    run_ethics_evaluation,
)
from src.models.tune import tune_candidate_models


def compute_file_sha256(filepath: str | Path) -> str:
    """Compute SHA-256 hash of a file."""
    p = Path(filepath)
    if not p.exists():
        return "sha256:unknown"
    sha = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return f"sha256:{sha.hexdigest()}"


def build_pipeline(vectorizer: Any, classifier: Any) -> Pipeline:
    """Package TextPreprocessor, Vectorizer, and Classifier as a single Pipeline."""
    return Pipeline(
        [
            ("preprocessor", TextPreprocessor()),
            ("vectorizer", vectorizer),
            ("classifier", classifier),
        ]
    )


def load_dataset_from_db_or_clean() -> pd.DataFrame:
    """Load reviews dataset from SQLite database or fallback to loading & cleaning raw CSV."""
    eng = get_engine()
    init_db(eng)
    with eng.connect() as conn:
        df = pd.read_sql_query("SELECT * FROM reviews", conn)
    if len(df) == 0:
        from src.data.clean import clean_data
        from src.data.load import load_raw_data
        from src.data.split import make_stratified_split

        df_raw = load_raw_data()
        df_clean, _ = clean_data(df_raw)
        df, _ = make_stratified_split(df_clean)
        from src.db.repository import ingest_reviews

        ingest_reviews(df, eng)
    return df


def train_and_select_model(
    version: str = DEFAULT_MODEL_VERSION,
    fast_mode: bool = False,
    allow_dirty: bool = True,
) -> dict[str, Any]:
    """Train baseline and candidate models, tune, evaluate on test, and persist artifacts."""
    print(f"=== Starting Model Training Pipeline for version: {version} ===")
    df = load_dataset_from_db_or_clean()

    # Model input text = Title + " " + Review Text
    title = df["title"].fillna("").astype(str).str.strip()
    rev = df["review_text"].fillna("").astype(str).str.strip()
    df["input_text"] = title.where(title == "", title + " ") + rev

    train_mask = df["split"] == "train"
    val_mask = df["split"] == "val"
    test_mask = df["split"] == "test"

    X_train, y_train = (
        df.loc[train_mask, "input_text"].tolist(),
        df.loc[train_mask, "true_label"].tolist(),
    )
    X_val, y_val = (
        df.loc[val_mask, "input_text"].tolist(),
        df.loc[val_mask, "true_label"].tolist(),
    )
    X_test, y_test = (
        df.loc[test_mask, "input_text"].tolist(),
        df.loc[test_mask, "true_label"].tolist(),
    )

    print(f"Dataset split sizes: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

    # 1. Baseline: Multinomial Naive Bayes (BoW & TF-IDF)
    print("\n[Stage 1/5] Training Untuned Baseline Models (Naive Bayes)...")
    bow_vect = CountVectorizer(ngram_range=(1, 2), min_df=3, max_features=50000)
    baseline_bow = build_pipeline(bow_vect, MultinomialNB(alpha=1.0))
    baseline_bow.fit(X_train, y_train)
    val_preds_bow = baseline_bow.predict(X_val)
    metrics_bow = compute_metrics(y_val, val_preds_bow)
    print(
        f"  Untuned MNB (BoW)     -> Val Macro-F1: {metrics_bow['macro_f1']:.4f}, Accuracy: {metrics_bow['accuracy']:.4f}"
    )

    tfidf_vect = TfidfVectorizer(
        ngram_range=(1, 2), min_df=3, max_df=0.9, sublinear_tf=True, max_features=50000
    )
    baseline_tfidf = build_pipeline(tfidf_vect, MultinomialNB(alpha=1.0))
    baseline_tfidf.fit(X_train, y_train)
    val_preds_tfidf = baseline_tfidf.predict(X_val)
    metrics_tfidf = compute_metrics(y_val, val_preds_tfidf)
    print(
        f"  Untuned MNB (TF-IDF)  -> Val Macro-F1: {metrics_tfidf['macro_f1']:.4f}, Accuracy: {metrics_tfidf['accuracy']:.4f}"
    )

    primary_baseline = (
        baseline_bow if metrics_bow["macro_f1"] >= metrics_tfidf["macro_f1"] else baseline_tfidf
    )
    (metrics_bow if metrics_bow["macro_f1"] >= metrics_tfidf["macro_f1"] else metrics_tfidf)

    # 2. Candidate comparison
    print("\n[Stage 2/5] Training Standard Candidates (Logistic Regression, Random Forest)...")
    lr_pipe = build_pipeline(
        TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=3,
            max_df=0.9,
            sublinear_tf=True,
            max_features=50000,
        ),
        LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED),
    )
    lr_pipe.fit(X_train, y_train)
    val_preds_lr = lr_pipe.predict(X_val)
    metrics_lr = compute_metrics(y_val, val_preds_lr)
    print(
        f"  Standard Logistic Regression -> Val Macro-F1: {metrics_lr['macro_f1']:.4f}, Accuracy: {metrics_lr['accuracy']:.4f}"
    )

    rf_pipe = build_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.9, max_features=20000),
        RandomForestClassifier(
            class_weight="balanced", n_estimators=100, random_state=SEED, n_jobs=-1
        ),
    )
    rf_pipe.fit(X_train, y_train)
    val_preds_rf = rf_pipe.predict(X_val)
    metrics_rf = compute_metrics(y_val, val_preds_rf)
    print(
        f"  Standard Random Forest       -> Val Macro-F1: {metrics_rf['macro_f1']:.4f}, Accuracy: {metrics_rf['accuracy']:.4f}"
    )

    # 3. Hyperparameter Tuning
    print("\n[Stage 3/5] Running Hyperparameter Search (GridSearchCV & RandomizedSearchCV)...")
    # Optimize tuning time by pre-transforming text for hyperparameter exploration
    preprocessor = TextPreprocessor()
    X_train_preprocessed = preprocessor.transform(X_train)
    tuning_results = tune_candidate_models(X_train_preprocessed, y_train, fast_mode=fast_mode)

    # Reconstruct best pipelines with TextPreprocessor
    candidates: dict[str, tuple[Pipeline, dict[str, Any], dict[str, Any]]] = {}

    # Candidate A: Tuned MNB
    best_nb_params = tuning_results["MultinomialNB"]["params"]
    v_type = best_nb_params["vectorizer"]
    ng = tuple(best_nb_params["ngram_range"])
    v_obj = (
        CountVectorizer(ngram_range=ng, min_df=3, max_features=50000)
        if v_type == "BoW"
        else TfidfVectorizer(
            ngram_range=ng, min_df=3, max_df=0.9, sublinear_tf=True, max_features=50000
        )
    )
    tuned_nb = build_pipeline(v_obj, MultinomialNB(alpha=best_nb_params["alpha"]))
    tuned_nb.fit(X_train, y_train)
    m_tuned_nb = compute_metrics(y_val, tuned_nb.predict(X_val))
    candidates["Tuned MultinomialNB"] = (tuned_nb, best_nb_params, m_tuned_nb)

    # Candidate B: Tuned Logistic Regression
    best_lr_params = tuning_results["LogisticRegression"]["params"]
    tuned_lr = build_pipeline(
        TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=3,
            max_df=0.9,
            sublinear_tf=True,
            max_features=50000,
        ),
        LogisticRegression(
            C=best_lr_params["C"],
            class_weight="balanced",
            max_iter=1000,
            random_state=SEED,
        ),
    )
    tuned_lr.fit(X_train, y_train)
    m_tuned_lr = compute_metrics(y_val, tuned_lr.predict(X_val))
    candidates["Tuned Logistic Regression"] = (tuned_lr, best_lr_params, m_tuned_lr)

    # Candidate C: Standard / Tuned Random Forest
    candidates["Random Forest"] = (
        rf_pipe,
        {"n_estimators": 100, "class_weight": "balanced"},
        metrics_rf,
    )

    # Candidate summary table
    candidate_summary = [
        {
            "name": "Untuned MNB (BoW)",
            "accuracy": metrics_bow["accuracy"],
            "macro_f1": metrics_bow["macro_f1"],
            "weighted_f1": metrics_bow["weighted_f1"],
            "notes": "Baseline",
        },
        {
            "name": "Untuned MNB (TF-IDF)",
            "accuracy": metrics_tfidf["accuracy"],
            "macro_f1": metrics_tfidf["macro_f1"],
            "weighted_f1": metrics_tfidf["weighted_f1"],
            "notes": "Baseline",
        },
        {
            "name": "Standard Logistic Regression",
            "accuracy": metrics_lr["accuracy"],
            "macro_f1": metrics_lr["macro_f1"],
            "weighted_f1": metrics_lr["weighted_f1"],
            "notes": "C=1.0, balanced",
        },
        {
            "name": "Standard Random Forest",
            "accuracy": metrics_rf["accuracy"],
            "macro_f1": metrics_rf["macro_f1"],
            "weighted_f1": metrics_rf["weighted_f1"],
            "notes": "Trees=100, balanced",
        },
        {
            "name": "Tuned MultinomialNB",
            "accuracy": m_tuned_nb["accuracy"],
            "macro_f1": m_tuned_nb["macro_f1"],
            "weighted_f1": m_tuned_nb["weighted_f1"],
            "notes": f"alpha={best_nb_params['alpha']}",
        },
        {
            "name": "Tuned Logistic Regression",
            "accuracy": m_tuned_lr["accuracy"],
            "macro_f1": m_tuned_lr["macro_f1"],
            "weighted_f1": m_tuned_lr["weighted_f1"],
            "notes": f"C={best_lr_params['C']:.2f}, balanced",
        },
    ]

    # Select best candidate on validation macro-F1
    best_candidate_name = max(candidates.keys(), key=lambda k: candidates[k][2]["macro_f1"])
    best_pipeline, best_params, best_val_metrics = candidates[best_candidate_name]
    print(
        f"\n[Selection Rule] Selected Best Model: '{best_candidate_name}' (Val Macro-F1 = {best_val_metrics['macro_f1']:.4f})"
    )

    # 4. Final Evaluation on Held-Out Test Set (Touched once)
    print("\n[Stage 4/5] Evaluating Best Model on Held-Out Test Set (NFR-05)...")
    test_preds = best_pipeline.predict(X_test)
    test_probs = best_pipeline.predict_proba(X_test)
    test_metrics = compute_metrics(y_test, test_preds)
    test_metrics["val_macro_f1"] = best_val_metrics["macro_f1"]
    test_metrics["val_accuracy"] = best_val_metrics["accuracy"]

    # Baseline test metrics for comparison
    baseline_test_preds = primary_baseline.predict(X_test)
    baseline_test_metrics = compute_metrics(y_test, baseline_test_preds)

    print(f"  Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(
        f"  Test Macro-F1: {test_metrics['macro_f1']:.4f} (Baseline: {baseline_test_metrics['macro_f1']:.4f})"
    )
    print(
        f"  Per-class F1:  Negative={test_metrics['per_class']['negative']['f1']:.4f}, Neutral={test_metrics['per_class']['neutral']['f1']:.4f}, Positive={test_metrics['per_class']['positive']['f1']:.4f}"
    )

    # Verify acceptance criteria
    assert test_metrics["macro_f1"] >= 0.65, (
        f"Acceptance criterion 1 failed: Test Macro-F1 {test_metrics['macro_f1']:.4f} < 0.65"
    )
    assert test_metrics["macro_f1"] > baseline_test_metrics["macro_f1"], (
        f"Acceptance criterion 2 failed: Final {test_metrics['macro_f1']:.4f} <= Baseline {baseline_test_metrics['macro_f1']:.4f}"
    )
    print("  => Acceptance Criteria 1 & 2 PASSED!")

    # 5. Persist Model, Metadata, Visuals, and Reports
    print("\n[Stage 5/5] Persisting Model Artifacts and Generating Reports...")
    version_dir = MODELS_DIR / version
    version_dir.mkdir(parents=True, exist_ok=True)

    # Save model.joblib
    model_path = version_dir / "model.joblib"
    joblib.dump(best_pipeline, model_path)
    model_size_mb = model_path.stat().st_size / (1024 * 1024)
    print(f"  Saved model artifact: {model_path} ({model_size_mb:.2f} MB)")
    assert model_size_mb < 100.0, f"NFR-04 failed: Model artifact {model_size_mb:.2f} MB >= 100 MB"

    # Save confusion matrix plot
    cm_path = version_dir / "confusion_matrix.png"
    plot_confusion_matrix(test_metrics["confusion_matrix_normalized"], cm_path)

    # Extract top terms
    top_terms = extract_top_terms_per_class(best_pipeline, top_n=10)
    save_term_stats(version, top_terms)

    # Error analysis
    df_test = df.loc[test_mask].copy()
    run_error_analysis(df_test, y_test, test_preds, test_probs)

    # Subgroup ethics evaluation
    ethics_results = run_ethics_evaluation(df_test, y_test, test_preds)

    # Discover themes in negative reviews (FR-11)
    neg_texts = df[df["true_label"] == "negative"]["input_text"].tolist()
    discover_negative_themes(neg_texts)

    # Metadata
    data_hash = compute_file_sha256(RAW_DATA_PATH)
    metadata = {
        "model_version": version,
        "algorithm": best_candidate_name,
        "params": best_params,
        "data_hash": data_hash,
        "git_commit": os.getenv("GIT_COMMIT", "local_build"),
        "seed": SEED,
        "sklearn_version": "1.7.2",
        "trained_at": pd.Timestamp.now().isoformat(),
        "train_rows": len(X_train),
        "val_rows": len(X_val),
        "test_rows": len(X_test),
        "metrics": {
            "val_macro_f1": best_val_metrics["macro_f1"],
            "test_macro_f1": test_metrics["macro_f1"],
            "test_accuracy": test_metrics["accuracy"],
            "test_weighted_f1": test_metrics["weighted_f1"],
        },
    }

    (version_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (version_dir / "metrics.json").write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")

    # Insert into model_runs table and promote as active
    register_model_run(
        model_version=version,
        algorithm=best_candidate_name,
        params=best_params,
        metrics={
            "test_accuracy": test_metrics["accuracy"],
            "test_macro_f1": test_metrics["macro_f1"],
            "val_macro_f1": best_val_metrics["macro_f1"],
        },
        data_hash=data_hash,
        git_commit=metadata["git_commit"],
        is_active=1,
    )
    set_active_model(version)

    # Generate evaluation.md
    generate_evaluation_report(
        baseline_metrics=baseline_test_metrics,
        final_metrics=test_metrics,
        candidate_summary=candidate_summary,
        ethics_metrics=ethics_results,
        top_terms=top_terms,
    )

    print(f"=== Model Training Complete! Version {version} is now ACTIVE ===")
    return {
        "model_version": version,
        "algorithm": best_candidate_name,
        "metrics": test_metrics,
        "baseline_metrics": baseline_test_metrics,
    }


def train_all_models() -> dict[str, Any]:
    """Entry point for make train."""
    return train_and_select_model(version=DEFAULT_MODEL_VERSION)
