"""Model evaluation, subgroup fairness metrics, error analysis, and theme discovery."""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    silhouette_score,
)

from src.config import CLASSES, ERRORS_PATH, EVALUATION_PATH, SEED, THEMES_PATH


def compute_metrics(y_true: Any, y_pred: Any) -> dict[str, Any]:
    """Calculate accuracy, macro-F1, weighted-F1, and per-class precision/recall/F1."""
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=CLASSES, zero_division=0
    )

    per_class = {}
    for i, cls_name in enumerate(CLASSES):
        per_class[cls_name] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }

    cm = confusion_matrix(y_true, y_pred, labels=CLASSES)
    cm_norm = confusion_matrix(y_true, y_pred, labels=CLASSES, normalize="true")

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_normalized": np.round(cm_norm, 4).tolist(),
    }


def plot_confusion_matrix(
    cm_normalized: list[list[float]],
    save_path: str | Path,
    labels: list[str] = CLASSES,
) -> None:
    """Plot row-normalized confusion matrix and save as PNG."""
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".2%",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar=False,
    )
    plt.title("Normalized Confusion Matrix (Row %)")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def extract_top_terms_per_class(pipeline: Any, top_n: int = 10) -> list[dict[str, Any]]:
    """Extract top N informative terms per class from the model.

    Works for Logistic Regression and Multinomial Naive Bayes.
    """
    vectorizer = pipeline.named_steps.get("vectorizer") or pipeline.named_steps.get("vect")
    classifier = pipeline.named_steps.get("classifier") or pipeline.named_steps.get("clf")

    if vectorizer is None or classifier is None:
        return []

    feature_names = np.array(vectorizer.get_feature_names_out())
    records: list[dict[str, Any]] = []

    if hasattr(classifier, "coef_"):
        coefs = classifier.coef_
        classes = list(getattr(classifier, "classes_", CLASSES))
        for class_idx, label in enumerate(classes):
            top_indices = np.argsort(coefs[class_idx])[-top_n:][::-1]
            for rank, idx in enumerate(top_indices, start=1):
                records.append(
                    {
                        "label": str(label),
                        "term": str(feature_names[idx]),
                        "weight": float(coefs[class_idx, idx]),
                        "rank": rank,
                    }
                )
    elif hasattr(classifier, "feature_log_prob_"):
        log_probs = classifier.feature_log_prob_
        classes = list(getattr(classifier, "classes_", CLASSES))
        overall_mean = np.mean(log_probs, axis=0)
        for class_idx, label in enumerate(classes):
            diff = log_probs[class_idx] - overall_mean
            top_indices = np.argsort(diff)[-top_n:][::-1]
            for rank, idx in enumerate(top_indices, start=1):
                records.append(
                    {
                        "label": str(label),
                        "term": str(feature_names[idx]),
                        "weight": float(diff[idx]),
                        "rank": rank,
                    }
                )

    return records


def run_error_analysis(
    df: pd.DataFrame,
    y_true: list[str],
    y_pred: list[str],
    probabilities: np.ndarray,
    output_path: str | Path | None = None,
    n_per_class: int = 20,
) -> pd.DataFrame:
    """Identify up to 20 misclassified reviews per class and save to reports/errors.csv."""
    out = Path(output_path) if output_path else ERRORS_PATH
    out.parent.mkdir(parents=True, exist_ok=True)

    df_eval = df.copy().reset_index(drop=True)
    df_eval["true_label"] = y_true
    df_eval["predicted_label"] = y_pred

    confidences = np.max(probabilities, axis=1)
    df_eval["confidence"] = np.round(confidences, 4)

    for idx, c in enumerate(CLASSES):
        df_eval[f"prob_{c}"] = np.round(probabilities[:, idx], 4)

    misclassified = df_eval[df_eval["true_label"] != df_eval["predicted_label"]].copy()

    sampled_records = []
    for cls in CLASSES:
        cls_errors = misclassified[misclassified["true_label"] == cls]
        cls_sample = cls_errors.sort_values(by="confidence", ascending=False).head(n_per_class)
        sampled_records.append(cls_sample)

    if sampled_records:
        error_df = pd.concat(sampled_records, ignore_index=True)
    else:
        error_df = pd.DataFrame()

    cols_to_save = [
        "review_id",
        "clothing_id",
        "Clothing ID",
        "age",
        "Age",
        "department_name",
        "Department Name",
        "class_name",
        "Class Name",
        "title",
        "Title",
        "review_text",
        "Review Text",
        "true_label",
        "predicted_label",
        "confidence",
        "prob_negative",
        "prob_neutral",
        "prob_positive",
    ]
    existing_cols = [c for c in cols_to_save if c in error_df.columns]
    error_df[existing_cols].to_csv(out, index=False)
    return error_df


def run_ethics_evaluation(df: pd.DataFrame, y_true: list[str], y_pred: list[str]) -> dict[str, Any]:
    """Evaluate performance across demographic/product subgroups (ETH-04).

    Groups:
    - Department Name
    - Age bands: <30, 30-45, >45
    Flags any group where macro-F1 differs by > 10 points (0.10) from overall.
    """
    df_eval = df.copy().reset_index(drop=True)
    df_eval["true_label"] = y_true
    df_eval["predicted_label"] = y_pred

    overall_macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    results: dict[str, Any] = {
        "overall_macro_f1": float(overall_macro_f1),
        "department_metrics": {},
        "age_band_metrics": {},
        "disparities": [],
    }

    dept_col = (
        "Department Name"
        if "Department Name" in df_eval.columns
        else "department_name"
        if "department_name" in df_eval.columns
        else None
    )
    if dept_col:
        for dept, group in df_eval.groupby(dept_col):
            if len(group) >= 30:
                dept_f1 = float(
                    f1_score(
                        group["true_label"],
                        group["predicted_label"],
                        average="macro",
                        zero_division=0,
                    )
                )
                diff = dept_f1 - overall_macro_f1
                results["department_metrics"][str(dept)] = {
                    "count": len(group),
                    "macro_f1": round(dept_f1, 4),
                    "diff_from_overall": round(diff, 4),
                }
                if abs(diff) > 0.10:
                    results["disparities"].append(
                        {
                            "subgroup_type": "Department",
                            "group": str(dept),
                            "macro_f1": round(dept_f1, 4),
                            "diff": round(diff, 4),
                        }
                    )

    age_col = "Age" if "Age" in df_eval.columns else "age" if "age" in df_eval.columns else None
    if age_col:

        def categorize_age(val):
            if pd.isna(val):
                return "Unknown"
            try:
                a = float(val)
                if a < 30:
                    return "<30"
                elif 30 <= a <= 45:
                    return "30–45"
                else:
                    return ">45"
            except Exception:
                return "Unknown"

        df_eval["age_band"] = df_eval[age_col].apply(categorize_age)
        for band, group in df_eval.groupby("age_band"):
            if len(group) >= 30:
                band_f1 = float(
                    f1_score(
                        group["true_label"],
                        group["predicted_label"],
                        average="macro",
                        zero_division=0,
                    )
                )
                diff = band_f1 - overall_macro_f1
                results["age_band_metrics"][str(band)] = {
                    "count": len(group),
                    "macro_f1": round(band_f1, 4),
                    "diff_from_overall": round(diff, 4),
                }
                if abs(diff) > 0.10:
                    results["disparities"].append(
                        {
                            "subgroup_type": "Age Band",
                            "group": str(band),
                            "macro_f1": round(band_f1, 4),
                            "diff": round(diff, 4),
                        }
                    )

    return results


def discover_negative_themes(
    negative_texts: list[str],
    k_range: range = range(3, 7),
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """FR-11: K-Means on TF-IDF of negative reviews to surface complaint themes."""
    out = Path(output_path) if output_path else THEMES_PATH
    out.parent.mkdir(parents=True, exist_ok=True)

    if len(negative_texts) < 20:
        return {"status": "insufficient_data"}

    vectorizer = TfidfVectorizer(max_features=1000, stop_words="english", min_df=2)
    X_tfidf = vectorizer.fit_transform(negative_texts)

    best_k = 3
    best_score = -1.0
    best_kmeans = None

    scores = {}
    for k in k_range:
        if k >= X_tfidf.shape[0]:
            break
        km = KMeans(n_clusters=k, random_state=SEED, n_init=5)
        cluster_labels = km.fit_predict(X_tfidf)
        score = float(silhouette_score(X_tfidf, cluster_labels))
        scores[k] = round(score, 4)
        if score > best_score:
            best_score = score
            best_k = k
            best_kmeans = km

    terms = vectorizer.get_feature_names_out()
    order_centroids = best_kmeans.cluster_centers_.argsort()[:, ::-1]

    themes = []
    lines = [
        "# Negative Review Complaint Themes (Clustering Analysis)",
        "",
        f"K-Means clustering evaluated for k in {list(k_range)}. Best silhouette score: **{best_score:.4f}** at **k={best_k}**.",
        "",
        "## Silhouette Scores by Cluster Count",
        "",
        "| Clusters (k) | Silhouette Score |",
        "|---|---|",
    ]
    for k, sc in scores.items():
        lines.append(f"| {k} | {sc:.4f} |")

    lines.extend(["", "## Complaint Themes & Top Keywords", ""])
    for i in range(best_k):
        top_words = [terms[ind] for ind in order_centroids[i, :10]]
        theme_summary = ", ".join(top_words)
        themes.append({"cluster": i, "top_words": top_words})
        lines.extend(
            [
                f"### Theme {i + 1}: {top_words[0].capitalize()} / {top_words[1]}",
                f"- **Key descriptors**: `{theme_summary}`",
                "",
            ]
        )

    out.write_text("\n".join(lines), encoding="utf-8")
    return {"best_k": best_k, "silhouette_score": best_score, "themes": themes}


def generate_evaluation_report(
    baseline_metrics: dict[str, Any],
    final_metrics: dict[str, Any],
    candidate_summary: list[dict[str, Any]],
    ethics_metrics: dict[str, Any],
    top_terms: list[dict[str, Any]],
    output_path: str | Path | None = None,
) -> str:
    """Generate reports/evaluation.md comparing baseline vs final and detailing performance."""
    out = Path(output_path) if output_path else EVALUATION_PATH
    out.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Model Evaluation Report",
        "",
        "## 1. Executive Summary & Acceptance Criteria",
        "",
        "| Criterion | Target | Baseline (MNB) | Final Model | Result |",
        "|---|---|---|---|---|",
        f"| Test Macro-F1 | >= 0.65 | {baseline_metrics['macro_f1']:.4f} | {final_metrics['macro_f1']:.4f} | {'PASSED' if final_metrics['macro_f1'] >= 0.65 else 'FAILED'} |",
        f"| Beat Baseline | Final > Baseline | {baseline_metrics['macro_f1']:.4f} | {final_metrics['macro_f1']:.4f} | {'PASSED' if final_metrics['macro_f1'] > baseline_metrics['macro_f1'] else 'FAILED'} |",
        f"| Minority Recall | >= 0.30 | {baseline_metrics['per_class']['negative']['recall']:.4f} | {final_metrics['per_class']['negative']['recall']:.4f} | {'PASSED' if final_metrics['per_class']['negative']['recall'] >= 0.30 else 'FAILED'} |",
        "",
        "## 2. Candidate Model Comparison (Validation Set)",
        "",
        "| Model | Accuracy | Macro-F1 | Weighted-F1 | Notes |",
        "|---|---|---|---|---|",
    ]

    for cand in candidate_summary:
        lines.append(
            f"| {cand['name']} | {cand['accuracy']:.4f} | {cand['macro_f1']:.4f} | {cand['weighted_f1']:.4f} | {cand.get('notes', '')} |"
        )

    lines.extend(
        [
            "",
            "## 3. Final Model Test Performance Breakdown",
            "",
            "| Class | Precision | Recall | F1-Score | Support |",
            "|---|---|---|---|---|",
        ]
    )

    for cls in CLASSES:
        stats = final_metrics["per_class"][cls]
        lines.append(
            f"| **{cls.capitalize()}** | {stats['precision']:.4f} | {stats['recall']:.4f} | {stats['f1']:.4f} | {stats['support']:,} |"
        )

    lines.extend(
        [
            "",
            "### Overall Metrics",
            f"- **Accuracy**: {final_metrics['accuracy']:.4f}",
            f"- **Macro-F1**: {final_metrics['macro_f1']:.4f}",
            f"- **Weighted-F1**: {final_metrics['weighted_f1']:.4f}",
            "",
            "## 4. Confusion Matrix (Normalized)",
            "",
            "| True \\ Pred | Negative | Neutral | Positive |",
            "|---|---|---|---|",
        ]
    )

    cm_norm = final_metrics["confusion_matrix_normalized"]
    for i, row in enumerate(cm_norm):
        lines.append(
            f"| **{CLASSES[i].capitalize()}** | {row[0] * 100:.2f}% | {row[1] * 100:.2f}% | {row[2] * 100:.2f}% |"
        )

    lines.extend(
        [
            "",
            "## 5. Subgroup & Fairness Evaluation (ETH-04)",
            "",
            f"Overall Macro-F1: **{ethics_metrics['overall_macro_f1']:.4f}**",
            "",
            "### Department Breakdown",
            "| Department | Count | Macro-F1 | Difference from Overall |",
            "|---|---|---|---|",
        ]
    )

    for dept, d_stats in ethics_metrics.get("department_metrics", {}).items():
        diff_str = f"{d_stats['diff_from_overall']:+.4f}"
        lines.append(f"| {dept} | {d_stats['count']} | {d_stats['macro_f1']:.4f} | {diff_str} |")

    lines.extend(
        [
            "",
            "### Age Band Breakdown",
            "| Age Band | Count | Macro-F1 | Difference from Overall |",
            "|---|---|---|---|",
        ]
    )

    for band, a_stats in ethics_metrics.get("age_band_metrics", {}).items():
        diff_str = f"{a_stats['diff_from_overall']:+.4f}"
        lines.append(f"| {band} | {a_stats['count']} | {a_stats['macro_f1']:.4f} | {diff_str} |")

    disparities = ethics_metrics.get("disparities", [])
    if disparities:
        lines.extend(["", "> [!WARNING]", "> **Disparities Detected (> 10 points difference)**:"])
        for d in disparities:
            lines.append(
                f"> - {d['subgroup_type']} `{d['group']}`: Macro-F1 = {d['macro_f1']:.4f} ({d['diff']:+.4f})"
            )
    else:
        lines.extend(
            [
                "",
                "> [!NOTE]",
                "> No demographic or department subgroups deviated by more than 10 points from overall macro-F1.",
            ]
        )

    lines.extend(
        [
            "",
            "## 6. Top 10 Informative Terms per Class",
            "",
            "| Rank | Negative | Neutral | Positive |",
            "|---|---|---|---|",
        ]
    )

    terms_by_label = {"negative": [], "neutral": [], "positive": []}
    for t in top_terms:
        if t["label"] in terms_by_label and len(terms_by_label[t["label"]]) < 10:
            terms_by_label[t["label"]].append(f"{t['term']} ({t['weight']:.2f})")

    for rank in range(10):
        neg = terms_by_label["negative"][rank] if rank < len(terms_by_label["negative"]) else ""
        neu = terms_by_label["neutral"][rank] if rank < len(terms_by_label["neutral"]) else ""
        pos = terms_by_label["positive"][rank] if rank < len(terms_by_label["positive"]) else ""
        lines.append(f"| {rank + 1} | {neg} | {neu} | {pos} |")

    content = "\n".join(lines)
    out.write_text(content, encoding="utf-8")
    return content
