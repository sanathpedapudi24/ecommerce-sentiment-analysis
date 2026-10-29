"""Exploratory Data Analysis (EDA) figure generation (FR-04)."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer

from src.config import FIGURES_DIR


def generate_eda_figures(df: pd.DataFrame, output_dir: str | Path | None = None) -> None:
    """Generate and save the four required EDA figures into reports/figures/:

    1. label_distribution.png
    2. text_length_distribution.png
    3. rating_by_department.png
    4. top_terms_per_class.png
    """
    out_dir = Path(output_dir) if output_dir else FIGURES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    df = df.copy()

    # Normalize column names for flexible access
    text_col = "Review Text" if "Review Text" in df.columns else "review_text"
    dept_col = "Department Name" if "Department Name" in df.columns else "department_name"
    rating_col = "Rating" if "Rating" in df.columns else "rating"
    label_col = "true_label" if "true_label" in df.columns else "label"

    # Set aesthetics
    sns.set_theme(style="whitegrid", palette="muted")
    palette = {"positive": "#2e7d32", "neutral": "#757575", "negative": "#d32f2f"}
    order = ["positive", "neutral", "negative"]

    # 1. Label distribution
    plt.figure(figsize=(7, 4.5))
    ax = sns.countplot(
        data=df, x=label_col, order=order, hue=label_col, palette=palette, legend=False
    )
    total = len(df)
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(
                f"{int(height):,} ({height / total * 100:.1f}%)",
                (p.get_x() + p.get_width() / 2.0, height / 2.0),
                ha="center",
                va="center",
                color="white",
                fontweight="bold",
                fontsize=11,
            )
    plt.title("Customer Sentiment Label Distribution", fontsize=13, fontweight="bold")
    plt.xlabel("Sentiment Class")
    plt.ylabel("Number of Reviews")
    plt.tight_layout()
    plt.savefig(out_dir / "label_distribution.png", dpi=200)
    plt.close()

    # 2. Text length distribution
    df["char_length"] = df[text_col].fillna("").astype(str).str.len()
    plt.figure(figsize=(8, 4.5))
    sns.histplot(
        data=df,
        x="char_length",
        hue=label_col,
        palette=palette,
        element="step",
        stat="density",
        common_norm=False,
        bins=35,
    )
    plt.title("Review Text Character Length by Sentiment", fontsize=13, fontweight="bold")
    plt.xlabel("Character Count")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig(out_dir / "text_length_distribution.png", dpi=200)
    plt.close()

    # 3. Rating by department
    if dept_col in df.columns:
        plt.figure(figsize=(9, 5))
        dept_order = df.groupby(dept_col)[rating_col].mean().sort_values(ascending=False).index
        sns.boxplot(
            data=df,
            x=dept_col,
            y=rating_col,
            order=dept_order,
            palette="Blues_r",
            hue=dept_col,
            legend=False,
        )
        plt.title("Customer Rating Distribution by Department", fontsize=13, fontweight="bold")
        plt.xlabel("Department")
        plt.ylabel("Rating (1-5)")
        plt.tight_layout()
        plt.savefig(out_dir / "rating_by_department.png", dpi=200)
        plt.close()

    # 4. Top terms per class
    _fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=False)

    for i, cls in enumerate(order):
        sub_text = df[df[label_col] == cls][text_col].dropna()
        if len(sub_text) > 0:
            min_doc_freq = 1 if len(sub_text) < 3 else 3
            vec = CountVectorizer(stop_words="english", max_features=1000, min_df=min_doc_freq)
            try:
                X_c = vec.fit_transform(sub_text)
                term_counts = X_c.sum(axis=0).A1
                words = vec.get_feature_names_out()
                top_indices = term_counts.argsort()[-10:][::-1]
                top_words = [words[idx] for idx in top_indices]
                top_vals = [term_counts[idx] for idx in top_indices]

                sns.barplot(
                    x=top_vals,
                    y=top_words,
                    ax=axes[i],
                    color=palette[cls],
                )
            except Exception:
                pass
            axes[i].set_title(f"Top Terms: {cls.capitalize()}", fontweight="bold")
            axes[i].set_xlabel("Term Count")

    plt.suptitle("Most Frequent Words by Sentiment Class", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / "top_terms_per_class.png", dpi=200)
    plt.close()

    print(f"EDA figures saved to: {out_dir}")
