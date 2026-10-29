"""Hyperparameter tuning for baseline and candidate models."""

from typing import Any

from scipy.stats import loguniform
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from src.config import SEED


def tune_multinomial_nb(
    X_train: Any, y_train: Any, cv_folds: int = 5
) -> tuple[Pipeline, dict[str, Any]]:
    """GridSearchCV for MultinomialNB baseline exploring vectorizers, ngrams, and alpha."""
    pipe = Pipeline(
        [
            ("vect", TfidfVectorizer()),
            ("clf", MultinomialNB()),
        ]
    )

    param_grid = [
        {
            "vect": [
                CountVectorizer(min_df=3, max_features=50000),
                TfidfVectorizer(min_df=3, max_df=0.9, sublinear_tf=True, max_features=50000),
            ],
            "vect__ngram_range": [(1, 1), (1, 2)],
            "clf__alpha": [0.01, 0.1, 0.5, 1.0],
            # A learned prior (77% positive) can outweigh the evidence in short reviews; a uniform
            # prior is NB's analogue of class_weight="balanced". Validation macro-F1 decides.
            "clf__fit_prior": [True, False],
        }
    ]

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=SEED)
    grid = GridSearchCV(
        pipe,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
        refit=True,
    )
    grid.fit(X_train, y_train)

    best_params = {
        "vectorizer": "TF-IDF" if isinstance(grid.best_params_["vect"], TfidfVectorizer) else "BoW",
        "ngram_range": list(grid.best_params_["vect__ngram_range"]),
        "alpha": float(grid.best_params_["clf__alpha"]),
        "fit_prior": bool(grid.best_params_["clf__fit_prior"]),
        "best_cv_f1_macro": float(grid.best_score_),
    }
    return grid.best_estimator_, best_params


def tune_logistic_regression(
    X_train: Any, y_train: Any, n_iter: int = 10, cv_folds: int = 5
) -> tuple[Pipeline, dict[str, Any]]:
    """RandomizedSearchCV for Logistic Regression with balanced class weights."""
    pipe = Pipeline(
        [
            (
                "vect",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=3,
                    max_df=0.9,
                    sublinear_tf=True,
                    max_features=50000,
                ),
            ),
            (
                "clf",
                LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED),
            ),
        ]
    )

    param_dist = {
        "clf__C": loguniform(1e-2, 1e2),
        "clf__solver": ["lbfgs"],
    }

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=SEED)
    search = RandomizedSearchCV(
        pipe,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="f1_macro",
        cv=cv,
        random_state=SEED,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train)

    best_params = {
        "C": float(search.best_params_["clf__C"]),
        "solver": str(search.best_params_["clf__solver"]),
        "best_cv_f1_macro": float(search.best_score_),
    }
    return search.best_estimator_, best_params


def tune_random_forest(
    X_train: Any, y_train: Any, n_iter: int = 4, cv_folds: int = 3
) -> tuple[Pipeline, dict[str, Any]]:
    """RandomizedSearchCV for Random Forest."""
    pipe = Pipeline(
        [
            (
                "vect",
                TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_df=0.9, max_features=15000),
            ),
            (
                "clf",
                RandomForestClassifier(class_weight="balanced", random_state=SEED, n_jobs=-1),
            ),
        ]
    )

    param_dist = {
        "clf__n_estimators": [100, 150],
        "clf__max_depth": [None, 20, 30],
        "clf__min_samples_leaf": [1, 2],
    }

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=SEED)
    search = RandomizedSearchCV(
        pipe,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="f1_macro",
        cv=cv,
        random_state=SEED,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train)

    best_params = {
        "n_estimators": int(search.best_params_["clf__n_estimators"]),
        "max_depth": search.best_params_["clf__max_depth"],
        "min_samples_leaf": int(search.best_params_["clf__min_samples_leaf"]),
        "best_cv_f1_macro": float(search.best_score_),
    }
    return search.best_estimator_, best_params


def tune_gradient_boosting(
    X_train: Any, y_train: Any, n_iter: int = 2, cv_folds: int = 3
) -> tuple[Pipeline, dict[str, Any]]:
    """Tuning Gradient Boosting on TruncatedSVD components using fast histogram method."""
    pipe = Pipeline(
        [
            ("vect", TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_features=5000)),
            ("svd", TruncatedSVD(n_components=50, random_state=SEED)),
            ("clf", HistGradientBoostingClassifier(max_iter=50, random_state=SEED)),
        ]
    )

    param_dist = {
        "clf__learning_rate": [0.05, 0.1],
        "clf__max_leaf_nodes": [15, 31],
    }

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=SEED)
    search = RandomizedSearchCV(
        pipe,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="f1_macro",
        cv=cv,
        random_state=SEED,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train)

    best_params = {
        "learning_rate": float(search.best_params_["clf__learning_rate"]),
        "max_leaf_nodes": int(search.best_params_["clf__max_leaf_nodes"]),
        "best_cv_f1_macro": float(search.best_score_),
    }
    return search.best_estimator_, best_params


def tune_candidate_models(
    X_train: Any, y_train: Any, fast_mode: bool = False
) -> dict[str, dict[str, Any]]:
    """Tune candidates and return dictionary of models and their tuning results."""
    results = {}

    print("--- 1/4 Tuning Multinomial Naive Bayes (GridSearchCV) ---")
    nb_model, nb_params = tune_multinomial_nb(X_train, y_train, cv_folds=5 if not fast_mode else 3)
    results["MultinomialNB"] = {
        "model": nb_model,
        "params": nb_params,
        "cv_f1": nb_params["best_cv_f1_macro"],
    }
    print(f"  Best MNB CV Macro-F1: {nb_params['best_cv_f1_macro']:.4f}")

    print("--- 2/4 Tuning Logistic Regression (RandomizedSearchCV) ---")
    lr_model, lr_params = tune_logistic_regression(
        X_train,
        y_train,
        n_iter=10 if not fast_mode else 4,
        cv_folds=5 if not fast_mode else 3,
    )
    results["LogisticRegression"] = {
        "model": lr_model,
        "params": lr_params,
        "cv_f1": lr_params["best_cv_f1_macro"],
    }
    print(f"  Best LR CV Macro-F1: {lr_params['best_cv_f1_macro']:.4f}")

    if not fast_mode:
        print("--- 3/4 Tuning Random Forest (RandomizedSearchCV) ---")
        rf_model, rf_params = tune_random_forest(X_train, y_train, n_iter=3, cv_folds=3)
        results["RandomForest"] = {
            "model": rf_model,
            "params": rf_params,
            "cv_f1": rf_params["best_cv_f1_macro"],
        }
        print(f"  Best RF CV Macro-F1: {rf_params['best_cv_f1_macro']:.4f}")

        print("--- 4/4 Tuning Gradient Boosting (RandomizedSearchCV) ---")
        gb_model, gb_params = tune_gradient_boosting(X_train, y_train, n_iter=2, cv_folds=3)
        results["GradientBoosting"] = {
            "model": gb_model,
            "params": gb_params,
            "cv_f1": gb_params["best_cv_f1_macro"],
        }
        print(f"  Best GB CV Macro-F1: {gb_params['best_cv_f1_macro']:.4f}")

    return results
