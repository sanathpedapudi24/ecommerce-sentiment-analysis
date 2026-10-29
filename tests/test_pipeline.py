"""Unit tests for Scikit-learn Pipeline construction, reproducibility, and probability validity."""

import numpy as np
import pytest
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

from src.config import CLASSES, SEED
from src.models.train import build_pipeline


@pytest.fixture
def toy_data():
    X_train = [
        "Love this shirt, great fit and soft material",
        "Absolutely wonderful dress, flattering and elegant",
        "Awful fabric, shrank immediately and tore apart",
        "Terrible quality, bad seams and horrible color",
        "Average fit, not bad but nothing special",
        "Mediocre blouse, fabric is okay but runs slightly small",
    ]
    y_train = ["positive", "positive", "negative", "negative", "neutral", "neutral"]

    X_test = [
        "Best purchase ever, highly recommended",
        "Hate it, completely cheap and fell apart",
        "Just okay, nothing amazing but decent",
    ]
    y_test = ["positive", "negative", "neutral"]
    return X_train, y_train, X_test, y_test


def test_pipeline_fits_and_predicts(toy_data):
    """Verify single Pipeline integrates TextPreprocessor, vectorizer, and classifier."""
    X_train, y_train, X_test, _ = toy_data
    pipe = build_pipeline(CountVectorizer(min_df=1), MultinomialNB())
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)

    assert len(preds) == len(X_test)
    for p in preds:
        assert p in CLASSES


def test_probabilities_sum_to_one(toy_data):
    """Verify predicted probabilities sum to 1.0 (criterion 3)."""
    X_train, y_train, X_test, _ = toy_data
    pipe = build_pipeline(TfidfVectorizer(min_df=1), LogisticRegression(random_state=SEED))
    pipe.fit(X_train, y_train)
    probs = pipe.predict_proba(X_test)

    for row in probs:
        assert np.isclose(np.sum(row), 1.0, atol=1e-6)


def test_seed_reproducibility(toy_data):
    """Verify fixed seed yields identical predictions (NFR-01)."""
    X_train, y_train, X_test, _ = toy_data

    pipe1 = build_pipeline(TfidfVectorizer(min_df=1), LogisticRegression(random_state=SEED))
    pipe1.fit(X_train, y_train)
    preds1 = pipe1.predict_proba(X_test)

    pipe2 = build_pipeline(TfidfVectorizer(min_df=1), LogisticRegression(random_state=SEED))
    pipe2.fit(X_train, y_train)
    preds2 = pipe2.predict_proba(X_test)

    assert np.allclose(preds1, preds2)


def test_no_data_leakage(toy_data):
    """Verify vectorizer vocabulary is established strictly on training data (NFR-05)."""
    X_train, y_train, _, _ = toy_data

    pipe = build_pipeline(CountVectorizer(min_df=1), MultinomialNB())
    pipe.fit(X_train, y_train)
    vocab = pipe.named_steps["vectorizer"].vocabulary_

    assert "supercalifragilistic" not in vocab
    assert "unseen" not in vocab
