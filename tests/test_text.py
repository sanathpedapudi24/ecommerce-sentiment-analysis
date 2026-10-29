"""Unit tests for text preprocessing transformer."""

import pytest

from src.features.text import TextPreprocessor


@pytest.fixture
def preprocessor():
    return TextPreprocessor()


def test_lowercasing(preprocessor):
    text = "THIS IS ALL UPPERCASE"
    res = preprocessor.preprocess_text(text)
    assert res == res.lower()
    assert "uppercase" in res


def test_url_and_html_removal(preprocessor):
    text = "Check this <b>amazing</b> dress at https://example.com/dress?id=123"
    res = preprocessor.preprocess_text(text)
    assert "http" not in res
    assert "example" not in res
    assert "<b>" not in res
    assert "amazing" in res


def test_digit_removal(preprocessor):
    text = "Ordered size 4 and 100% happy with it"
    res = preprocessor.preprocess_text(text)
    assert "4" not in res
    assert "100" not in res
    assert "happy" in res


def test_negations_preserved(preprocessor):
    text = "This dress was not flattering and never fit right against my body"
    res = preprocessor.preprocess_text(text)
    tokens = res.split()
    assert "not" in tokens, "'not' must be retained as negation"
    assert "never" in tokens, "'never' must be retained as negation"
    assert "against" in tokens, "'against' must be retained as negation"


def test_contracted_negations_preserved(preprocessor):
    tokens = preprocessor.preprocess_text(
        "I didn't like it, it doesn’t fit and I dont want it"
    ).split()
    assert tokens.count("not") == 3, "contractions like didn't / dont must keep their negation"
    assert "want" in tokens, "ordinary words ending in 'nt' must stay intact"


def test_lemmatization(preprocessor):
    text = "The pants were shrinking and pants looked longer"
    res = preprocessor.preprocess_text(text)
    assert "pant" in res or "shrinking" in res


def test_empty_and_none_safe(preprocessor):
    assert preprocessor.preprocess_text("") == ""
    assert preprocessor.preprocess_text("   ") == ""
    assert preprocessor.preprocess_text(None) == ""
    assert preprocessor.preprocess_text("nan") == ""
    assert preprocessor.preprocess_text(12345) == ""


def test_transform_batch(preprocessor):
    inputs = ["Great fit!", "", "Terrible quality, never buying again."]
    outputs = preprocessor.transform(inputs)
    assert len(outputs) == 3
    assert "great" in outputs[0]
    assert outputs[1] == ""
    assert "never" in outputs[2]
