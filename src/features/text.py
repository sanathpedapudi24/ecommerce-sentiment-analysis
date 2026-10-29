"""Text preprocessing transformer and feature vectorization."""

import re
from collections.abc import Iterable
from typing import Any

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from src.config import MAX_DF, MAX_FEATURES, MIN_DF, NGRAM_RANGE, SUBLINEAR_TF

# Ensure resources are available
try:
    _STOPWORDS: set[str] = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords", quiet=True)
    _STOPWORDS = set(stopwords.words("english"))

try:
    WordNetLemmatizer().lemmatize("dresses")
except LookupError:
    nltk.download("wordnet", quiet=True)

# Negations that must be preserved
NEGATIONS: set[str] = {"no", "not", "nor", "never", "n't", "against"}
CUSTOM_STOPWORDS: set[str] = _STOPWORDS - NEGATIONS

# "didn't" / "didnt" -> "did not": otherwise punctuation stripping leaves "didn" (a stop word) or the
# token "didnt", and the negation is lost. Apostrophe-less forms need an explicit stem list because
# ordinary words also end in "nt" (want, print, elegant).
NEGATION_CONTRACTION_PATTERN = re.compile(
    r"\b(do|does|did|is|was|were|are|ca|could|would|should|wo|have|has|had|ai|need|must)n['’]?t\b"
    r"|n['’]t\b"
)
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
HTML_PATTERN = re.compile(r"<[^>]+>")
DIGIT_PATTERN = re.compile(r"\d+")
PUNCT_PATTERN = re.compile(r"[^\w\s]")
WHITESPACE_PATTERN = re.compile(r"\s+")


class TextPreprocessor(BaseEstimator, TransformerMixin):
    """Scikit-learn compatible transformer for text cleaning, stop-word removal, and lemmatization.

    - Lowercases text
    - Strips URLs, HTML tags, digits, punctuation
    - Tokenises using NLTK word_tokenize
    - Filters out stop words (retains negations)
    - Lemmatises tokens using WordNet
    - Drops tokens shorter than 2 characters
    """

    def __init__(self, keep_negations: bool = True, min_token_len: int = 2):
        self.keep_negations = keep_negations
        self.min_token_len = min_token_len
        self._lemmatizer = None
        self._stop_words = None

    def _get_lemmatizer(self) -> WordNetLemmatizer:
        if self._lemmatizer is None:
            self._lemmatizer = WordNetLemmatizer()
        return self._lemmatizer

    def _get_stopwords(self) -> set[str]:
        if self._stop_words is None:
            if self.keep_negations:
                self._stop_words = CUSTOM_STOPWORDS
            else:
                self._stop_words = _STOPWORDS
        return self._stop_words

    def fit(self, X: Any, y: Any = None) -> "TextPreprocessor":
        """No-op fit to satisfy Scikit-learn transformer protocol."""
        return self

    def preprocess_text(self, text: str | float | None) -> str:
        """Clean a single string."""
        if text is None:
            return ""
        text_str = str(text).lower()
        if not text_str or text_str in ("nan", "none"):
            return ""

        # Remove URLs, HTML, digits, punctuation
        text_str = URL_PATTERN.sub(" ", text_str)
        text_str = HTML_PATTERN.sub(" ", text_str)
        text_str = NEGATION_CONTRACTION_PATTERN.sub(r"\1 not", text_str)
        text_str = DIGIT_PATTERN.sub(" ", text_str)
        text_str = PUNCT_PATTERN.sub(" ", text_str)
        text_str = WHITESPACE_PATTERN.sub(" ", text_str).strip()

        if not text_str:
            return ""

        try:
            tokens = word_tokenize(text_str)
        except Exception:
            # Fallback regex tokenization if punkt lookup fails
            tokens = text_str.split()

        lemmatizer = self._get_lemmatizer()
        stop_words = self._get_stopwords()

        cleaned_tokens: list[str] = []
        for t in tokens:
            if t in stop_words:
                continue
            if len(t) < self.min_token_len:
                continue
            lemma = lemmatizer.lemmatize(t)
            cleaned_tokens.append(lemma)

        return " ".join(cleaned_tokens)

    def transform(self, X: Iterable[Any]) -> list[str]:
        """Transform an iterable of texts into preprocessed strings."""
        return [self.preprocess_text(item) for item in X]


def build_bow_vectorizer(
    ngram_range: tuple = NGRAM_RANGE,
    min_df: int = MIN_DF,
    max_features: int = MAX_FEATURES,
) -> CountVectorizer:
    """Build a CountVectorizer with project settings."""
    return CountVectorizer(
        ngram_range=ngram_range,
        min_df=min_df,
        max_features=max_features,
    )


def build_tfidf_vectorizer(
    ngram_range: tuple = NGRAM_RANGE,
    min_df: int = MIN_DF,
    max_df: float = MAX_DF,
    sublinear_tf: bool = SUBLINEAR_TF,
    max_features: int = MAX_FEATURES,
) -> TfidfVectorizer:
    """Build a TfidfVectorizer with project settings."""
    return TfidfVectorizer(
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=sublinear_tf,
        max_features=max_features,
    )
