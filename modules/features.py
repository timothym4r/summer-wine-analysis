"""Feature builders and scikit-learn pipelines."""

from __future__ import annotations

import re
import string
from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from .config import COUNT_VECTORIZER_CONFIG, TEXT_COL, TFIDF_CONFIG


def _sentence_count(text: str) -> int:
    return max(1, len(re.findall(r"[.!?]+", text)))


def build_text_stats_features(
    df: pd.DataFrame,
    text_col: str = TEXT_COL,
) -> pd.DataFrame:
    """Build simple numeric features from review text."""
    texts = df[text_col].fillna("").astype(str)
    word_lists = texts.str.findall(r"\b\w+\b")
    word_counts = word_lists.map(len)
    char_counts = texts.str.len()

    return pd.DataFrame(
        {
            "char_count": char_counts,
            "word_count": word_counts,
            "avg_word_length": [
                sum(map(len, words)) / len(words) if words else 0.0
                for words in word_lists
            ],
            "sentence_count": texts.map(_sentence_count),
            "comma_count": texts.str.count(","),
            "period_count": texts.str.count(r"\."),
            "exclamation_count": texts.str.count("!"),
            "question_count": texts.str.count(r"\?"),
            "semicolon_count": texts.str.count(";"),
            "punctuation_count": texts.map(
                lambda text: sum(1 for char in text if char in string.punctuation)
            ),
        }
    )


def _extract_text_column(x: Any, text_col: str = TEXT_COL) -> pd.Series:
    if isinstance(x, pd.DataFrame):
        return x[text_col].fillna("").astype(str)
    return pd.Series(x).fillna("").astype(str)


class TextStatsTransformer(BaseEstimator, TransformerMixin):
    """Scikit-learn transformer for text statistics."""

    def __init__(self, text_col: str = TEXT_COL):
        self.text_col = text_col

    def fit(self, x: pd.DataFrame, y: Any = None) -> "TextStatsTransformer":
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        return build_text_stats_features(x, self.text_col)


def build_tfidf_vectorizer(**kwargs: Any) -> TfidfVectorizer:
    """Create a TF-IDF vectorizer with project defaults."""
    config = {**TFIDF_CONFIG, **kwargs}
    return TfidfVectorizer(**config)


def build_count_vectorizer(**kwargs: Any) -> CountVectorizer:
    """Create a bag-of-words/count vectorizer with project defaults."""
    config = {**COUNT_VECTORIZER_CONFIG, **kwargs}
    return CountVectorizer(**config)


def build_count_pipeline(
    model: Any,
    task: str = "classification",
    text_col: str = TEXT_COL,
    **count_kwargs: Any,
) -> Pipeline:
    """Build a text-column extraction + bag-of-words counts + model pipeline."""
    return Pipeline(
        [
            ("text", FunctionTransformer(_extract_text_column, kw_args={"text_col": text_col})),
            ("count", build_count_vectorizer(**count_kwargs)),
            ("model", model),
        ]
    )


def build_tfidf_pipeline(
    model: Any,
    task: str = "classification",
    text_col: str = TEXT_COL,
    **tfidf_kwargs: Any,
) -> Pipeline:
    """Build a text-column extraction + TF-IDF + model pipeline."""
    return Pipeline(
        [
            ("text", FunctionTransformer(_extract_text_column, kw_args={"text_col": text_col})),
            ("tfidf", build_tfidf_vectorizer(**tfidf_kwargs)),
            ("model", model),
        ]
    )


def build_text_stats_pipeline(
    model: Any,
    task: str = "classification",
    text_col: str = TEXT_COL,
) -> Pipeline:
    """Build a text-statistics baseline pipeline."""
    return Pipeline(
        [
            ("text_stats", TextStatsTransformer(text_col=text_col)),
            ("scale", StandardScaler()),
            ("model", model),
        ]
    )
