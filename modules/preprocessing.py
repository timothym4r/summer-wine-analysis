"""Text cleaning for wine review descriptions."""

from __future__ import annotations

import re
import string

import pandas as pd


def clean_text(
    text: str,
    lowercase: bool = False,
    normalize_whitespace: bool = True,
    preserve_punctuation: bool = True,
) -> str:
    """Clean review text without destroying wine-description signal."""
    if text is None:
        return ""

    cleaned = str(text)
    if lowercase:
        cleaned = cleaned.lower()
    if not preserve_punctuation:
        cleaned = cleaned.translate(str.maketrans("", "", string.punctuation))
    if normalize_whitespace:
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

    return cleaned


def preprocess_dataframe(
    df: pd.DataFrame,
    text_col: str = "description",
    lowercase: bool = False,
    preserve_punctuation: bool = True,
) -> pd.DataFrame:
    """Clean text and remove rows that become empty after cleaning."""
    result = df.copy()
    result[text_col] = result[text_col].map(
        lambda value: clean_text(
            value,
            lowercase=lowercase,
            preserve_punctuation=preserve_punctuation,
        )
    )
    result = result[result[text_col].str.len() > 0].reset_index(drop=True)
    return result

