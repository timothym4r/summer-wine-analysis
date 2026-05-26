"""Data loading, validation, and splitting."""

from __future__ import annotations

from typing import Optional

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import RANDOM_SEED, TARGET_COL, TEST_SIZE, TEXT_COL, VALID_SIZE


def load_wine_data(
    path: str,
    text_col: str = TEXT_COL,
    target_col: str = TARGET_COL,
    drop_duplicate_descriptions: bool = True,
) -> pd.DataFrame:
    """Load and validate the Wine Enthusiast CSV."""
    df = pd.read_csv(path)
    missing = {text_col, target_col} - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.dropna(subset=[text_col, target_col]).copy()
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df = df.dropna(subset=[target_col])

    if drop_duplicate_descriptions:
        df = df.drop_duplicates(subset=[text_col], keep="first")

    return df.reset_index(drop=True)


def train_valid_test_split(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    label_col: Optional[str] = None,
    test_size: float = TEST_SIZE,
    valid_size: float = VALID_SIZE,
    random_state: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split into train/validation/test, stratifying by labels when provided."""
    stratify = df[label_col] if label_col else None
    train_valid, test = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    valid_fraction_of_remaining = valid_size / (1.0 - test_size)
    stratify_train_valid = train_valid[label_col] if label_col else None
    train, valid = train_test_split(
        train_valid,
        test_size=valid_fraction_of_remaining,
        random_state=random_state,
        stratify=stratify_train_valid,
    )

    return (
        train.reset_index(drop=True),
        valid.reset_index(drop=True),
        test.reset_index(drop=True),
    )

