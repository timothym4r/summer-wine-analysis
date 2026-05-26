"""Prediction-level error analysis for selected baseline models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import confusion_matrix

from .config import CLASS_COL, RANDOM_SEED, TARGET_COL, TEXT_COL
from .data import train_valid_test_split
from .features import build_count_pipeline, build_text_stats_pipeline, build_tfidf_pipeline
from .utils import ensure_dir


def _build_regression_pipeline(feature_family: str) -> Any:
    if feature_family == "tfidf":
        return build_tfidf_pipeline(Ridge(alpha=1.0), task="regression")
    if feature_family == "count":
        return build_count_pipeline(Ridge(alpha=1.0), task="regression")
    if feature_family == "text_stats":
        return build_text_stats_pipeline(Ridge(alpha=1.0), task="regression")
    raise ValueError(f"Unsupported regression feature family: {feature_family}")


def _build_classification_pipeline(feature_family: str) -> Any:
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_SEED,
    )
    if feature_family == "tfidf":
        return build_tfidf_pipeline(model, task="classification")
    if feature_family == "count":
        return build_count_pipeline(model, task="classification")
    if feature_family == "text_stats":
        return build_text_stats_pipeline(model, task="classification")
    raise ValueError(f"Unsupported classification feature family: {feature_family}")


def save_regression_error_analysis(
    df: pd.DataFrame,
    output_dir: str | Path,
    feature_family: str = "tfidf",
    text_col: str = TEXT_COL,
    target_col: str = TARGET_COL,
) -> Path:
    """Fit one selected regressor and save row-level test predictions."""
    output_path = ensure_dir(Path(output_dir) / "error_analysis")
    train, valid, test = train_valid_test_split(
        df,
        target_col=target_col,
        label_col=None,
        random_state=RANDOM_SEED,
    )
    train_eval = pd.concat([train, valid], ignore_index=True)

    pipeline = _build_regression_pipeline(feature_family)
    pipeline.fit(train_eval, train_eval[target_col])
    predictions = pipeline.predict(test)

    result = test[[text_col, target_col]].copy()
    result["predicted_points"] = predictions
    result["error"] = result["predicted_points"] - result[target_col]
    result["absolute_error"] = result["error"].abs()
    result = result.sort_values("absolute_error", ascending=False)

    path = output_path / "regression_predictions.csv"
    result.to_csv(path, index=False)
    return path


def save_classification_error_analysis(
    df: pd.DataFrame,
    output_dir: str | Path,
    feature_family: str = "tfidf",
    text_col: str = TEXT_COL,
    target_col: str = TARGET_COL,
    class_col: str = CLASS_COL,
) -> dict[str, Path]:
    """Fit one selected classifier and save predictions plus confusion matrix."""
    output_path = ensure_dir(Path(output_dir) / "error_analysis")
    train, valid, test = train_valid_test_split(
        df,
        target_col=target_col,
        label_col=class_col,
        random_state=RANDOM_SEED,
    )
    train_eval = pd.concat([train, valid], ignore_index=True)

    labels = (
        list(df[class_col].cat.categories)
        if hasattr(df[class_col], "cat")
        else sorted(df[class_col].astype(str).unique().tolist())
    )
    label_to_index = {label: index for index, label in enumerate(labels)}

    pipeline = _build_classification_pipeline(feature_family)
    pipeline.fit(train_eval, train_eval[class_col])
    predictions = pipeline.predict(test)

    result = test[[text_col, target_col, class_col]].copy()
    result["predicted_class"] = predictions
    result["class_error"] = (
        result["predicted_class"].astype(str).map(label_to_index)
        - result[class_col].astype(str).map(label_to_index)
    )
    result["absolute_class_error"] = result["class_error"].abs()
    result = result.sort_values("absolute_class_error", ascending=False)

    predictions_path = output_path / "classification_predictions.csv"
    result.to_csv(predictions_path, index=False)

    matrix = confusion_matrix(test[class_col].astype(str), predictions.astype(str), labels=labels)
    confusion_df = pd.DataFrame(matrix, index=labels, columns=labels)
    confusion_df.index.name = "true_class"
    confusion_path = output_path / "classification_confusion_matrix.csv"
    confusion_df.to_csv(confusion_path)

    return {
        "predictions": predictions_path,
        "confusion_matrix": confusion_path,
    }

