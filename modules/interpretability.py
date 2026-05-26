"""Feature-importance reports for interpretable baseline models."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge

from .config import CLASS_COL, RANDOM_SEED, TARGET_COL, TEXT_COL
from .data import train_valid_test_split
from .features import (
    TextStatsTransformer,
    build_count_pipeline,
    build_text_stats_pipeline,
    build_tfidf_pipeline,
)
from .utils import ensure_dir


TEXT_STATS_FEATURE_NAMES = [
    "char_count",
    "word_count",
    "avg_word_length",
    "sentence_count",
    "comma_count",
    "period_count",
    "exclamation_count",
    "question_count",
    "semicolon_count",
    "punctuation_count",
]


def _top_indices(values: np.ndarray, top_n: int) -> tuple[np.ndarray, np.ndarray]:
    positive = np.argsort(values)[-top_n:][::-1]
    negative = np.argsort(values)[:top_n]
    return positive, negative


def _linear_coefficients_to_frame(
    feature_names: Iterable[str],
    coefficients: np.ndarray,
    model_name: str,
    feature_family: str,
    task: str,
    class_labels: Optional[list[str]] = None,
    top_n: int = 50,
) -> pd.DataFrame:
    feature_names_array = np.asarray(list(feature_names))
    rows: list[dict[str, Any]] = []

    if coefficients.ndim == 1:
        positive, negative = _top_indices(coefficients, top_n)
        for direction, indices in [("positive", positive), ("negative", negative)]:
            for rank, idx in enumerate(indices, start=1):
                rows.append(
                    {
                        "task": task,
                        "feature_family": feature_family,
                        "model": model_name,
                        "class_label": None,
                        "direction": direction,
                        "rank": rank,
                        "feature": feature_names_array[idx],
                        "coefficient": float(coefficients[idx]),
                        "absolute_coefficient": float(abs(coefficients[idx])),
                    }
                )
        return pd.DataFrame(rows)

    labels = class_labels or [f"class_{idx}" for idx in range(coefficients.shape[0])]
    for class_idx, label in enumerate(labels):
        class_coefficients = coefficients[class_idx]
        positive, negative = _top_indices(class_coefficients, top_n)
        for direction, indices in [("positive", positive), ("negative", negative)]:
            for rank, idx in enumerate(indices, start=1):
                rows.append(
                    {
                        "task": task,
                        "feature_family": feature_family,
                        "model": model_name,
                        "class_label": label,
                        "direction": direction,
                        "rank": rank,
                        "feature": feature_names_array[idx],
                        "coefficient": float(class_coefficients[idx]),
                        "absolute_coefficient": float(abs(class_coefficients[idx])),
                    }
                )

    return pd.DataFrame(rows)


def extract_pipeline_feature_importance(
    pipeline: Any,
    model_name: str,
    feature_family: str,
    task: str,
    class_labels: Optional[list[str]] = None,
    top_n: int = 50,
) -> pd.DataFrame:
    """Extract coefficient-based feature importance from a fitted pipeline."""
    model = pipeline.named_steps["model"]
    if not hasattr(model, "coef_"):
        raise ValueError(f"Model {model_name} does not expose coef_.")

    if "tfidf" in pipeline.named_steps:
        feature_names = pipeline.named_steps["tfidf"].get_feature_names_out()
    elif "count" in pipeline.named_steps:
        feature_names = pipeline.named_steps["count"].get_feature_names_out()
    elif isinstance(pipeline.named_steps.get("text_stats"), TextStatsTransformer):
        feature_names = TEXT_STATS_FEATURE_NAMES
    else:
        raise ValueError("Could not infer feature names from pipeline.")

    coefficients = np.asarray(model.coef_)
    if coefficients.ndim == 2 and coefficients.shape[0] == 1:
        coefficients = coefficients.ravel()

    return _linear_coefficients_to_frame(
        feature_names=feature_names,
        coefficients=coefficients,
        model_name=model_name,
        feature_family=feature_family,
        task=task,
        class_labels=class_labels,
        top_n=top_n,
    )


def save_interpretability_reports(
    df: pd.DataFrame,
    binned_df: pd.DataFrame,
    output_dir: str | Path,
    text_col: str = TEXT_COL,
    target_col: str = TARGET_COL,
    class_col: str = CLASS_COL,
    top_n: int = 50,
) -> dict[str, str]:
    """Fit selected interpretable models and save top feature reports.

    Reports are coefficient-based, so they are best read as directional
    associations learned by the model, not causal explanations.
    """
    output_path = ensure_dir(Path(output_dir) / "interpretability")
    saved_paths: dict[str, str] = {}

    train_reg, valid_reg, _ = train_valid_test_split(
        df,
        target_col=target_col,
        label_col=None,
        random_state=RANDOM_SEED,
    )
    train_reg = pd.concat([train_reg, valid_reg], ignore_index=True)

    regression_specs = [
        ("count_ridge", "count", build_count_pipeline(Ridge(alpha=1.0), "regression", text_col)),
        ("tfidf_ridge", "tfidf", build_tfidf_pipeline(Ridge(alpha=1.0), "regression", text_col)),
        ("text_stats_ridge", "text_stats", build_text_stats_pipeline(Ridge(alpha=1.0), "regression", text_col)),
    ]
    regression_reports = []
    for model_name, feature_family, pipeline in regression_specs:
        pipeline.fit(train_reg, train_reg[target_col])
        regression_reports.append(
            extract_pipeline_feature_importance(
                pipeline,
                model_name=model_name,
                feature_family=feature_family,
                task="regression",
                top_n=top_n,
            )
        )

    regression_path = output_path / "regression_feature_importance.csv"
    pd.concat(regression_reports, ignore_index=True).to_csv(regression_path, index=False)
    saved_paths["regression_feature_importance"] = str(regression_path)

    train_clf, valid_clf, _ = train_valid_test_split(
        binned_df,
        target_col=target_col,
        label_col=class_col,
        random_state=RANDOM_SEED,
    )
    train_clf = pd.concat([train_clf, valid_clf], ignore_index=True)
    class_labels = (
        list(binned_df[class_col].cat.categories)
        if hasattr(binned_df[class_col], "cat")
        else sorted(binned_df[class_col].astype(str).unique().tolist())
    )

    clf_model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_SEED,
    )
    classification_specs = [
        ("count_logistic_regression", "count", build_count_pipeline(clf_model, "classification", text_col)),
        (
            "tfidf_logistic_regression",
            "tfidf",
            build_tfidf_pipeline(
                LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED),
                "classification",
                text_col,
            ),
        ),
        (
            "text_stats_logistic_regression",
            "text_stats",
            build_text_stats_pipeline(
                LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED),
                "classification",
                text_col,
            ),
        ),
    ]

    classification_reports = []
    for model_name, feature_family, pipeline in classification_specs:
        pipeline.fit(train_clf, train_clf[class_col])
        classification_reports.append(
            extract_pipeline_feature_importance(
                pipeline,
                model_name=model_name,
                feature_family=feature_family,
                task="classification",
                class_labels=class_labels,
                top_n=top_n,
            )
        )

    classification_path = output_path / "classification_feature_importance.csv"
    pd.concat(classification_reports, ignore_index=True).to_csv(classification_path, index=False)
    saved_paths["classification_feature_importance"] = str(classification_path)

    return saved_paths

