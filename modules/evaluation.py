"""Evaluation helpers for classification and regression."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    cohen_kappa_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def _ordered_class_indices(values: Any, labels: Optional[list[str]]) -> np.ndarray:
    if labels is None:
        labels = list(pd.Series(values).astype(str).unique())
    mapping = {label: idx for idx, label in enumerate(labels)}
    return pd.Series(values).astype(str).map(mapping).to_numpy()


def evaluate_classification(
    y_true: Any,
    y_pred: Any,
    labels: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Evaluate ordered classification predictions."""
    y_true_s = pd.Series(y_true).astype(str)
    y_pred_s = pd.Series(y_pred).astype(str)
    labels = labels or sorted(y_true_s.unique().tolist())

    true_idx = _ordered_class_indices(y_true_s, labels)
    pred_idx = _ordered_class_indices(y_pred_s, labels)
    valid_ordered = ~pd.isna(true_idx) & ~pd.isna(pred_idx)

    metrics: dict[str, Any] = {
        "accuracy": accuracy_score(y_true_s, y_pred_s),
        "macro_f1": f1_score(y_true_s, y_pred_s, labels=labels, average="macro", zero_division=0),
        "weighted_f1": f1_score(
            y_true_s,
            y_pred_s,
            labels=labels,
            average="weighted",
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_true_s, y_pred_s, labels=labels).tolist(),
        "labels": labels,
    }

    if valid_ordered.any():
        metrics["mean_absolute_class_error"] = float(
            np.mean(np.abs(true_idx[valid_ordered] - pred_idx[valid_ordered]))
        )
        metrics["quadratic_weighted_kappa"] = float(
            cohen_kappa_score(
                true_idx[valid_ordered],
                pred_idx[valid_ordered],
                weights="quadratic",
            )
        )
    else:
        metrics["mean_absolute_class_error"] = None
        metrics["quadratic_weighted_kappa"] = None

    return metrics


def evaluate_regression(y_true: Any, y_pred: Any) -> dict[str, float]:
    """Evaluate numeric rating predictions."""
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def print_metrics(metrics: dict[str, Any]) -> None:
    """Print scalar metrics in a compact format."""
    for key, value in metrics.items():
        if isinstance(value, (int, float, str)) or value is None:
            print(f"{key}: {value}")


def save_results_csv(results: pd.DataFrame, path: str | Path) -> None:
    """Save experiment results to CSV."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(path, index=False)


def save_results_json(results: dict[str, Any], path: str | Path) -> None:
    """Save experiment results to JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
