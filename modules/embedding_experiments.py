"""Stage 1 advanced models: frozen sentence embeddings plus prediction heads."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, LinearSVR

from .binning import make_rating_bins
from .config import (
    CLASS_COL,
    EMBEDDING_CACHE_DIR,
    EMBEDDING_MODELS,
    RANDOM_SEED,
    TARGET_COL,
    TEXT_COL,
)
from .data import train_valid_test_split
from .evaluation import evaluate_classification, evaluate_regression
from .transformer_models import get_cached_sentence_embeddings, slugify_model_name
from .utils import ensure_dir, progress_iter


def get_embedding_regression_heads(include_nn: bool = True) -> dict[str, Any]:
    """Prediction heads for frozen embedding regression."""
    heads: dict[str, Any] = {
        "dummy_mean": DummyRegressor(strategy="mean"),
        "ridge": Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=1.0))]),
        "linear_svr": Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", LinearSVR(C=1.0, max_iter=5000, random_state=RANDOM_SEED)),
            ]
        ),
    }
    if include_nn:
        heads["mlp"] = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    MLPRegressor(
                        hidden_layer_sizes=(256, 64),
                        activation="relu",
                        alpha=1e-4,
                        early_stopping=True,
                        max_iter=100,
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        )
    return heads


def get_embedding_classification_heads(include_nn: bool = True) -> dict[str, Any]:
    """Prediction heads for frozen embedding classification."""
    heads: dict[str, Any] = {
        "dummy_most_frequent": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        ),
        "linear_svc": Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", LinearSVC(class_weight="balanced", random_state=RANDOM_SEED)),
            ]
        ),
    }
    if include_nn:
        heads["mlp"] = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    MLPClassifier(
                        hidden_layer_sizes=(256, 64),
                        activation="relu",
                        alpha=1e-4,
                        early_stopping=True,
                        max_iter=100,
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        )
    return heads


def _prepare_embedding_frame(
    df: pd.DataFrame,
    sample_size: Optional[int] = None,
) -> pd.DataFrame:
    result = df.copy().reset_index(drop=True)
    if sample_size is not None and sample_size < len(result):
        result = result.sample(sample_size, random_state=RANDOM_SEED).reset_index(drop=True)
    result["_embedding_row_id"] = np.arange(len(result))
    return result


def _split_embedding_arrays(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
    embeddings: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        embeddings[train["_embedding_row_id"].to_numpy()],
        embeddings[valid["_embedding_row_id"].to_numpy()],
        embeddings[test["_embedding_row_id"].to_numpy()],
    )


def _scalar_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in metrics.items()
        if isinstance(value, (int, float, str)) or value is None
    }


def _save_regression_predictions(
    test: pd.DataFrame,
    predictions: np.ndarray,
    path: Path,
    text_col: str,
    target_col: str,
) -> None:
    result = test[[text_col, target_col]].copy()
    result["predicted_points"] = predictions
    result["error"] = result["predicted_points"] - result[target_col]
    result["absolute_error"] = result["error"].abs()
    result.sort_values("absolute_error", ascending=False).to_csv(path, index=False)


def _save_classification_predictions(
    test: pd.DataFrame,
    predictions: np.ndarray,
    labels: list[str],
    predictions_path: Path,
    confusion_path: Path,
    text_col: str,
    target_col: str,
    class_col: str,
) -> None:
    label_to_index = {label: index for index, label in enumerate(labels)}
    result = test[[text_col, target_col, class_col]].copy()
    result["predicted_class"] = predictions
    result["class_error"] = (
        result["predicted_class"].astype(str).map(label_to_index)
        - result[class_col].astype(str).map(label_to_index)
    )
    result["absolute_class_error"] = result["class_error"].abs()
    result.sort_values("absolute_class_error", ascending=False).to_csv(
        predictions_path,
        index=False,
    )

    matrix = confusion_matrix(test[class_col].astype(str), predictions.astype(str), labels=labels)
    confusion_df = pd.DataFrame(matrix, index=labels, columns=labels)
    confusion_df.index.name = "true_class"
    confusion_df.to_csv(confusion_path)


def run_embedding_stage1_experiments(
    df: pd.DataFrame,
    output_dir: str | Path = "outputs/advanced_embeddings",
    text_col: str = TEXT_COL,
    target_col: str = TARGET_COL,
    embedding_models: Optional[dict[str, str]] = None,
    include_nn: bool = True,
    sample_size: Optional[int] = None,
    batch_size: int = 64,
) -> dict[str, pd.DataFrame]:
    """Run frozen sentence-embedding baselines for regression and classification.

    Outputs are written under ``output_dir`` and do not overwrite the classical
    baseline artifacts in ``outputs/``.
    """
    output_path = ensure_dir(output_dir)
    results_path = ensure_dir(output_path / "results")
    error_path = ensure_dir(output_path / "error_analysis")
    summary_path = ensure_dir(output_path / "summary")
    cache_dir = output_path / "cache"

    frame = _prepare_embedding_frame(df, sample_size=sample_size)
    binned_frame, bin_metadata = make_rating_bins(frame, target_col=target_col)
    labels = list(binned_frame[CLASS_COL].cat.categories)

    regression_rows: list[dict[str, Any]] = []
    classification_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    models = embedding_models or EMBEDDING_MODELS
    for short_name, model_name in progress_iter(models.items(), desc="Embedding models"):
        model_slug = slugify_model_name(model_name)
        embeddings = get_cached_sentence_embeddings(
            frame[text_col].fillna("").astype(str).tolist(),
            model_name=model_name,
            cache_dir=cache_dir or EMBEDDING_CACHE_DIR,
            batch_size=batch_size,
        )

        train_reg, valid_reg, test_reg = train_valid_test_split(
            frame,
            target_col=target_col,
            label_col=None,
            random_state=RANDOM_SEED,
        )
        x_train_reg, x_valid_reg, x_test_reg = _split_embedding_arrays(
            train_reg,
            valid_reg,
            test_reg,
            embeddings,
        )
        x_train_reg = np.vstack([x_train_reg, x_valid_reg])
        y_train_reg = pd.concat([train_reg[target_col], valid_reg[target_col]])

        for head_name, head in progress_iter(
            get_embedding_regression_heads(include_nn=include_nn).items(),
            desc=f"{short_name} regression heads",
        ):
            model = clone(head)
            model.fit(x_train_reg, y_train_reg)
            predictions = model.predict(x_test_reg)
            metrics = evaluate_regression(test_reg[target_col], predictions)
            row = {
                "task": "regression",
                "feature_family": "sentence_embedding",
                "embedding_short_name": short_name,
                "embedding_model": model_name,
                "head": head_name,
                "embedding_dim": int(embeddings.shape[1]),
                "sample_size": len(frame),
                **metrics,
            }
            regression_rows.append(row)

            prediction_path = (
                error_path / f"{model_slug}_{head_name}_regression_predictions.csv"
            )
            _save_regression_predictions(
                test_reg,
                predictions,
                prediction_path,
                text_col=text_col,
                target_col=target_col,
            )

        train_clf, valid_clf, test_clf = train_valid_test_split(
            binned_frame,
            target_col=target_col,
            label_col=CLASS_COL,
            random_state=RANDOM_SEED,
        )
        x_train_clf, x_valid_clf, x_test_clf = _split_embedding_arrays(
            train_clf,
            valid_clf,
            test_clf,
            embeddings,
        )
        x_train_clf = np.vstack([x_train_clf, x_valid_clf])
        y_train_clf = pd.concat([train_clf[CLASS_COL], valid_clf[CLASS_COL]])
        label_to_index = {label: index for index, label in enumerate(labels)}

        for head_name, head in progress_iter(
            get_embedding_classification_heads(include_nn=include_nn).items(),
            desc=f"{short_name} classification heads",
        ):
            model = clone(head)
            if head_name == "mlp":
                y_fit = y_train_clf.astype(str).map(label_to_index).to_numpy()
                model.fit(x_train_clf, y_fit)
                prediction_codes = model.predict(x_test_clf)
                predictions = np.asarray([labels[int(code)] for code in prediction_codes])
            else:
                model.fit(x_train_clf, y_train_clf)
                predictions = model.predict(x_test_clf)
            metrics = evaluate_classification(test_clf[CLASS_COL], predictions, labels=labels)
            row = {
                "task": "classification",
                "feature_family": "sentence_embedding",
                "embedding_short_name": short_name,
                "embedding_model": model_name,
                "head": head_name,
                "embedding_dim": int(embeddings.shape[1]),
                "sample_size": len(frame),
                **_scalar_metrics(metrics),
            }
            classification_rows.append(row)

            prediction_path = (
                error_path / f"{model_slug}_{head_name}_classification_predictions.csv"
            )
            confusion_path = (
                error_path / f"{model_slug}_{head_name}_classification_confusion_matrix.csv"
            )
            _save_classification_predictions(
                test_clf,
                predictions,
                labels,
                prediction_path,
                confusion_path,
                text_col=text_col,
                target_col=target_col,
                class_col=CLASS_COL,
            )

        summary_rows.append(
            {
                "embedding_short_name": short_name,
                "embedding_model": model_name,
                "embedding_dim": int(embeddings.shape[1]),
                "sample_size": len(frame),
                "interpretability_note": (
                    "Frozen sentence embeddings are less directly interpretable than "
                    "TF-IDF/count coefficients. We save prediction-level errors and "
                    "confusion matrices; use these with nearest examples or follow-up "
                    "LLM attribute extraction for richer explanations."
                ),
            }
        )

    regression_results = pd.DataFrame(regression_rows)
    classification_results = pd.DataFrame(classification_rows)
    interpretability_summary = pd.DataFrame(summary_rows)

    regression_results.to_csv(results_path / "embedding_regression_results.csv", index=False)
    classification_results.to_csv(
        results_path / "embedding_classification_results.csv",
        index=False,
    )
    interpretability_summary.to_csv(
        summary_path / "embedding_interpretability_summary.csv",
        index=False,
    )
    with (summary_path / "embedding_bin_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(bin_metadata, f, indent=2, default=str)

    return {
        "regression": regression_results,
        "classification": classification_results,
        "interpretability": interpretability_summary,
    }
