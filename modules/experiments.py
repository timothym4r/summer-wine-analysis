"""High-level experiment runners."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .binning import make_rating_bins
from .config import CLASS_COL, RANDOM_SEED, TARGET_COL, TEXT_COL
from .data import train_valid_test_split
from .evaluation import evaluate_classification, evaluate_regression
from .features import build_count_pipeline, build_text_stats_pipeline, build_tfidf_pipeline
from .models import get_classification_models, get_regression_models
from .utils import progress_iter

TFIDF_CLASSIFICATION_MODELS = {"dummy_most_frequent", "logistic_regression", "linear_svc"}
TFIDF_REGRESSION_MODELS = {"dummy_mean", "ridge", "linear_svr"}
COUNT_CLASSIFICATION_MODELS = {"logistic_regression", "linear_svc"}
COUNT_REGRESSION_MODELS = {"ridge", "linear_svr"}
INDICATOR_UNIGRAM_CLASSIFICATION_MODELS = {"decision_tree"}
INDICATOR_UNIGRAM_REGRESSION_MODELS = {"decision_tree"}


def _flatten_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in metrics.items()
        if isinstance(value, (int, float, str)) or value is None
    }


def run_regression_experiments(
    df: pd.DataFrame,
    text_col: str = TEXT_COL,
    target_col: str = TARGET_COL,
    run_embeddings: bool = False,
    run_finetuning: bool = False,
    run_llm_features: bool = False,
    run_few_shot_llm: bool = False,
) -> pd.DataFrame:
    """Run classical regression baselines and optional extension points."""
    train, valid, test = train_valid_test_split(
        df,
        target_col=target_col,
        label_col=None,
        random_state=RANDOM_SEED,
    )
    train_eval = pd.concat([train, valid], ignore_index=True)

    rows: list[dict[str, Any]] = []
    models = get_regression_models()
    for model_name, model in progress_iter(models.items(), desc="Regression TF-IDF"):
        if model_name not in TFIDF_REGRESSION_MODELS:
            continue
        feature_family = "dummy" if model_name.startswith("dummy") else "tfidf"
        pipeline = (
            model
            if feature_family == "dummy"
            else build_tfidf_pipeline(model, task="regression", text_col=text_col)
        )
        x_train = train_eval if feature_family != "dummy" else train_eval[[target_col]]
        x_test = test if feature_family != "dummy" else test[[target_col]]
        pipeline.fit(x_train, train_eval[target_col])
        pred = pipeline.predict(x_test)
        rows.append(
            {
                "task": "regression",
                "feature_family": feature_family,
                "model": model_name,
                **evaluate_regression(test[target_col], pred),
            }
        )

    for model_name, model in progress_iter(models.items(), desc="Regression counts"):
        if model_name not in COUNT_REGRESSION_MODELS:
            continue
        pipeline = build_count_pipeline(model, task="regression", text_col=text_col)
        pipeline.fit(train_eval, train_eval[target_col])
        pred = pipeline.predict(test)
        rows.append(
            {
                "task": "regression",
                "feature_family": "count",
                "model": model_name,
                **evaluate_regression(test[target_col], pred),
            }
        )

    for model_name, model in progress_iter(
        models.items(),
        desc="Regression unigram indicator trees",
    ):
        if model_name not in INDICATOR_UNIGRAM_REGRESSION_MODELS:
            continue
        pipeline = build_count_pipeline(
            model,
            task="regression",
            text_col=text_col,
            ngram_range=(1, 1),
            binary=True,
        )
        pipeline.fit(train_eval, train_eval[target_col])
        pred = pipeline.predict(test)
        rows.append(
            {
                "task": "regression",
                "feature_family": "indicator_unigram",
                "model": model_name,
                **evaluate_regression(test[target_col], pred),
            }
        )

    for model_name, model in progress_iter(
        get_regression_models().items(),
        desc="Regression text stats",
    ):
        if model_name.startswith("dummy"):
            continue
        pipeline = build_text_stats_pipeline(model, task="regression", text_col=text_col)
        pipeline.fit(train_eval, train_eval[target_col])
        pred = pipeline.predict(test)
        rows.append(
            {
                "task": "regression",
                "feature_family": "text_stats",
                "model": model_name,
                **evaluate_regression(test[target_col], pred),
            }
        )

    if run_embeddings or run_finetuning or run_llm_features or run_few_shot_llm:
        rows.append(
            {
                "task": "regression",
                "feature_family": "optional",
                "model": "optional_methods_not_run_in_baseline",
                "mae": None,
                "rmse": None,
                "r2": None,
            }
        )

    return pd.DataFrame(rows)


def run_classification_experiments(
    df: pd.DataFrame,
    text_col: str = TEXT_COL,
    target_col: str = TARGET_COL,
    bin_strategy: str = "quantile",
    run_embeddings: bool = False,
    run_finetuning: bool = False,
    run_llm_features: bool = False,
    run_few_shot_llm: bool = False,
) -> pd.DataFrame:
    """Run classical classification baselines and optional extension points."""
    if CLASS_COL not in df.columns:
        df, _ = make_rating_bins(df, target_col=target_col, strategy=bin_strategy)

    train, valid, test = train_valid_test_split(
        df,
        target_col=target_col,
        label_col=CLASS_COL,
        random_state=RANDOM_SEED,
    )
    train_eval = pd.concat([train, valid], ignore_index=True)
    labels = list(df[CLASS_COL].cat.categories) if hasattr(df[CLASS_COL], "cat") else None

    rows: list[dict[str, Any]] = []
    models = get_classification_models()
    for model_name, model in progress_iter(models.items(), desc="Classification TF-IDF"):
        if model_name not in TFIDF_CLASSIFICATION_MODELS:
            continue
        feature_family = "dummy" if model_name.startswith("dummy") else "tfidf"
        pipeline = (
            model
            if feature_family == "dummy"
            else build_tfidf_pipeline(model, task="classification", text_col=text_col)
        )
        x_train = train_eval if feature_family != "dummy" else train_eval[[target_col]]
        x_test = test if feature_family != "dummy" else test[[target_col]]
        pipeline.fit(x_train, train_eval[CLASS_COL])
        pred = pipeline.predict(x_test)
        rows.append(
            {
                "task": "classification",
                "feature_family": feature_family,
                "model": model_name,
                **_flatten_metrics(evaluate_classification(test[CLASS_COL], pred, labels=labels)),
            }
        )

    for model_name, model in progress_iter(models.items(), desc="Classification counts"):
        if model_name not in COUNT_CLASSIFICATION_MODELS:
            continue
        pipeline = build_count_pipeline(model, task="classification", text_col=text_col)
        pipeline.fit(train_eval, train_eval[CLASS_COL])
        pred = pipeline.predict(test)
        rows.append(
            {
                "task": "classification",
                "feature_family": "count",
                "model": model_name,
                **_flatten_metrics(evaluate_classification(test[CLASS_COL], pred, labels=labels)),
            }
        )

    for model_name, model in progress_iter(
        models.items(),
        desc="Classification unigram indicator trees",
    ):
        if model_name not in INDICATOR_UNIGRAM_CLASSIFICATION_MODELS:
            continue
        pipeline = build_count_pipeline(
            model,
            task="classification",
            text_col=text_col,
            ngram_range=(1, 1),
            binary=True,
        )
        pipeline.fit(train_eval, train_eval[CLASS_COL])
        pred = pipeline.predict(test)
        rows.append(
            {
                "task": "classification",
                "feature_family": "indicator_unigram",
                "model": model_name,
                **_flatten_metrics(evaluate_classification(test[CLASS_COL], pred, labels=labels)),
            }
        )

    for model_name, model in progress_iter(
        get_classification_models().items(),
        desc="Classification text stats",
    ):
        if model_name.startswith("dummy"):
            continue
        pipeline = build_text_stats_pipeline(model, task="classification", text_col=text_col)
        pipeline.fit(train_eval, train_eval[CLASS_COL])
        pred = pipeline.predict(test)
        rows.append(
            {
                "task": "classification",
                "feature_family": "text_stats",
                "model": model_name,
                **_flatten_metrics(evaluate_classification(test[CLASS_COL], pred, labels=labels)),
            }
        )

    if run_embeddings or run_finetuning or run_llm_features or run_few_shot_llm:
        rows.append(
            {
                "task": "classification",
                "feature_family": "optional",
                "model": "optional_methods_not_run_in_baseline",
            }
        )

    return pd.DataFrame(rows)
