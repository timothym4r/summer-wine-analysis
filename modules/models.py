"""Classical models used in baseline experiments."""

from __future__ import annotations

from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from .config import (
    CLASSIFICATION_MODEL_CONFIG,
    RANDOM_SEED,
    REGRESSION_MODEL_CONFIG,
)


def get_classification_models() -> dict[str, object]:
    """Return sensible, not-too-slow classification models."""
    return {
        "dummy_most_frequent": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": LogisticRegression(
            random_state=RANDOM_SEED,
            **CLASSIFICATION_MODEL_CONFIG["logistic_regression"],
        ),
        "linear_svc": LinearSVC(
            random_state=RANDOM_SEED,
            **CLASSIFICATION_MODEL_CONFIG["linear_svc"],
        ),
        "decision_tree": DecisionTreeClassifier(
            random_state=RANDOM_SEED,
            **CLASSIFICATION_MODEL_CONFIG["decision_tree"],
        ),
        "random_forest": RandomForestClassifier(
            random_state=RANDOM_SEED,
            **CLASSIFICATION_MODEL_CONFIG["random_forest"],
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            random_state=RANDOM_SEED,
            **CLASSIFICATION_MODEL_CONFIG["hist_gradient_boosting"],
        ),
    }


def get_regression_models() -> dict[str, object]:
    """Return sensible, not-too-slow regression models."""
    return {
        "dummy_mean": DummyRegressor(strategy="mean"),
        "ridge": Ridge(**REGRESSION_MODEL_CONFIG["ridge"]),
        "linear_svr": LinearSVR(
            random_state=RANDOM_SEED,
            **REGRESSION_MODEL_CONFIG["linear_svr"],
        ),
        "decision_tree": DecisionTreeRegressor(
            random_state=RANDOM_SEED,
            **REGRESSION_MODEL_CONFIG["decision_tree"],
        ),
        "random_forest": RandomForestRegressor(
            random_state=RANDOM_SEED,
            **REGRESSION_MODEL_CONFIG["random_forest"],
        ),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            random_state=RANDOM_SEED,
            **REGRESSION_MODEL_CONFIG["hist_gradient_boosting"],
        ),
    }
