"""Project defaults and model configuration."""

from __future__ import annotations

TEXT_COL = "description"
TARGET_COL = "points"
CLASS_COL = "rating_class"

RANDOM_SEED = 42
TEST_SIZE = 0.2
VALID_SIZE = 0.1
MIN_CLASS_COUNT = 50
N_CLASSIFICATION_BINS = 4

DEFAULT_DATA_PATH = "data/WineEnthusiast-data/winemag-data-130k-v2.csv"
DEFAULT_OUTPUT_DIR = "outputs"

EMBEDDING_MODELS = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "mpnet": "sentence-transformers/all-mpnet-base-v2",
}

EMBEDDING_CACHE_DIR = "outputs/advanced_embeddings/cache"

TFIDF_CONFIG = {
    "max_features": 50000,
    "ngram_range": (1, 2),
    "min_df": 2,
    "max_df": 0.95,
    "sublinear_tf": True,
}

COUNT_VECTORIZER_CONFIG = {
    "max_features": 50000,
    "ngram_range": (1, 2),
    "min_df": 2,
    "max_df": 0.95,
}

PREPOSITION_CONJUNCTION_STOP_WORDS = frozenset(
    [
        "aboard",
        "about",
        "above",
        "across",
        "after",
        "against",
        "along",
        "alongside",
        "although",
        "amid",
        "among",
        "and",
        "around",
        "as",
        "at",
        "because",
        "before",
        "behind",
        "below",
        "beneath",
        "beside",
        "besides",
        "between",
        "beyond",
        "but",
        "by",
        "despite",
        "down",
        "during",
        "except",
        "for",
        "from",
        "if",
        "in",
        "inside",
        "into",
        "like",
        "near",
        "of",
        "off",
        "on",
        "onto",
        "or",
        "out",
        "outside",
        "over",
        "past",
        "plus",
        "since",
        "so",
        "than",
        "through",
        "throughout",
        "till",
        "to",
        "toward",
        "towards",
        "under",
        "underneath",
        "unless",
        "unlike",
        "until",
        "up",
        "upon",
        "versus",
        "via",
        "when",
        "where",
        "whereas",
        "wherever",
        "whether",
        "while",
        "with",
        "within",
        "yet",
    ]
)

TEXT_STATS_CONFIG = {
    "preserve_punctuation": True,
}

CLASSIFICATION_MODEL_CONFIG = {
    "logistic_regression": {"max_iter": 1000, "class_weight": "balanced"},
    "linear_svc": {"class_weight": "balanced"},
    "decision_tree": {"max_depth": 30, "min_samples_leaf": 5},
    "random_forest": {"n_estimators": 200, "max_depth": 30, "n_jobs": -1},
    "hist_gradient_boosting": {"max_iter": 150, "learning_rate": 0.08},
}

REGRESSION_MODEL_CONFIG = {
    "ridge": {"alpha": 1.0},
    "linear_svr": {"C": 1.0, "max_iter": 5000},
    "decision_tree": {"max_depth": 30, "min_samples_leaf": 5},
    "random_forest": {"n_estimators": 200, "max_depth": 30, "n_jobs": -1},
    "hist_gradient_boosting": {"max_iter": 150, "learning_rate": 0.08},
}
