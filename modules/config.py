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
DEFAULT_MAX_NGRAM = 2

DEFAULT_DATA_PATH = "data/WineEnthusiast-data/winemag-data-130k-v2.csv"
DEFAULT_OUTPUT_DIR = "outputs/classical"

EMBEDDING_MODELS = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "mpnet": "sentence-transformers/all-mpnet-base-v2",
}

EMBEDDING_CACHE_DIR = "outputs/advanced_embeddings/cache"

TFIDF_CONFIG = {
    "max_features": 50000,
    "ngram_range": (1, DEFAULT_MAX_NGRAM),
    "min_df": 2,
    "max_df": 0.95,
    "sublinear_tf": True,
}

COUNT_VECTORIZER_CONFIG = {
    "max_features": 50000,
    "ngram_range": (1, DEFAULT_MAX_NGRAM),
    "min_df": 2,
    "max_df": 0.95,
}

PREPOSITION_CONJUNCTION_STOP_WORDS = frozenset(
    [
        # Articles
        "a",
        "an",
        "the",
        # Pronouns
        "he",
        "her",
        "him",
        "his",
        "i",
        "it",
        "its",
        "me",
        "my",
        "our",
        "she",
        "them",
        "their",
        "they",
        "us",
        "we",
        "who",
        "what",
        "which",
        "you",
        "your",
        # Demonstratives
        "that",
        "these",
        "this",
        "those",
        # Auxiliary / copular verbs
        "am",
        "are",
        "be",
        "been",
        "being",
        "can",
        "could",
        "did",
        "do",
        "does",
        "had",
        "has",
        "have",
        "having",
        "is",
        "may",
        "might",
        "must",
        "shall",
        "should",
        "was",
        "were",
        "will",
        "would",
        # Prepositions (low-signal, minimal idiomatic sentiment use)
        "aboard",
        "about",
        "across",
        "after",
        "along",
        "alongside",
        "amid",
        "among",
        "around",
        "as",
        "at",
        "before",
        "beside",
        "besides",
        "between",
        "by",
        "during",
        "except",
        "for",
        "from",
        "if",
        "in",
        "inside",
        "into",
        "near",
        "of",
        "on",
        "onto",
        "or",
        "past",
        "since",
        "so",
        "than",
        "through",
        "throughout",
        "till",
        "to",
        "toward",
        "towards",
        "until",
        "upon",
        "versus",
        "via",
        "when",
        "where",
        "wherever",
        # Low-signal adverbs
        "also",
        "here",
        "just",
        "now",
        "there",
        "then",
        # Conjunctions (non-contrastive)
        "and",
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
