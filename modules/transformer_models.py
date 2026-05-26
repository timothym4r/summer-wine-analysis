"""Optional transformer and sentence-embedding models."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Literal, Optional

import numpy as np

from .config import EMBEDDING_CACHE_DIR
from .utils import ensure_dir


def _require_sentence_transformers() -> Any:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "Install sentence-transformers to use embeddings: "
            "pip install sentence-transformers"
        ) from exc
    return SentenceTransformer


def compute_sentence_embeddings(
    texts: list[str],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 64,
    show_progress_bar: bool = True,
    normalize_embeddings: bool = True,
) -> Any:
    """Compute sentence embeddings for review texts."""
    SentenceTransformer = _require_sentence_transformers()
    model = SentenceTransformer(model_name)
    return model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress_bar,
        convert_to_numpy=True,
        normalize_embeddings=normalize_embeddings,
    )


def slugify_model_name(model_name: str) -> str:
    """Make a filesystem-safe name for an embedding model."""
    return model_name.replace("/", "__").replace(":", "_")


def _hash_texts(texts: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for text in texts:
        digest.update(str(text).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()[:16]


def get_cached_sentence_embeddings(
    texts: list[str],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    cache_dir: str | Path = EMBEDDING_CACHE_DIR,
    batch_size: int = 64,
    show_progress_bar: bool = True,
    normalize_embeddings: bool = True,
    cache_key: Optional[str] = None,
) -> np.ndarray:
    """Load sentence embeddings from cache or compute and save them.

    The cache key includes the model name and a content hash of the texts by
    default, so cached arrays remain aligned to the exact input rows.
    """
    cache_path = ensure_dir(cache_dir)
    text_hash = cache_key or _hash_texts(texts)
    model_slug = slugify_model_name(model_name)
    embedding_path = cache_path / f"{model_slug}_{text_hash}.npz"
    metadata_path = cache_path / f"{model_slug}_{text_hash}.json"

    if embedding_path.exists():
        return np.load(embedding_path)["embeddings"]

    embeddings = compute_sentence_embeddings(
        texts,
        model_name=model_name,
        batch_size=batch_size,
        show_progress_bar=show_progress_bar,
        normalize_embeddings=normalize_embeddings,
    )
    np.savez_compressed(embedding_path, embeddings=embeddings)
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "model_name": model_name,
                "n_texts": len(texts),
                "embedding_dim": int(embeddings.shape[1]),
                "normalize_embeddings": normalize_embeddings,
                "cache_key": text_hash,
            },
            f,
            indent=2,
        )
    return embeddings


def train_embedding_model(
    train_texts: list[str],
    y_train: Any,
    test_texts: list[str],
    model: Any,
    task: Literal["classification", "regression"] = "classification",
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> tuple[Any, Any]:
    """Train a classical model on frozen sentence embeddings."""
    x_train = compute_sentence_embeddings(train_texts, embedding_model_name)
    x_test = compute_sentence_embeddings(test_texts, embedding_model_name)
    model.fit(x_train, y_train)
    return model, model.predict(x_test)


def _require_transformers() -> None:
    try:
        import datasets  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Install transformers/datasets for fine-tuning: "
            "pip install transformers datasets evaluate accelerate"
        ) from exc


def fine_tune_transformer_classifier(
    train_df: Any,
    valid_df: Any,
    text_col: str = "description",
    label_col: str = "rating_class",
    model_name: str = "distilbert-base-uncased",
    output_dir: Optional[str] = None,
    **training_args: Any,
) -> Any:
    """Optional Hugging Face classifier fine-tuning entry point."""
    _require_transformers()
    raise NotImplementedError(
        "Fine-tuning is intentionally a stub. Add tokenization, Dataset creation, "
        "Trainer arguments, and metrics here when ready."
    )


def fine_tune_transformer_regressor(
    train_df: Any,
    valid_df: Any,
    text_col: str = "description",
    target_col: str = "points",
    model_name: str = "distilbert-base-uncased",
    output_dir: Optional[str] = None,
    **training_args: Any,
) -> Any:
    """Optional Hugging Face regressor fine-tuning entry point."""
    _require_transformers()
    raise NotImplementedError(
        "Fine-tuning is intentionally a stub. Add tokenization, Dataset creation, "
        "Trainer arguments, and regression metrics here when ready."
    )
