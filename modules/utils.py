"""Small utilities used across experiments."""

from __future__ import annotations

import json
import random
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


def set_random_seed(seed: int = 42) -> None:
    """Set Python and, when available, NumPy random seeds."""
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if it does not exist and return it as a Path."""
    output_path = Path(path)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def save_json(data: dict[str, Any], path: str | Path) -> None:
    """Save a dictionary as pretty JSON."""
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def save_csv(df: Any, path: str | Path, index: bool = False) -> None:
    """Save a pandas DataFrame to CSV."""
    path = Path(path)
    ensure_dir(path.parent)
    df.to_csv(path, index=index)


def progress_iter(iterable: Any, desc: str | None = None) -> Any:
    """Wrap an iterable with tqdm when available, otherwise return it unchanged."""
    try:
        from tqdm.auto import tqdm

        return tqdm(iterable, desc=desc)
    except ImportError:
        return iterable


@contextmanager
def timer(label: str) -> Iterator[dict[str, float]]:
    """Context manager that records elapsed wall-clock time."""
    result: dict[str, float] = {}
    start = time.perf_counter()
    try:
        yield result
    finally:
        result["seconds"] = time.perf_counter() - start
        print(f"{label}: {result['seconds']:.2f}s")
