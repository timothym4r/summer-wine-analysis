"""Create rating classes from numeric wine ratings."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from .config import CLASS_COL, MIN_CLASS_COUNT, N_CLASSIFICATION_BINS, TARGET_COL


@dataclass
class BinMetadata:
    """Metadata describing rating-class bins."""

    strategy: str
    bin_edges: list[float]
    class_names: list[str]
    class_counts: dict[str, int]

    def to_dict(self) -> dict:
        return asdict(self)


def summarize_rating_distribution(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
) -> dict:
    """Return compact distribution statistics for the target rating."""
    values = pd.to_numeric(df[target_col], errors="coerce").dropna()
    return {
        "count": int(values.shape[0]),
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "median": float(values.median()),
        "std": float(values.std()),
        "value_counts": values.value_counts().sort_index().to_dict(),
        "quantiles": values.quantile([0.05, 0.25, 0.5, 0.75, 0.95]).to_dict(),
    }


def _unique_edges(edges: Iterable[float]) -> list[float]:
    unique = sorted(set(float(edge) for edge in edges))
    if len(unique) < 2:
        raise ValueError("Could not create at least two unique bin edges.")
    unique[0] = -np.inf
    unique[-1] = np.inf
    return unique


def _make_edges(
    values: pd.Series,
    strategy: str,
    n_bins: int,
    custom_bins: Optional[list[float]],
) -> list[float]:
    if strategy == "custom":
        if not custom_bins:
            raise ValueError("custom_bins must be provided when strategy='custom'.")
        return _unique_edges(custom_bins)
    if strategy == "fixed":
        return _unique_edges(np.linspace(values.min(), values.max(), n_bins + 1))
    if strategy == "quantile":
        quantiles = np.linspace(0, 1, n_bins + 1)
        return _unique_edges(values.quantile(quantiles).to_list())
    raise ValueError(f"Unknown binning strategy: {strategy}")


def _label_for_interval(interval: pd.Interval) -> str:
    left = "-inf" if np.isneginf(interval.left) else f"{interval.left:g}"
    right = "inf" if np.isposinf(interval.right) else f"{interval.right:g}"
    return f"{left}-{right}"


def make_rating_bins(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    strategy: str = "quantile",
    n_bins: int = N_CLASSIFICATION_BINS,
    min_class_count: int = MIN_CLASS_COUNT,
    custom_bins: Optional[list[float]] = None,
    label_col: str = CLASS_COL,
) -> tuple[pd.DataFrame, dict]:
    """Create ordered rating classes while avoiding tiny classes.

    The function retries with fewer bins until every class has at least
    ``min_class_count`` rows, or until only two bins remain.
    """
    result = df.copy()
    values = pd.to_numeric(result[target_col], errors="coerce")
    if values.isna().any():
        raise ValueError(f"{target_col} contains non-numeric or missing values.")

    last_counts: pd.Series | None = None
    for candidate_bins in range(n_bins, 1, -1):
        edges = _make_edges(values, strategy, candidate_bins, custom_bins)
        labels = pd.cut(values, bins=edges, include_lowest=True, duplicates="drop")
        counts = labels.value_counts(sort=False)
        last_counts = counts
        if counts.min() >= min_class_count or len(counts) <= 2:
            class_names = [_label_for_interval(interval) for interval in counts.index]
            name_map = dict(zip(counts.index, class_names))
            result[label_col] = labels.map(name_map).astype(str)
            result[label_col] = pd.Categorical(
                result[label_col],
                categories=class_names,
                ordered=True,
            )
            metadata = BinMetadata(
                strategy=strategy,
                bin_edges=edges,
                class_names=class_names,
                class_counts={
                    name_map[interval]: int(count)
                    for interval, count in counts.items()
                },
            )
            return result, metadata.to_dict()

    raise ValueError(f"Could not create usable bins. Last counts: {last_counts}")

