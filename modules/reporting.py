"""Summaries for classical baseline results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .utils import ensure_dir


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def pretty_bin_label(label: str) -> str:
    """Convert internal interval labels into report-friendly text."""
    if label.startswith("-inf-"):
        right = label.replace("-inf-", "", 1)
        return f"<= {right}"
    if label.endswith("-inf"):
        left = label[: -len("-inf")].rstrip("-")
        return f"> {left}"

    left, right = label.split("-", maxsplit=1)
    return f"> {left} to <= {right}"


def _best_row(df: pd.DataFrame, metric: str, ascending: bool) -> dict[str, Any]:
    row = df.sort_values(metric, ascending=ascending).iloc[0]
    return row.to_dict()


def _top_features(path: Path, top_n: int = 10) -> dict[str, list[dict[str, Any]]]:
    df = pd.read_csv(path)
    groups: dict[str, list[dict[str, Any]]] = {}
    for (task, family, model, direction), group in df.groupby(
        ["task", "feature_family", "model", "direction"],
        dropna=False,
    ):
        key = f"{task}/{family}/{model}/{direction}"
        groups[key] = (
            group.sort_values("rank")
            .head(top_n)[["feature", "coefficient", "rank"]]
            .to_dict(orient="records")
        )
    return groups


def create_traditional_summary(
    output_dir: str | Path = "outputs",
    top_n_features: int = 10,
) -> dict[str, Any]:
    """Create a compact summary from existing result artifacts."""
    output_path = Path(output_dir)
    regression = pd.read_csv(output_path / "regression_results.csv")
    classification = pd.read_csv(output_path / "classification_results.csv")
    bins = _read_json(output_path / "classification_bins.json")
    distribution = _read_json(output_path / "rating_distribution.json")

    dummy_reg = regression[regression["feature_family"] == "dummy"].iloc[0]
    best_reg = _best_row(regression, "mae", ascending=True)
    dummy_clf = classification[classification["feature_family"] == "dummy"].iloc[0]
    best_clf = _best_row(classification, "macro_f1", ascending=False)

    summary = {
        "rating_distribution": {
            key: distribution[key]
            for key in ["count", "min", "max", "mean", "median", "std"]
            if key in distribution
        },
        "classification_bins": {
            "strategy": bins["strategy"],
            "class_counts": bins["class_counts"],
            "pretty_class_labels": {
                label: pretty_bin_label(label) for label in bins["class_names"]
            },
        },
        "best_regression": best_reg,
        "regression_improvement_vs_dummy": {
            "mae_reduction": float(dummy_reg["mae"] - best_reg["mae"]),
            "mae_reduction_percent": float(
                (dummy_reg["mae"] - best_reg["mae"]) / dummy_reg["mae"] * 100
            ),
            "rmse_reduction": float(dummy_reg["rmse"] - best_reg["rmse"]),
            "r2_gain": float(best_reg["r2"] - dummy_reg["r2"]),
        },
        "best_classification": best_clf,
        "classification_improvement_vs_dummy": {
            "accuracy_gain": float(best_clf["accuracy"] - dummy_clf["accuracy"]),
            "macro_f1_gain": float(best_clf["macro_f1"] - dummy_clf["macro_f1"]),
            "weighted_f1_gain": float(best_clf["weighted_f1"] - dummy_clf["weighted_f1"]),
            "mean_absolute_class_error_reduction": float(
                dummy_clf["mean_absolute_class_error"]
                - best_clf["mean_absolute_class_error"]
            ),
            "quadratic_weighted_kappa_gain": float(
                best_clf["quadratic_weighted_kappa"]
                - dummy_clf["quadratic_weighted_kappa"]
            ),
        },
        "feature_importance_highlights": {
            "regression": _top_features(
                output_path / "interpretability" / "regression_feature_importance.csv",
                top_n=top_n_features,
            ),
            "classification": _top_features(
                output_path / "interpretability" / "classification_feature_importance.csv",
                top_n=top_n_features,
            ),
        },
    }
    return summary


def save_traditional_summary(
    output_dir: str | Path = "outputs",
    top_n_features: int = 10,
) -> dict[str, Path]:
    """Save JSON and Markdown summaries from existing artifacts."""
    output_path = ensure_dir(Path(output_dir) / "summary")
    summary = create_traditional_summary(output_dir, top_n_features=top_n_features)

    json_path = output_path / "traditional_methods_summary.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    markdown_path = output_path / "traditional_methods_summary.md"
    with markdown_path.open("w", encoding="utf-8") as f:
        f.write("# Traditional Methods Summary\n\n")
        f.write("## Rating Distribution\n\n")
        for key, value in summary["rating_distribution"].items():
            f.write(f"- {key}: {value}\n")

        f.write("\n## Classification Bins\n\n")
        for label, count in summary["classification_bins"]["class_counts"].items():
            pretty = summary["classification_bins"]["pretty_class_labels"][label]
            f.write(f"- {label} ({pretty}): {count}\n")

        f.write("\n## Best Regression Model\n\n")
        best_reg = summary["best_regression"]
        f.write(
            f"- {best_reg['feature_family']} + {best_reg['model']}: "
            f"MAE={best_reg['mae']:.4f}, RMSE={best_reg['rmse']:.4f}, "
            f"R2={best_reg['r2']:.4f}\n"
        )
        reg_imp = summary["regression_improvement_vs_dummy"]
        f.write(
            f"- MAE improves over dummy by {reg_imp['mae_reduction']:.4f} "
            f"({reg_imp['mae_reduction_percent']:.1f}%).\n"
        )

        f.write("\n## Best Classification Model\n\n")
        best_clf = summary["best_classification"]
        f.write(
            f"- {best_clf['feature_family']} + {best_clf['model']}: "
            f"accuracy={best_clf['accuracy']:.4f}, macro F1={best_clf['macro_f1']:.4f}, "
            f"weighted F1={best_clf['weighted_f1']:.4f}, "
            f"QWK={best_clf['quadratic_weighted_kappa']:.4f}\n"
        )
        clf_imp = summary["classification_improvement_vs_dummy"]
        f.write(
            f"- Macro F1 improves over dummy by {clf_imp['macro_f1_gain']:.4f}; "
            f"mean absolute class error drops by "
            f"{clf_imp['mean_absolute_class_error_reduction']:.4f}.\n"
        )

        f.write("\n## Interpretability Notes\n\n")
        f.write(
            "- Coefficients are directional model associations, not causal claims.\n"
            "- Positive regression coefficients push predicted points upward; "
            "negative coefficients push predicted points downward.\n"
            "- Classification coefficients are class-specific associations.\n"
        )

    return {"json": json_path, "markdown": markdown_path}
