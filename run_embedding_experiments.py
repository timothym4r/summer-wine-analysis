"""Run Stage 1 advanced models: sentence embeddings plus prediction heads."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from modules.config import DEFAULT_DATA_PATH, EMBEDDING_MODELS, TARGET_COL, TEXT_COL
from modules.data import load_wine_data
from modules.embedding_experiments import run_embedding_stage1_experiments
from modules.preprocessing import preprocess_dataframe
from modules.utils import ensure_dir, timer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", default="outputs/advanced_embeddings")
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(EMBEDDING_MODELS.keys()),
        choices=list(EMBEDDING_MODELS.keys()),
        help="Embedding model aliases to run.",
    )
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--no-nn", action="store_true", help="Skip MLP heads.")
    return parser.parse_args()


def _best_rows(results: dict[str, pd.DataFrame]) -> dict:
    regression = results["regression"].sort_values("mae").iloc[0].to_dict()
    classification = (
        results["classification"].sort_values("macro_f1", ascending=False).iloc[0].to_dict()
    )
    return {
        "best_embedding_regression": regression,
        "best_embedding_classification": classification,
    }


def _load_traditional_comparison() -> dict:
    paths = {
        "traditional_regression": Path("outputs/regression_results.csv"),
        "traditional_classification": Path("outputs/classification_results.csv"),
    }
    if not all(path.exists() for path in paths.values()):
        return {}

    regression = pd.read_csv(paths["traditional_regression"]).sort_values("mae").iloc[0]
    classification = (
        pd.read_csv(paths["traditional_classification"])
        .sort_values("macro_f1", ascending=False)
        .iloc[0]
    )
    return {
        "best_traditional_regression": regression.to_dict(),
        "best_traditional_classification": classification.to_dict(),
    }


def _save_summary(results: dict[str, pd.DataFrame], output_dir: str | Path) -> None:
    output_path = ensure_dir(Path(output_dir) / "summary")
    summary = {
        **_best_rows(results),
        **_load_traditional_comparison(),
        "interpretability_note": (
            "Embedding dimensions do not correspond to human-readable words. "
            "For Stage 1, interpretability artifacts are prediction files, "
            "large-error examples, confusion matrices, and comparison against "
            "the interpretable TF-IDF/count baselines."
        ),
    }

    json_path = output_path / "embedding_stage1_summary.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    md_path = output_path / "embedding_stage1_summary.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Embedding Stage 1 Summary\n\n")
        reg = summary["best_embedding_regression"]
        f.write("## Best Embedding Regression\n\n")
        f.write(
            f"- {reg['embedding_short_name']} + {reg['head']}: "
            f"MAE={reg['mae']:.4f}, RMSE={reg['rmse']:.4f}, R2={reg['r2']:.4f}\n"
        )
        clf = summary["best_embedding_classification"]
        f.write("\n## Best Embedding Classification\n\n")
        f.write(
            f"- {clf['embedding_short_name']} + {clf['head']}: "
            f"accuracy={clf['accuracy']:.4f}, macro F1={clf['macro_f1']:.4f}, "
            f"weighted F1={clf['weighted_f1']:.4f}, "
            f"QWK={clf['quadratic_weighted_kappa']:.4f}\n"
        )
        if "best_traditional_regression" in summary:
            trad_reg = summary["best_traditional_regression"]
            f.write("\n## Traditional Comparison\n\n")
            f.write(
                f"- Best traditional regression: {trad_reg['feature_family']} + "
                f"{trad_reg['model']} with MAE={trad_reg['mae']:.4f}\n"
            )
            trad_clf = summary["best_traditional_classification"]
            f.write(
                f"- Best traditional classification: {trad_clf['feature_family']} + "
                f"{trad_clf['model']} with macro F1={trad_clf['macro_f1']:.4f}\n"
            )
        f.write("\n## Interpretability\n\n")
        f.write(f"- {summary['interpretability_note']}\n")


def main() -> None:
    args = parse_args()
    selected_models = {alias: EMBEDDING_MODELS[alias] for alias in args.models}

    with timer("load and preprocess"):
        df = load_wine_data(args.data, text_col=TEXT_COL, target_col=TARGET_COL)
        df = preprocess_dataframe(df, text_col=TEXT_COL)

    with timer("embedding stage 1 experiments"):
        results = run_embedding_stage1_experiments(
            df,
            output_dir=args.output_dir,
            text_col=TEXT_COL,
            target_col=TARGET_COL,
            embedding_models=selected_models,
            include_nn=not args.no_nn,
            sample_size=args.sample_size,
            batch_size=args.batch_size,
        )

    _save_summary(results, args.output_dir)
    print(f"Saved Stage 1 embedding outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
