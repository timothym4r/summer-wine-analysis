"""Run baseline wine review rating prediction experiments."""

from __future__ import annotations

import argparse

from modules.binning import make_rating_bins, summarize_rating_distribution
from modules.config import DEFAULT_DATA_PATH, DEFAULT_OUTPUT_DIR, TARGET_COL, TEXT_COL
from modules.data import load_wine_data
from modules.evaluation import save_results_json
from modules.experiments import run_classification_experiments, run_regression_experiments
from modules.interpretability import save_interpretability_reports
from modules.preprocessing import preprocess_dataframe
from modules.utils import ensure_dir, save_csv, set_random_seed, timer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=DEFAULT_DATA_PATH, help="Input Wine Enthusiast CSV path.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory for results.")
    parser.add_argument("--run-embeddings", action="store_true", help="Run sentence embedding experiments.")
    parser.add_argument("--run-finetuning", action="store_true", help="Run transformer fine-tuning.")
    parser.add_argument("--run-llm-features", action="store_true", help="Run LLM attribute extraction.")
    parser.add_argument("--run-few-shot-llm", action="store_true", help="Run few-shot LLM predictors.")
    parser.add_argument(
        "--skip-interpretability",
        action="store_true",
        help="Skip coefficient-based feature importance reports.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_random_seed()
    output_dir = ensure_dir(args.output_dir)

    with timer("load and preprocess"):
        df = load_wine_data(args.data, text_col=TEXT_COL, target_col=TARGET_COL)
        df = preprocess_dataframe(df, text_col=TEXT_COL)

    distribution = summarize_rating_distribution(df, target_col=TARGET_COL)
    print("Rating distribution:")
    for key in ["count", "min", "max", "mean", "median", "std"]:
        print(f"  {key}: {distribution[key]}")
    save_results_json(distribution, output_dir / "rating_distribution.json")

    with timer("regression experiments"):
        regression_results = run_regression_experiments(
            df,
            text_col=TEXT_COL,
            target_col=TARGET_COL,
            run_embeddings=args.run_embeddings,
            run_finetuning=args.run_finetuning,
            run_llm_features=args.run_llm_features,
            run_few_shot_llm=args.run_few_shot_llm,
        )
    save_csv(regression_results, output_dir / "regression_results.csv")

    binned_df, bin_metadata = make_rating_bins(df, target_col=TARGET_COL, strategy="quantile")
    save_results_json(bin_metadata, output_dir / "classification_bins.json")

    with timer("classification experiments"):
        classification_results = run_classification_experiments(
            binned_df,
            text_col=TEXT_COL,
            target_col=TARGET_COL,
            bin_strategy="quantile",
            run_embeddings=args.run_embeddings,
            run_finetuning=args.run_finetuning,
            run_llm_features=args.run_llm_features,
            run_few_shot_llm=args.run_few_shot_llm,
        )
    save_csv(classification_results, output_dir / "classification_results.csv")

    if not args.skip_interpretability:
        with timer("interpretability reports"):
            saved_paths = save_interpretability_reports(
                df,
                binned_df,
                output_dir=output_dir,
                text_col=TEXT_COL,
                target_col=TARGET_COL,
            )
        for name, path in saved_paths.items():
            print(f"Saved {name}: {path}")

    print(f"Saved results to {output_dir}")


if __name__ == "__main__":
    main()
