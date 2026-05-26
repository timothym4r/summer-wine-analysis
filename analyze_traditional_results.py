"""Generate summaries and focused error analysis for classical baselines."""

from __future__ import annotations

import argparse

from modules.binning import make_rating_bins
from modules.config import DEFAULT_DATA_PATH, DEFAULT_OUTPUT_DIR, TARGET_COL, TEXT_COL
from modules.data import load_wine_data
from modules.error_analysis import (
    save_classification_error_analysis,
    save_regression_error_analysis,
)
from modules.preprocessing import preprocess_dataframe
from modules.reporting import save_traditional_summary
from modules.utils import timer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--skip-predictions",
        action="store_true",
        help="Only summarize existing outputs; do not refit models for prediction files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with timer("summary from existing outputs"):
        summary_paths = save_traditional_summary(args.output_dir)
    for name, path in summary_paths.items():
        print(f"Saved {name} summary: {path}")

    if args.skip_predictions:
        return

    with timer("focused prediction error analysis"):
        df = load_wine_data(args.data, text_col=TEXT_COL, target_col=TARGET_COL)
        df = preprocess_dataframe(df, text_col=TEXT_COL)
        binned_df, _ = make_rating_bins(df, target_col=TARGET_COL, strategy="quantile")

        regression_path = save_regression_error_analysis(
            df,
            output_dir=args.output_dir,
            feature_family="tfidf",
            text_col=TEXT_COL,
            target_col=TARGET_COL,
        )
        classification_paths = save_classification_error_analysis(
            binned_df,
            output_dir=args.output_dir,
            feature_family="tfidf",
            text_col=TEXT_COL,
            target_col=TARGET_COL,
        )

    print(f"Saved regression predictions: {regression_path}")
    for name, path in classification_paths.items():
        print(f"Saved classification {name}: {path}")


if __name__ == "__main__":
    main()
