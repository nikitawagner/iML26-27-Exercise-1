from __future__ import annotations

import argparse
from pathlib import Path

from iml_interpretability import const
from iml_interpretability.training import train_tree


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a decision tree from a comma-separated CSV with a header.")
    parser.add_argument("dataset", type=Path, help="Path to the CSV dataset")
    parser.add_argument("--target", required=True, help="Name of the target column")
    parser.add_argument("--task", choices=["classification", "regression"], default=const.DEFAULT_TASK)
    parser.add_argument("--drop-columns", nargs="+", default=[], help="Exclude IDs or columns that reveal the target")
    parser.add_argument(
        "--max-depth",
        type=int,
        default=const.DEFAULT_MAX_DEPTH,
        help=f"Maximum tree depth (default: {const.DEFAULT_MAX_DEPTH})",
    )
    parser.add_argument(
        "--min-samples-split",
        type=int,
        default=const.DEFAULT_MIN_SAMPLES_SPLIT,
        help=f"Minimum rows to split a node (default: {const.DEFAULT_MIN_SAMPLES_SPLIT})",
    )
    parser.add_argument(
        "--min-samples-leaf",
        type=int,
        default=const.DEFAULT_MIN_SAMPLES_LEAF,
        help=f"Minimum rows per leaf (default: {const.DEFAULT_MIN_SAMPLES_LEAF})",
    )
    parser.add_argument(
        "--ccp-alpha",
        type=float,
        default=const.DEFAULT_CCP_ALPHA,
        help=f"Pruning complexity parameter (default: {const.DEFAULT_CCP_ALPHA})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=const.DEFAULT_SEED,
        help=f"Random seed for splitting and training (default: {const.DEFAULT_SEED})",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=const.DEFAULT_TEST_SIZE,
        help=f"Held-out fraction (default: {const.DEFAULT_TEST_SIZE})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / const.DEFAULT_OUTPUT_DIR,
        help=f"Root directory for per-dataset, timestamped runs (default: repository {const.DEFAULT_OUTPUT_DIR}/)",
    )
    args = parser.parse_args()

    try:
        train_tree(
            args.dataset,
            args.target,
            task=args.task,
            output_dir=args.output_dir,
            drop_columns=args.drop_columns,
            max_depth=args.max_depth,
            min_samples_split=args.min_samples_split,
            min_samples_leaf=args.min_samples_leaf,
            ccp_alpha=args.ccp_alpha,
            seed=args.seed,
            test_size=args.test_size,
        )
    except (OSError, ValueError) as error:
        parser.exit(2, f"Training failed: {error}\n")


if __name__ == "__main__":
    main()
