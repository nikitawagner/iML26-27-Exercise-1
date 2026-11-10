from __future__ import annotations

import argparse
from pathlib import Path

from iml_interpretability.training import train_tree


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a decision tree from a comma-separated CSV with a header.")
    parser.add_argument("dataset", type=Path, help="Path to the CSV dataset")
    parser.add_argument("--target", required=True, help="Name of the target column")
    parser.add_argument("--task", choices=["classification", "regression"], default="classification")
    parser.add_argument("--drop-columns", nargs="+", default=[], help="Exclude IDs or columns that reveal the target")
    parser.add_argument("--max-depth", type=int, default=5, help="Maximum tree depth (default: 5)")
    parser.add_argument("--min-samples-split", type=int, default=2, help="Minimum rows to split a node (default: 2)")
    parser.add_argument("--min-samples-leaf", type=int, default=5, help="Minimum rows per leaf (default: 5)")
    parser.add_argument("--ccp-alpha", type=float, default=0.0, help="Pruning complexity parameter (default: 0)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for splitting and training (default: 0)")
    parser.add_argument("--test-size", type=float, default=0.25, help="Held-out fraction (default: 0.25)")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "results",
        help="Root directory for per-dataset, timestamped runs (default: repository results/)",
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
