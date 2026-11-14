"""Run the interpretability evaluation and store the results as a CSV file.

Usage::

    uv run python scripts/run_experiment.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from iml_interpretability.training import train_tree

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "scores.csv"
DATASETS = (
    {
        "dataset_path": ROOT / "data" / "car_evaluation.csv",
        "target": "class",
        "task": "classification",
    },
    {
        "dataset_path": ROOT / "data" / "bike_sharing_day.csv",
        "target": "cnt",
        "task": "regression",
        "drop_columns": ("instant", "dteday", "casual", "registered"),
    },
)
MAX_DEPTHS = (3, 8)
MODELS = (
    {"model_type": "decision_tree"},
    {"model_type": "random_forest", "n_estimators": 10},
    {"model_type": "random_forest", "n_estimators": 100},
)


def main() -> None:
    runs = []
    for dataset in DATASETS:
        for max_depth in MAX_DEPTHS:
            for model in MODELS:
                run_dir = train_tree(**dataset, **model, max_depth=max_depth, output_dir=OUTPUT.parent)
                scores = pd.read_csv(run_dir / "scores.csv")
                scores["run_dir"] = str(run_dir.relative_to(ROOT))
                runs.append(scores)

    results = pd.concat(runs, ignore_index=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT, index=False)
    print(results.to_string(index=False))
    print(f"Saved combined scores to {OUTPUT}")


if __name__ == "__main__":
    main()
