from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree


def train_tree(
    dataset_path: Path,
    target: str,
    *,
    task: Literal["classification", "regression"] = "classification",
    output_dir: Path = Path("results"),
    drop_columns: Sequence[str] = (),
    max_depth: int = 5,
    min_samples_split: int = 2,
    min_samples_leaf: int = 5,
    ccp_alpha: float = 0.0,
    seed: int = 0,
    test_size: float = 0.25,
) -> Path:
    print(f"[1/5] Loading {dataset_path}", flush=True)
    frame = pd.read_csv(dataset_path)
    missing_columns = {target, *drop_columns} - set(frame.columns)
    if missing_columns:
        raise ValueError(f"Columns not found in CSV: {', '.join(sorted(missing_columns))}")
    if frame.empty:
        raise ValueError("The CSV contains no data rows.")
    if frame[target].isna().any():
        raise ValueError(f"Target column '{target}' contains missing values.")

    X = frame.drop(columns=[target, *drop_columns])
    y = frame[target]
    if X.shape[1] == 0:
        raise ValueError("No feature columns remain after dropping the target and excluded columns.")

    print(f"[2/5] Preparing {len(frame)} rows and {X.shape[1]} features", flush=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y if task == "classification" else None
    )
    preprocessing = ColumnTransformer(
        [
            (
                "numeric",
                SimpleImputer(strategy="median", keep_empty_features=True),
                make_column_selector(dtype_include="number"),
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                make_column_selector(dtype_exclude=["number", "bool"]),
            ),
            ("boolean", "passthrough", make_column_selector(dtype_include="bool")),
        ]
    )
    tree_class = DecisionTreeClassifier if task == "classification" else DecisionTreeRegressor
    tree = tree_class(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        ccp_alpha=ccp_alpha,
        random_state=seed,
    )
    model = Pipeline([("preprocessing", preprocessing), ("tree", tree)])

    print(f"[3/5] Training {type(tree).__name__} on {len(X_train)} rows", flush=True)
    model.fit(X_train, y_train)

    print(f"[4/5] Evaluating on {len(X_test)} held-out rows", flush=True)
    predictions = model.predict(X_test)
    if task == "classification":
        scores = {"accuracy": accuracy_score(y_test, predictions)}
    else:
        scores = {
            "mae": mean_absolute_error(y_test, predictions),
            "rmse": root_mean_squared_error(y_test, predictions),
            "r2": r2_score(y_test, predictions),
        }

    run_name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = output_dir / dataset_path.stem / run_name
    print(f"[5/5] Saving artifacts to {run_dir}", flush=True)
    run_dir.mkdir(parents=True)
    joblib.dump(model, run_dir / "model.joblib")
    pd.DataFrame([{"dataset": dataset_path.stem, "model": type(tree).__name__, **scores}]).to_csv(
        run_dir / "scores.csv", index=False
    )
    feature_names = preprocessing.get_feature_names_out().tolist()
    parameters = {
        "dataset": str(dataset_path.resolve()),
        "target": target,
        "task": task,
        "drop_columns": list(drop_columns),
        "features": X.columns.tolist(),
        "transformed_features": feature_names,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "test_size": test_size,
        "seed": seed,
        "hyperparameters": tree.get_params(),
    }
    (run_dir / "parameters.json").write_text(json.dumps(parameters, indent=2) + "\n", encoding="utf-8")

    figure = Figure(figsize=(24, 12))
    FigureCanvasAgg(figure)
    axes = figure.subplots()
    plot_tree(
        tree,
        feature_names=feature_names,
        class_names=[str(label) for label in tree.classes_] if task == "classification" else None,
        filled=True,
        rounded=True,
        fontsize=8,
        ax=axes,
    )
    axes.set_title(f"{dataset_path.stem} — {type(tree).__name__}")
    figure.savefig(run_dir / "tree.png", dpi=150, bbox_inches="tight")
    print("Scores: " + ", ".join(f"{name}={score:.4f}" for name, score in scores.items()), flush=True)
    return run_dir
