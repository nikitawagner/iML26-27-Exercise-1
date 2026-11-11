from __future__ import annotations

import json
import shutil
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from iml_interpretability import const
from iml_interpretability.visualization import save_visualization


def train_tree(
    dataset_path: Path,
    target: str,
    *,
    task: Literal["classification", "regression"] = const.DEFAULT_TASK,
    model_type: Literal["decision_tree", "random_forest"] = const.DEFAULT_MODEL,
    output_dir: Path = const.DEFAULT_OUTPUT_DIR,
    drop_columns: Sequence[str] = (),
    max_depth: int = const.DEFAULT_MAX_DEPTH,
    min_samples_split: int = const.DEFAULT_MIN_SAMPLES_SPLIT,
    min_samples_leaf: int = const.DEFAULT_MIN_SAMPLES_LEAF,
    ccp_alpha: float = const.DEFAULT_CCP_ALPHA,
    seed: int = const.DEFAULT_SEED,
    test_size: float = const.DEFAULT_TEST_SIZE,
    n_estimators: int = const.DEFAULT_N_ESTIMATORS,
    n_jobs: int = const.DEFAULT_N_JOBS,
    max_features: Literal["sqrt", "log2", "all"] | None = const.DEFAULT_MAX_FEATURES,
) -> Path:
    dot_command = shutil.which("dot")
    if dot_command is None:
        raise FileNotFoundError("Graphviz (dot) is required for visualization. On macOS: brew install graphviz")
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
    estimator_parameters = {
        "max_depth": max_depth,
        "min_samples_split": min_samples_split,
        "min_samples_leaf": min_samples_leaf,
        "ccp_alpha": ccp_alpha,
        "random_state": seed,
    }
    if max_features is not None:
        estimator_parameters["max_features"] = None if max_features == "all" else max_features
    if model_type == "random_forest":
        estimator_class = RandomForestClassifier if task == "classification" else RandomForestRegressor
        estimator_parameters.update(n_estimators=n_estimators, n_jobs=n_jobs)
    else:
        estimator_class = DecisionTreeClassifier if task == "classification" else DecisionTreeRegressor
    estimator = estimator_class(**estimator_parameters)
    model = Pipeline([("preprocessing", preprocessing), ("tree", estimator)])

    print(f"[3/5] Training {type(estimator).__name__} on {len(X_train)} rows", flush=True)
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
    pd.DataFrame([{"dataset": dataset_path.stem, "model": type(estimator).__name__, **scores}]).to_csv(
        run_dir / "scores.csv", index=False
    )
    feature_names = preprocessing.get_feature_names_out().tolist()
    parameters = {
        "dataset": str(dataset_path.resolve()),
        "target": target,
        "task": task,
        "model_type": model_type,
        "drop_columns": list(drop_columns),
        "features": X.columns.tolist(),
        "transformed_features": feature_names,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "test_size": test_size,
        "seed": seed,
        "hyperparameters": estimator.get_params(),
        "visualized_tree_index": 0 if model_type == "random_forest" else None,
    }
    (run_dir / "parameters.json").write_text(json.dumps(parameters, indent=2) + "\n", encoding="utf-8")

    save_visualization(estimator, feature_names, run_dir, dataset_path.stem, dot_command)
    print("Scores: " + ", ".join(f"{name}={score:.4f}" for name, score in scores.items()), flush=True)
    return run_dir
