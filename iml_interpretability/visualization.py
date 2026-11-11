from __future__ import annotations

import subprocess
from pathlib import Path

from sklearn.base import BaseEstimator, is_classifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import export_graphviz

from iml_interpretability import const


def save_visualization(
    estimator: BaseEstimator,
    feature_names: list[str],
    run_dir: Path,
    dataset_name: str,
    dot_command: str,
) -> None:
    tree = estimator
    tree_count = 1
    if isinstance(estimator, (RandomForestClassifier, RandomForestRegressor)):
        tree = estimator.estimators_[0]
        tree_count = len(estimator.estimators_)
    class_names = [str(label) for label in estimator.classes_] if is_classifier(estimator) else None
    display_names = [name.split("__", 1)[-1] for name in feature_names]
    source = export_graphviz(
        tree,
        out_file=None,
        feature_names=display_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        precision=3,
        fontname=const.TREE_FONT_NAME,
    ).encode("utf-8")
    subprocess.run(
        [
            dot_command,
            "-Tsvg",
            f"-Nfontsize={const.TREE_FONT_SIZE}",
            f"-Gfontsize={const.TREE_FONT_SIZE}",
            f"-Gfontname={const.TREE_FONT_NAME}",
            "-Glabelloc=t",
            f"-Glabel={dataset_name} — {type(estimator).__name__} — Baum 1/{tree_count}",
            "-o",
            str(run_dir / "tree.svg"),
        ],
        input=source,
        check=True,
    )
