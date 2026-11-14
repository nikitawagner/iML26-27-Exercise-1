"""Evaluate interpretability metrics for every (dataset, model) combination."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd
from sklearn.base import ClassifierMixin, clone
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from iml_interpretability.data import Dataset
from iml_interpretability.metric import InterpretabilityMetric


def evaluate(
    datasets: Iterable[Dataset],
    models: dict[str, ClassifierMixin],
    metrics: Iterable[InterpretabilityMetric],
    seed: int = 0,
    test_size: float = 0.25,
) -> pd.DataFrame:
    """Fit every model on every dataset and compute test accuracy and all metrics.

    :param datasets: Datasets to evaluate on.
    :param models: Unfitted models keyed by name, e.g. from :func:`iml_interpretability.models.get_models`.
    :param metrics: Interpretability metrics to compute for each fitted model.
    :param seed: Random seed for the train/test split.
    :param test_size: Fraction of each dataset held out for computing accuracy.
    :return: One row per (dataset, model) with columns ``dataset``, ``model``, ``accuracy`` and one column per metric.
    :rtype: pd.DataFrame
    """
    metrics = list(metrics)
    rows = []
    for dataset in datasets:
        X_train, X_test, y_train, y_test = train_test_split(
            dataset.X, dataset.y, test_size=test_size, random_state=seed, stratify=dataset.y
        )
        for model_name, model in models.items():
            # Clone so that the same unfitted model can be reused across datasets.
            fitted = clone(model).fit(X_train, y_train)
            row = {
                "dataset": dataset.name,
                "model": model_name,
                "accuracy": accuracy_score(y_test, fitted.predict(X_test)),
            }
            for metric in metrics:
                row[metric.name] = metric.score(fitted)
            rows.append(row)
    return pd.DataFrame(rows)
