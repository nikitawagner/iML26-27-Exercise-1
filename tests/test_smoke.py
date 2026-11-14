"""Smoke tests that run offline on a synthetic dataset."""

from __future__ import annotations

import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from iml_interpretability import Dataset, InterpretabilityMetric, RelativeNumberOfLeaves, evaluate, get_models


class _ConstantMetric(InterpretabilityMetric):
    name = "constant"

    def score(self, model):
        return 1.0


@pytest.fixture
def dataset() -> Dataset:
    X, y = make_classification(n_samples=100, n_features=5, random_state=0)
    return Dataset(name="synthetic", X=pd.DataFrame(X).add_prefix("x"), y=pd.Series(y))


def test_evaluate_returns_one_row_per_dataset_and_model(dataset):
    models = get_models()
    results = evaluate([dataset], models, [_ConstantMetric()])
    assert len(results) == len(models)
    assert set(results.columns) == {"dataset", "model", "accuracy", "constant"}


def test_relative_number_of_leaves_matches_tree(dataset):
    model = DecisionTreeClassifier(max_depth=2, random_state=0).fit(dataset.X, dataset.y)
    score = RelativeNumberOfLeaves().score(model)
    assert score == model.get_n_leaves() / 2 ** model.get_depth()
    assert 0 < score <= 1


def test_relative_number_of_leaves_rejects_non_tree_models(dataset):
    model = LogisticRegression().fit(dataset.X, dataset.y)
    with pytest.raises(TypeError):
        RelativeNumberOfLeaves().score(model)
