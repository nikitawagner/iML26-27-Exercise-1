"""Interpretability metrics.

Implement your own metric by subclassing :class:`InterpretabilityMetric`. The metric must be
applicable to every model of the chosen model class, so that scores are comparable across
models and datasets. :class:`LeafDepthComplexity` scores decision trees and random forests;
:class:`RelativeNumberOfLeaves` remains as the original example.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from math import log2

from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_is_fitted

from iml_interpretability.data import Dataset


class InterpretabilityMetric(ABC):
    """Base class for a metric that quantifies how interpretable a fitted model is."""

    #: Column name under which the score is reported.
    name: str

    @abstractmethod
    def score(self, model: ClassifierMixin, dataset: Dataset) -> float:
        """Compute the interpretability score of a fitted model.

        :param model: A model fitted on ``dataset``.
        :param dataset: The dataset the model was fitted on.
        :return: The interpretability score. Document whether higher means more or less interpretable.
        :rtype: float
        """


class RelativeNumberOfLeaves(InterpretabilityMetric):
    """Number of leaves of a decision tree relative to the maximum possible for its depth.

    A binary tree of depth ``d`` has at most ``2 ** d`` leaves, so the score lies in ``(0, 1]``, where
    1 means a complete tree. Lower means the tree uses fewer of the decision rules its depth would
    allow. This is only an example: replace it with your own metric, which should go beyond such
    simple size measures, and argue why it captures interpretability.
    """

    name = "relative_n_leaves"

    def score(self, model: ClassifierMixin, dataset: Dataset) -> float:
        if not isinstance(model, DecisionTreeClassifier):
            raise TypeError(f"{type(self).__name__} only supports DecisionTreeClassifier, got {type(model).__name__}")
        return model.get_n_leaves() / 2 ** model.get_depth()


class LeafDepthComplexity(InterpretabilityMetric):
    """Structural complexity of a decision tree or random forest.

    For a tree, the score is ``log2(L) + mean(leaf depths)``, where every
    leaf contributes equally and the root has depth zero. For a forest, the
    scores of its trees are summed and ``T - 1`` is added for ``T`` trees.
    Higher scores indicate greater structural complexity and thus potentially
    lower interpretability. The score is deliberately not normalized because
    the normalization has not yet been specified in the team protocol.
    ``dataset`` is unused: the score describes model structure, not how often
    the dataset visits each leaf.
    """

    name = "leaf_depth_complexity"

    @staticmethod
    def _tree_score(model: DecisionTreeClassifier) -> float:
        tree = model.tree_
        leaf_count = 0
        depth_sum = 0
        stack = [(0, 0)]

        while stack:
            node, depth = stack.pop()
            left = tree.children_left[node]
            right = tree.children_right[node]
            if left == right:  # A leaf has no children (-1, -1).
                leaf_count += 1
                depth_sum += depth
            else:
                stack.append((left, depth + 1))
                stack.append((right, depth + 1))

        return float(log2(leaf_count) + depth_sum / leaf_count)
    def score(self, model: ClassifierMixin, dataset: Dataset) -> float:
        if isinstance(model, DecisionTreeClassifier):
            check_is_fitted(model)
            return self._tree_score(model)
        if isinstance(model, RandomForestClassifier):
            check_is_fitted(model)
            return float(sum(self._tree_score(tree) for tree in model.estimators_) + len(model.estimators_) - 1)
        raise TypeError(
            f"{type(self).__name__} only supports DecisionTreeClassifier and "
            f"RandomForestClassifier, got {type(model).__name__}"
        )
