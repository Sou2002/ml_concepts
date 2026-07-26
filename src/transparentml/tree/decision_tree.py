"""Implementation of Decision Tree."""

from collections import Counter
from typing import Optional

import numpy as np

from transparentml._typing import (
    ClassLabels,
    FeatureMatrix,
    FeatureMatrixArray,
    TargetVector,
    TargetVectorArray,
)


class _Node:
    """
    A single node in the decision tree.

    Represents either a decision (internal) node, which splits data
    on a feature/threshold, or a leaf node, which holds a predicted
    class value.

    Parameters
    ----------
    feature_idx : int, optional
        Index of the feature this node splits on. ``None`` for leaf
        nodes.
    threshold : float, optional
        Threshold value for the split. Samples with
        ``feature_value <= threshold`` go left, others go right.
        ``None`` for leaf nodes.
    info_gain : float, optional
        Information gain achieved by this split. ``None`` for leaf
        nodes.
    left : _Node, optional
        Left child node. ``None`` for leaf nodes.
    right : _Node, optional
        Right child node. ``None`` for leaf nodes.
    value : float, optional
        Predicted class label. Only set for leaf nodes; ``None`` for
        decision nodes.
    """

    def __init__(
        self,
        feature_idx: int | None = None,
        threshold: np.float64 | None = None,
        info_gain: np.float64 | None = None,
        left: Optional["_Node"] = None,
        right: Optional["_Node"] = None,
        value: np.float64 | None = None,
    ) -> None:
        """Initialize a decision tree node."""
        # Attributes of a decision (internal) node
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.info_gain = info_gain
        self.left = left
        self.right = right

        # Leaf Node
        self.value = value


class DecisionTree:
    """
    A decision tree classifier built via recursive binary splitting.

    At each node, the tree searches every feature and every candidate
    threshold, choosing the split that maximizes information gain
    (entropy reduction). Recursion stops once a node runs out of
    samples to split, reaches `max_depth`, or no split improves
    information gain, at which point it becomes a leaf holding the
    majority class label.

    Parameters
    ----------
    min_samples_split : int, default=2
        Minimum number of samples required to consider splitting a
        node further. Nodes with fewer samples become leaves.
    max_depth : int, default=3
        Maximum depth of the tree. Depth 0 is the root.

    Attributes
    ----------
    root : _Node or None
        Root node of the fitted tree. ``None`` until `fit` is called.
    """

    def __init__(self, min_samples_split: int = 2, max_depth: int = 3) -> None:
        """
        Initialize an unfitted DecisionTree.

        Parameters
        ----------
        min_samples_split : int, default=2
            Minimum number of samples required to split a node.
        max_depth : int, default=3
            Maximum depth of the tree.
        """
        self.min_samples_split: int = min_samples_split
        self.max_depth: int = max_depth
        self.root: _Node | None = None

    def _build_tree(self, dataset: FeatureMatrixArray, curr_depth: int = 0) -> "_Node":
        """
        Recursively build the decision tree.

        Parameters
        ----------
        dataset : ndarray of shape (n_samples, n_features + 1)
            Training data with the target column concatenated as the
            last column.
        curr_depth : int, default=0
            Current recursion depth.

        Returns
        -------
        _Node
            The root of the (sub)tree built from `dataset`.
        """
        X, y = dataset[:, :-1], dataset[:, -1]
        n_samples, n_features = X.shape

        if n_samples >= self.min_samples_split and curr_depth < self.max_depth:
            best_split = self._best_split(dataset=dataset, n_features=n_features)

            if best_split["info_gain"] > 0:
                left_node = self._build_tree(
                    dataset=best_split["left_dataset"], curr_depth=curr_depth + 1
                )
                right_node = self._build_tree(
                    dataset=best_split["right_dataset"], curr_depth=curr_depth + 1
                )

                return _Node(
                    feature_idx=best_split["feature_idx"],
                    threshold=best_split["threshold"],
                    info_gain=best_split["info_gain"],
                    left=left_node,
                    right=right_node,
                )

        leaf_value = Counter(y).most_common(1)[0][0]

        return _Node(value=leaf_value)

    def _best_split(self, dataset: FeatureMatrixArray, n_features: int) -> dict:
        """
        Find the feature and threshold that maximize information gain.

        Exhaustively searches every feature and every unique value of
        that feature as a candidate threshold.

        Parameters
        ----------
        dataset : ndarray of shape (n_samples, n_features + 1)
            Data to split, with target as the last column.
        n_features : int
            Number of features to search over.

        Returns
        -------
        dict
            Dictionary with keys ``feature_idx``, ``threshold``,
            ``info_gain``, ``left_dataset``, and ``right_dataset``
            describing the best split found. ``info_gain`` is ``-1``
            if no valid split exists.
        """
        best_split_dict = {
            "feature_idx": None,
            "threshold": None,
            "info_gain": -1,
            "left_dataset": None,
            "right_dataset": None,
        }

        for feature_idx in range(n_features):
            feature_values = dataset[:, feature_idx]
            thresholds = np.unique(feature_values)

            for threshold in thresholds:
                left_dataset, right_dataset = self._split(
                    dataset=dataset, feature_idx=feature_idx, threshold=threshold
                )

                if len(left_dataset) and len(right_dataset):
                    parent_y = dataset[:, -1]
                    left_y, right_y = left_dataset[:, -1], right_dataset[:, -1]

                    info_gain = self._information_gain(
                        parent_y=parent_y, left_y=left_y, right_y=right_y
                    )

                    if info_gain > best_split_dict["info_gain"]:
                        best_split_dict["feature_idx"] = feature_idx
                        best_split_dict["threshold"] = threshold
                        best_split_dict["info_gain"] = info_gain
                        best_split_dict["left_dataset"] = left_dataset
                        best_split_dict["right_dataset"] = right_dataset

        return best_split_dict

    def _split(
        self, dataset: FeatureMatrixArray, feature_idx: int, threshold: np.float64
    ) -> tuple[FeatureMatrixArray, FeatureMatrixArray]:
        """
        Partition a dataset into left/right subsets on a threshold.

        Parameters
        ----------
        dataset : ndarray of shape (n_samples, n_features + 1)
            Data to partition.
        feature_idx : int
            Index of the feature to split on.
        threshold : float
            Split threshold. Rows with ``feature_value <= threshold``
            go left, others go right.

        Returns
        -------
        left_dataset : ndarray
            Rows where the feature value is ``<= threshold``.
        right_dataset : ndarray
            Rows where the feature value is ``> threshold``.
        """
        left_dataset = np.array([row for row in dataset if row[feature_idx] <= threshold])
        right_dataset = np.array([row for row in dataset if row[feature_idx] > threshold])

        return left_dataset, right_dataset

    def _information_gain(
        self, parent_y: TargetVectorArray, left_y: TargetVectorArray, right_y: TargetVectorArray
    ) -> np.float64:
        """
        Compute information gain from splitting `parent_y` into two subsets.

        Information gain is the entropy of the parent minus the
        weighted average entropy of the two children.

        Parameters
        ----------
        parent_y : ndarray
            Target values before the split.
        left_y : ndarray
            Target values in the left child.
        right_y : ndarray
            Target values in the right child.

        Returns
        -------
        float
            Information gain achieved by the split.
        """
        left_weight = len(left_y) / len(parent_y)
        right_weight = len(right_y) / len(parent_y)

        info_gain = self._entropy(parent_y) - (
            left_weight * self._entropy(left_y) + right_weight * self._entropy(right_y)
        )

        return info_gain

    def _entropy(self, y: TargetVectorArray) -> np.float64:
        """
        Compute the Shannon entropy of a set of class labels.

        Parameters
        ----------
        y : ndarray
            Class labels.

        Returns
        -------
        float
            Entropy, in bits (base-2 log).
        """
        entropy_value = np.float64(0.0)

        class_labels = np.unique(y)

        for class_label in class_labels:
            p = len(y[y == class_label]) / len(y)
            entropy_value += -p * np.log2(p)

        return entropy_value

    def fit(self, X_train: FeatureMatrix, y_train: TargetVector) -> "DecisionTree":
        """
        Fit the decision tree to training data.

        Parameters
        ----------
        X_train : array-like of shape (n_samples, n_features)
            Training feature matrix.
        y_train : array-like of shape (n_samples,)
            Training class labels.

        Returns
        -------
        self : DecisionTree
            The fitted estimator.
        """
        X_train = np.asarray(X_train, dtype=np.float64)
        y_train = np.asarray(y_train, dtype=np.float64)

        dataset = np.concatenate([X_train, y_train.reshape(-1, 1)], axis=1)
        self.root = self._build_tree(dataset=dataset)

        return self

    def predict(self, X_test: FeatureMatrix) -> ClassLabels:
        """
        Predict class labels for new samples.

        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            Samples to predict on.

        Returns
        -------
        ndarray of shape (n_samples,)
            Predicted class labels.
        """
        X_test = np.asarray(X_test, dtype=np.float64)
        predictions = np.array([self._predict_row(row, self.root) for row in X_test])
        return predictions

    def _predict_row(self, row: FeatureMatrixArray, node: "_Node") -> np.int64:
        """
        Traverse the tree to predict a single sample's class.

        Parameters
        ----------
        row : ndarray of shape (n_features,)
            A single sample's feature values.
        node : _Node
            Current node in the traversal, starting at `self.root`.

        Returns
        -------
        int
            Predicted class label from the reached leaf node.
        """
        if node.value is not None:
            return node.value

        feature_val = row[node.feature_idx]

        if feature_val <= node.threshold:
            return self._predict_row(row, node.left)
        else:
            return self._predict_row(row, node.right)
