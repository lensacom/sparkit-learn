# -*- coding: utf-8 -*-
"""

"""
import numpy as np
import scipy.sparse as sp
from sklearn.ensemble import RandomForestClassifier

from ..utils.validation import check_rdd


class SparkRandomForestClassifier(RandomForestClassifier):

    """Distributed version of sklearn's Random Forest Classification.

    A random forest is a meta estimator that fits a number of decision tree
    classifiers on various sub-samples of the dataset and use averaging to
    improve the predictive accuracy and control over-fitting.

    It is 'Pseudo' since bootstrapping is different than excepted. The input
    RDD is split in several part, then random forest are learnt independently
    on each part and then merged

    Parameters
    ----------
    n_estimators : integer, optional (default=10)
        The number of trees in the forest for each independently trained forest
        At the end, there are thus n_estimators * block counts tree

    criterion : string, optional (default="gini")
        The function to measure the quality of a split. Supported criteria are
        "gini" for the Gini impurity and "entropy" for the information gain.
        Note: this parameter is tree-specific.

    max_features : int, float, string or None, optional (default="auto")
        The number of features to consider when looking for the best split:

        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a percentage and
          `int(max_features * n_features)` features are considered at each
          split.
        - If "auto", then `max_features=sqrt(n_features)`.
        - If "sqrt", then `max_features=sqrt(n_features)`.
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.
        Note: this parameter is tree-specific.

    max_depth : integer or None, optional (default=None)
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.
        Ignored if ``max_samples_leaf`` is not None.
        Note: this parameter is tree-specific.

    min_samples_split : integer, optional (default=2)
        The minimum number of samples required to split an internal node.
        Note: this parameter is tree-specific.

    min_samples_leaf : integer, optional (default=1)
        The minimum number of samples in newly created leaves.  A split is
        discarded if after the split, one of the leaves would contain less then
        ``min_samples_leaf`` samples.
        Note: this parameter is tree-specific.

    max_leaf_nodes : int or None, optional (default=None)
        Grow trees with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.
        If not None then ``max_depth`` will be ignored.
        Note: this parameter is tree-specific.

    bootstrap : boolean, optional (default=True)
        Whether bootstrap samples are used when building trees.

    oob_score : bool
        Whether to use out-of-bag samples to estimate
        the generalization error.

    n_jobs : integer, optional (default=1)
        The number of jobs to run in parallel for both `fit` and `predict`.
        If -1, then the number of jobs is set to the number of cores.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    verbose : int, optional (default=0)
        Controls the verbosity of the tree building process.

    Attributes
    ----------
    `estimators_`: list of DecisionTreeClassifier
        The collection of fitted sub-estimators.

    `classes_`: array of shape = [n_classes] or a list of such arrays
        The classes labels (single output problem), or a list of arrays of
        class labels (multi-output problem).

    `n_classes_`: int or list
        The number of classes (single output problem), or a list containing the
        number of classes for each output (multi-output problem).

    `feature_importances_` : array of shape = [n_features]
        The feature importances (the higher, the more important the feature).

    `oob_score_` : float
        Score of the training dataset obtained using an out-of-bag estimate.

    `oob_decision_function_` : array of shape = [n_samples, n_classes]
        Decision function computed with out-of-bag estimate on the training
        set. If n_estimators is small it might be possible that a data point
        was never left out during the bootstrap. In this case,
        `oob_decision_function_` might contain NaN.
    """

    def fit(self, Z, classes=None):
        """Fit the model according to the given training data.

        Parameters
        ----------
        Z : DictRDD containing (X, y) pairs
            X - Training vector
            y - Target labels
        classes : iterable
            The set of available classes

        Returns
        -------
        self : object
            Returns self.
        """
        check_rdd(Z, {'X': (sp.spmatrix, np.ndarray)})
        mapper = lambda X_y: super(SparkRandomForestClassifier, self).fit(
            X_y[0], X_y[1]
        )

        models = Z.map(mapper).collect()

        self.__dict__ = models[0].__dict__
        self.estimators_ = []
        for m in models:
            self.estimators_ += m.estimators_
        self.n_estimators = len(self.estimators_)
        return self

    def predict(self, X):
        """Distributed method to predict class labels for samples in X.

        Parameters
        ----------
        X : ArrayRDD containing {array-like, sparse matrix}
            Samples.

        Returns
        -------
        C : ArrayRDD
            Predicted class label per sample.
        """
        check_rdd(X, (sp.spmatrix, np.ndarray))
        return X.map(lambda X: super(SparkRandomForestClassifier, self).predict(X))

    def to_scikit(self):
        new = RandomForestClassifier()
        new.__dict__ = self.__dict__
        return new
