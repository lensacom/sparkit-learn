import numpy as np
import scipy.sparse as sp
from sklearn.feature_selection import VarianceThreshold
from sklearn.utils import check_array
from sklearn.utils.sparsefuncs import mean_variance_axis

from ..rdd import DictRDD
from ..utils.validation import check_rdd
from .base import SparkSelectorMixin


class SparkVarianceThreshold(VarianceThreshold, SparkSelectorMixin):

    """Feature selector that removes all low-variance features.

    This feature selection algorithm looks only at the features (X), not the
    desired outputs (y), and can thus be used for unsupervised learning.

    Parameters
    ----------
    threshold : float, optional
        Features with a training-set variance lower than this threshold will
        be removed. The default is to keep all features with non-zero variance,
        i.e. remove the features that have the same value in all samples.

    Attributes
    ----------
    variances_ : array, shape (n_features,)
        Variances of individual features.

    Examples
    --------
    The following dataset has integer features, two of which are the same
    in every sample. These are removed with the default setting for threshold::

        >>> X = [[0, 2, 0, 3], [0, 1, 4, 3], [0, 1, 1, 3]]
        >>> selector = VarianceThreshold()
        >>> selector.fit_transform(X)
        array([[2, 0],
               [1, 4],
               [1, 1]])
    """

    __transient__ = ['variances_']

    def fit(self, Z):
        """Learn empirical variances from X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Sample vectors from which to compute variances.

        y : any
            Ignored. This parameter exists only for compatibility with
            sklearn.pipeline.Pipeline.

        Returns
        -------
        self
        """

        X = Z[:, 'X'] if isinstance(Z, DictRDD) else Z
        check_rdd(X, (np.ndarray, sp.spmatrix))

        def mapper(X):
            """Calculate statistics for every numpy or scipy blocks."""
            X = check_array(X, ('csr', 'csc'), dtype=np.float64)
            if hasattr(X, "toarray"):   # sparse matrix
                mean, var = mean_variance_axis(X, axis=0)
            else:
                mean, var = np.mean(X, axis=0), np.var(X, axis=0)
            return X.shape[0], mean, var

        def reducer(a, b):
            """Calculate the combined statistics."""
            n_a, mean_a, var_a = a
            n_b, mean_b, var_b = b
            n_ab = n_a + n_b
            mean_ab = ((mean_a * n_a) + (mean_b * n_b)) / n_ab
            var_ab = (((n_a * var_a) + (n_b * var_b)) / n_ab) + \
                     ((n_a * n_b) * ((mean_b - mean_a) / n_ab) ** 2)
            return (n_ab, mean_ab, var_ab)

        _, _, self.variances_ = X.map(mapper).treeReduce(reducer)

        if np.all(self.variances_ <= self.threshold):
            msg = "No feature in X meets the variance threshold {0:.5f}"
            if X.shape[0] == 1:
                msg += " (X contains only one sample)"
            raise ValueError(msg.format(self.threshold))

        return self

    def transform(self, Z):
        """Reduce X to the selected features.

        Parameters
        ----------
        X : array of shape [n_samples, n_features]
            The input samples.

        Returns
        -------
        X_r : array of shape [n_samples, n_selected_features]
            The input samples with only the selected features.
        """
        X = Z[:, 'X'] if isinstance(Z, DictRDD) else Z
        check_rdd(X, (np.ndarray, sp.spmatrix))

        mapper = self.broadcast(
            super(SparkVarianceThreshold, self).transform, Z.context)
        return Z.transform(mapper, column='X')
