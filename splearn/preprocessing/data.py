# -*- coding: utf-8 -*-
"""

"""
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing.data import _handle_zeros_in_scale
from splearn.base import SparkTransformerMixin
import numpy as np
import scipy.sparse as sp
from sklearn.utils.sparsefuncs import mean_variance_axis, inplace_column_scale
from sklearn.utils import check_array

from ..rdd import DictRDD
from ..utils.validation import check_rdd, check_rdd_dtype


class SparkStandardScaler(StandardScaler, SparkTransformerMixin):
    """Standardize features by removing the mean and scaling to unit variance
    Centering and scaling happen independently on each feature by computing
    the relevant statistics on the samples in the training set. Mean and
    standard deviation are then stored to be used on later data using the
    `transform` method.
    Standardization of a dataset is a common requirement for many
    machine learning estimators: they might behave badly if the
    individual feature do not more or less look like standard normally
    distributed data (e.g. Gaussian with 0 mean and unit variance).
    For instance many elements used in the objective function of
    a learning algorithm (such as the RBF kernel of Support Vector
    Machines or the L1 and L2 regularizers of linear models) assume that
    all features are centered around 0 and have variance in the same
    order. If a feature has a variance that is orders of magnitude larger
    that others, it might dominate the objective function and make the
    estimator unable to learn from other features correctly as expected.
    This scaler can also be applied to sparse CSR or CSC matrices by passing
    `with_mean=False` to avoid breaking the sparsity structure of the data.
    Read more in the :ref:`User Guide <preprocessing_scaler>`.
    Parameters
    ----------
    with_mean : boolean, True by default
        If True, center the data before scaling.
        This does not work (and will raise an exception) when attempted on
        sparse matrices, because centering them entails building a dense
        matrix which in common use cases is likely to be too large to fit in
        memory.
    with_std : boolean, True by default
        If True, scale the data to unit variance (or equivalently,
        unit standard deviation).
    copy : boolean, optional, default True
        If False, try to avoid a copy and do inplace scaling instead.
        This is not guaranteed to always work inplace; e.g. if the data is
        not a NumPy array or scipy.sparse CSR matrix, a copy may still be
        returned.
    Attributes
    ----------
    scale_ : ndarray, shape (n_features,)
        Per feature relative scaling of the data.
        .. versionadded:: 0.17
           *scale_* is recommended instead of deprecated *std_*.
    mean_ : array of floats with shape [n_features]
        The mean value for each feature in the training set.
    var_ : array of floats with shape [n_features]
        The variance for each feature in the training set. Used to compute
        `scale_`
    n_samples_seen_ : int
        The number of samples processed by the estimator. Will be reset on
        new calls to fit, but increments across ``partial_fit`` calls.
    See also
    --------
    :func:`sklearn.preprocessing.scale` to perform centering and
    scaling without using the ``Transformer`` object oriented API
    :class:`sklearn.decomposition.RandomizedPCA` with `whiten=True`
    to further remove the linear correlation across features.
    """

    def fit(self, Z):
        """Compute the mean and std to be used for later scaling.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape [n_samples, n_features]
            The data used to compute the mean and standard deviation
            used for later scaling along the features axis.
        y: Passthrough for ``Pipeline`` compatibility.
        """

        # Reset internal state before fitting
        self._reset()
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

        if check_rdd_dtype(X, (sp.spmatrix)):
            if self.with_mean:
                raise ValueError(
                    "Cannot center sparse matrices: pass `with_mean=False` "
                    "instead. See docstring for motivation and alternatives.")
            if self.with_std:
                _, _, self.var_ = X.map(mapper).treeReduce(reducer)
            else:
                self.mean_ = None
                self.var_ = None
        else:
            _, self.mean_, self.var_ = X.map(mapper).treeReduce(reducer)

        if self.with_std:
            self.scale_ = _handle_zeros_in_scale(np.sqrt(self.var_))
        else:
            self.scale_ = None

        return self

    def transform(self, Z):
        """Perform standardization by centering and scaling
        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data used to scale along the features axis.
        """
        X = Z[:, 'X'] if isinstance(Z, DictRDD) else Z
        check_rdd(X, (np.ndarray, sp.spmatrix))

        if check_rdd_dtype(X, (sp.spmatrix)):
            if self.with_mean:
                raise ValueError(
                    "Cannot center sparse matrices: pass `with_mean=False` "
                    "instead. See docstring for motivation and alternatives.")
            if self.scale_ is not None:
                def mapper(X):
                    inplace_column_scale(X, 1 / self.scale_)
                    return X
        else:
            if self.with_mean:
                if self.with_std:
                    def mapper(X):
                        X -= self.mean_
                        X /= self.scale_
                        return X
                else:
                    def mapper(X):
                        X -= self.mean_
                        return X
            else:
                if self.with_std:
                    def mapper(X):
                        X /= self.scale_
                        return X
                else:
                    raise ValueError("Need with_std or with_mean ")
        return Z.transform(mapper, column="X")

    def to_scikit(self):
        scaler = StandardScaler(with_mean=self.with_mean,
                                with_std=self.with_std,
                                copy=self.copy)
        scaler.__dict__ = self.__dict__
        return scaler
