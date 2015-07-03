import warnings

import numpy as np
import scipy.sparse as sp
from numpy.testing import assert_equal
from sklearn.random_projection import (BaseRandomProjection,
                                       GaussianRandomProjection,
                                       SparseRandomProjection,
                                       johnson_lindenstrauss_min_dim)
from sklearn.utils import DataDimensionalityWarning

from .base import SparkBroadcasterMixin
from .rdd import DictRDD
from .utils.validation import check_rdd


class SparkBaseRandomProjection(BaseRandomProjection, SparkBroadcasterMixin):

    __transient__ = ['components_']

    def fit(self, Z):
        """Generate a sparse random projection matrix
        Parameters
        ----------
        X : numpy array or scipy.sparse of shape [n_samples, n_features]
            Training set: only the shape is used to find optimal random
            matrix dimensions based on the theory referenced in the
            afore mentioned papers.
        y : is not used: placeholder to allow for usage in a Pipeline.
        Returns
        -------
        self
        """
        X = Z[:, 'X'] if isinstance(Z, DictRDD) else Z
        check_rdd(X, (np.ndarray, sp.spmatrix))

        n_samples, n_features = X.shape

        if self.n_components == 'auto':
            self.n_components_ = johnson_lindenstrauss_min_dim(
                n_samples=n_samples, eps=self.eps)

            if self.n_components_ <= 0:
                raise ValueError(
                    'eps=%f and n_samples=%d lead to a target dimension of '
                    '%d which is invalid' % (
                        self.eps, n_samples, self.n_components_))

            elif self.n_components_ > n_features:
                raise ValueError(
                    'eps=%f and n_samples=%d lead to a target dimension of '
                    '%d which is larger than the original space with '
                    'n_features=%d' % (self.eps, n_samples, self.n_components_,
                                       n_features))
        else:
            if self.n_components <= 0:
                raise ValueError("n_components must be greater than 0, got %s"
                                 % self.n_components_)

            elif self.n_components > n_features:
                warnings.warn(
                    "The number of components is higher than the number of"
                    " features: n_features < n_components (%s < %s)."
                    "The dimensionality of the problem will not be reduced."
                    % (n_features, self.n_components),
                    DataDimensionalityWarning)

            self.n_components_ = self.n_components

        # Generate a projection matrix of size [n_components, n_features]
        self.components_ = self._make_random_matrix(self.n_components_,
                                                    n_features)

        # Check contract
        assert_equal(
            self.components_.shape,
            (self.n_components_, n_features),
            err_msg=('An error has occurred the self.components_ matrix has '
                     ' not the proper shape.'))

        return self

    def transform(self, Z):
        """Project the data by using matrix product with the random matrix
        Parameters
        ----------
        X : numpy array or scipy.sparse of shape [n_samples, n_features]
            The input data to project into a smaller dimensional space.
        y : is not used: placeholder to allow for usage in a Pipeline.
        Returns
        -------
        X_new : numpy array or scipy sparse of shape [n_samples, n_components]
            Projected array.
        """
        X = Z[:, 'X'] if isinstance(Z, DictRDD) else Z
        check_rdd(X, (np.ndarray, sp.spmatrix))

        dtype = np.ndarray if self.dense_output else None
        mapper = self.broadcast(
            super(SparkBaseRandomProjection, self).transform, Z.context)
        return Z.transform(mapper, column='X', dtype=dtype)


class SparkGaussianRandomProjection(GaussianRandomProjection,
                                    SparkBaseRandomProjection):
    pass


class SparkSparseRandomProjection(SparseRandomProjection,
                                  SparkBaseRandomProjection):
    pass
