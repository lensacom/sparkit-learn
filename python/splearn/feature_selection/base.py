# -*- coding: utf-8 -*-
"""Generic feature selection mixin"""

from abc import ABCMeta

import numpy as np
from scipy.sparse import csc_matrix, issparse
from sklearn.externals import six
from sklearn.feature_selection.base import SelectorMixin
from sklearn.utils import check_array, safe_mask

from ..base import SparkBroadcasterMixin, SparkTransformerMixin


class SparkSelectorMixin(six.with_metaclass(ABCMeta, SelectorMixin,
                                            SparkTransformerMixin,
                                            SparkBroadcasterMixin)):

    """
    Tranformer mixin that performs feature selection given a support mask

    This mixin provides a feature selector implementation with `transform` and
    `inverse_transform` functionality given an implementation of
    `_get_support_mask`.
    """

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
        support_mask = self._broadcast(Z._rdd.ctx, 'mask', self.get_support)

        def mapper(X, mask=support_mask):
            X = check_array(X, accept_sparse='csr')
            if len(mask.value) != X.shape[1]:
                raise ValueError("X has a different shape than during fitting.")
            return check_array(X, accept_sparse='csr')[:, safe_mask(X, mask.value)]

        return Z.transform(mapper, column='X')

    def inverse_transform(self, Z):
        raise NotImplementedError("Inverse transformation hasn't been implemented yet.")

    # def inverse_transform(self, X):
    #     """
    #     Reverse the transformation operation

    #     Parameters
    #     ----------
    #     X : array of shape [n_samples, n_selected_features]
    #         The input samples.

    #     Returns
    #     -------
    #     X_r : array of shape [n_samples, n_original_features]
    #         `X` with columns of zeros inserted where features would have
    #         been removed by `transform`.
    #     """
    #     if issparse(X):
    #         X = X.tocsc()
    #         # insert additional entries in indptr:
    #         # e.g. if transform changed indptr from [0 2 6 7] to [0 2 3]
    #         # col_nonzeros here will be [2 0 1] so indptr becomes [0 2 2 3]
    #         col_nonzeros = self.inverse_transform(np.diff(X.indptr)).ravel()
    #         indptr = np.concatenate([[0], np.cumsum(col_nonzeros)])
    #         Xt = csc_matrix((X.data, X.indices, indptr),
    #                         shape=(X.shape[0], len(indptr) - 1), dtype=X.dtype)
    #         return Xt

    #     support = self.get_support()
    #     X = check_array(X)
    #     if support.sum() != X.shape[1]:
    #         raise ValueError("X has a different shape than during fitting.")

    #     if X.ndim == 1:
    #         X = X[None, :]
    #     Xt = np.zeros((X.shape[0], support.size), dtype=X.dtype)
    #     Xt[:, support] = X
    #     return Xt
