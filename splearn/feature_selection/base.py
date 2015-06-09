# -*- coding: utf-8 -*-
"""Generic feature selection mixin"""

from abc import ABCMeta

import numpy as np
from pyspark import Broadcast
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

    @property
    def support_mask_(self):
        if not hasattr(self, '_bc_support_mask'):
            return self._get_support_mask()
        elif isinstance(self._bc_support_mask, Broadcast):
            return self._bc_support_mask.value
        else:
            return self._bc_support_mask

    @support_mask_.setter
    def support_mask_(self, value):
        self._bc_support_mask = value

    def get_support(self, indices=False):
        """
        Get a mask, or integer index, of the features selected

        Parameters
        ----------
        indices : boolean (default False)
            If True, the return value will be an array of integers, rather
            than a boolean mask.

        Returns
        -------
        support : array
            An index that selects the retained features from a feature vector.
            If `indices` is False, this is a boolean array of shape
            [# input features], in which an element is True iff its
            corresponding feature is selected for retention. If `indices` is
            True, this is an integer array of shape [# output features] whose
            values are indices into the input feature vector.
        """
        mask = self.support_mask_
        return mask if not indices else np.where(mask)[0]

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
        self._broadcast(Z.context, '_bc_support_mask', self._get_support_mask())
        return Z.transform(
            super(SparkSelectorMixin, self).transform, column='X')

    def inverse_transform(self, Z):
        raise NotImplementedError("Inverse transformation hasn't been"
                                  " implemented yet.")

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
