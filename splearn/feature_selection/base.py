# -*- coding: utf-8 -*-
"""Generic feature selection mixin"""

from abc import ABCMeta

import numpy as np
import scipy.sparse as sp
from sklearn.externals import six
from sklearn.feature_selection.base import \
    SelectorMixin as SklearnSelectorMixin

from ..base import BroadcasterMixin, TransformerMixin
from ..rdd import DictRDD
from ..utils.validation import check_rdd


class SelectorMixin(six.with_metaclass(ABCMeta, BroadcasterMixin,
                                       TransformerMixin,
                                       SklearnSelectorMixin)):

    """
    Tranformer mixin that performs feature selection given a support mask

    This mixin provides a feature selector implementation with `transform` and
    `inverse_transform` functionality given an implementation of
    `_get_support_mask`.
    """

    def spark_transform(self, Z):
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
            super(SelectorMixin, self).transform, Z.context)
        return Z.transform(mapper, column='X')
