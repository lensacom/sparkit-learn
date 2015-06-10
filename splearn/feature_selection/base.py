# -*- coding: utf-8 -*-
"""Generic feature selection mixin"""

from abc import ABCMeta

from sklearn.externals import six
from sklearn.feature_selection.base import SelectorMixin

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

    pass
