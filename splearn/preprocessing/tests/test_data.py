# -*- coding: utf-8 -*-
"""

"""
from numpy.testing import assert_array_almost_equal
from sklearn.preprocessing import StandardScaler
from splearn.preprocessing.data import SparkStandardScaler
from splearn.utils.testing import SplearnTestCase


class TestSparkStandardScaler(SplearnTestCase):

    def test_same_fit_transform(self):
        X, X_rdd = self.make_dense_rdd()

        local = StandardScaler()
        dist = SparkStandardScaler()

        X_trans = local.fit_transform(X)
        X_rdd_trans = dist.fit_transform(X_rdd).toarray()
        X_converted = dist.to_scikit().transform(X)

        assert_array_almost_equal(X_trans, X_rdd_trans)
        assert_array_almost_equal(X_trans, X_converted)

        local = StandardScaler(with_mean=False)
        dist = SparkStandardScaler(with_mean=False)

        X_trans = local.fit_transform(X)
        X_rdd_trans = dist.fit_transform(X_rdd).toarray()
        X_converted = dist.to_scikit().transform(X)

        assert_array_almost_equal(X_trans, X_rdd_trans)
        assert_array_almost_equal(X_trans, X_converted)

        local = StandardScaler(with_std=False)
        dist = SparkStandardScaler(with_std=False)

        X_trans = local.fit_transform(X)
        X_rdd_trans = dist.fit_transform(X_rdd).toarray()
        X_converted = dist.to_scikit().transform(X)

        assert_array_almost_equal(X_trans, X_rdd_trans)
        assert_array_almost_equal(X_trans, X_converted)

    def test_same_fit_transform_sparse(self):
        X, X_rdd = self.make_sparse_rdd()

        local = StandardScaler(with_mean=False)
        dist = SparkStandardScaler(with_mean=False)

        X_trans = local.fit_transform(X).toarray()
        X_rdd_trans = dist.fit_transform(X_rdd).toarray()
        X_converted = dist.to_scikit().transform(X).toarray()

        assert_array_almost_equal(X_trans, X_rdd_trans)
        assert_array_almost_equal(X_trans, X_converted)
