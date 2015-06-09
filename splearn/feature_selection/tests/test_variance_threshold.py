import numpy as np
import scipy.sparse as sp
from numpy.testing import assert_array_almost_equal
from sklearn.feature_selection import VarianceThreshold
from splearn.feature_selection import SparkVarianceThreshold
from splearn.rdd import DictRDD
from splearn.utils.testing import SplearnTestCase


class TestVarianceThreshold(SplearnTestCase):

    def test_same_variances(self):
        local = VarianceThreshold()
        dist = SparkVarianceThreshold()

        shapes = [((10, 5), None),
                  ((1e3, 20), None),
                  ((1e3, 20), 100),
                  ((1e4, 100), None),
                  ((1e4, 100), 600)]

        for shape, block_size in shapes:
            X_dense, X_dense_rdd = self.make_dense_rdd()
            X_sparse, X_sparse_rdd = self.make_sparse_rdd()
            Z = DictRDD([X_sparse_rdd, X_dense_rdd], columns=('X', 'Y'))

            local.fit(X_dense)
            dist.fit(X_dense_rdd)
            assert_array_almost_equal(local.variances_, dist.variances_)

            local.fit(X_sparse)
            dist.fit(X_sparse_rdd)
            assert_array_almost_equal(local.variances_, dist.variances_)

            dist.fit(Z)
            assert_array_almost_equal(local.variances_, dist.variances_)

    def test_same_transform_result(self):
        local = VarianceThreshold()
        dist = SparkVarianceThreshold()

        X_dense, X_dense_rdd = self.make_dense_rdd()
        X_sparse, X_sparse_rdd = self.make_sparse_rdd()
        Z_rdd = DictRDD([X_sparse_rdd, X_dense_rdd], columns=('X', 'Y'))

        result_local = local.fit_transform(X_dense)
        result_dist = np.vstack(dist.fit_transform(X_dense_rdd).collect())
        assert_array_almost_equal(result_local, result_dist)

        result_local = local.fit_transform(X_sparse)
        result_dist = sp.vstack(dist.fit_transform(X_sparse_rdd).collect())
        assert_array_almost_equal(result_local.toarray(),
                                  result_dist.toarray())

        result_dist = sp.vstack(dist.fit_transform(Z_rdd)[:, 'X'].collect())
        assert_array_almost_equal(result_local.toarray(),
                                  result_dist.toarray())

    def test_same_transform_with_treshold(self):
        local = VarianceThreshold(.03)
        dist = SparkVarianceThreshold(.03)

        X_dense, X_dense_rdd = self.make_dense_rdd()
        X_sparse, X_sparse_rdd = self.make_sparse_rdd()
        Z_rdd = DictRDD([X_sparse_rdd, X_dense_rdd], columns=('X', 'Y'))

        result_local = local.fit_transform(X_dense)
        result_dist = np.vstack(dist.fit_transform(X_dense_rdd).collect())
        assert_array_almost_equal(result_local, result_dist)

        result_local = local.fit_transform(X_sparse)
        result_dist = sp.vstack(dist.fit_transform(X_sparse_rdd).collect())
        assert_array_almost_equal(result_local.toarray(),
                                  result_dist.toarray())
        result_dist = sp.vstack(dist.fit_transform(Z_rdd)[:, 'X'].collect())
        assert_array_almost_equal(result_local.toarray(),
                                  result_dist.toarray())
