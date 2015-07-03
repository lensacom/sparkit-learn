import numpy as np
import scipy.sparse as sp
from sklearn.random_projection import (GaussianRandomProjection,
                                       SparseRandomProjection)
from splearn.random_projection import (SparkGaussianRandomProjection,
                                       SparkSparseRandomProjection)
from splearn.rdd import DictRDD
from splearn.utils.testing import (SplearnTestCase, assert_array_almost_equal,
                                   assert_true)
from splearn.utils.validation import check_rdd_dtype


class TestGaussianRandomProjection(SplearnTestCase):

    def test_same_components(self):
        local = GaussianRandomProjection(n_components=20, random_state=42)
        dist = SparkGaussianRandomProjection(n_components=20, random_state=42)

        shapes = [((1e3, 50), None),
                  ((1e4, 100), 600)]

        for shape, block_size in shapes:
            X_dense, X_dense_rdd = self.make_dense_rdd(shape, block_size)
            X_sparse, X_sparse_rdd = self.make_sparse_rdd(shape, block_size)
            Z = DictRDD([X_sparse_rdd, X_dense_rdd], columns=('X', 'Y'))

            local.fit(X_dense)
            dist.fit(X_dense_rdd)
            assert_array_almost_equal(local.components_, dist.components_)

            local.fit(X_sparse)
            dist.fit(X_sparse_rdd)
            assert_array_almost_equal(local.components_, dist.components_)

            dist.fit(Z)
            assert_array_almost_equal(local.components_, dist.components_)

    def test_same_transform_result(self):
        local = GaussianRandomProjection(n_components=4, random_state=42)
        dist = SparkGaussianRandomProjection(n_components=4, random_state=42)

        X_dense, X_dense_rdd = self.make_dense_rdd()
        X_sparse, X_sparse_rdd = self.make_sparse_rdd()
        Z_rdd = DictRDD([X_sparse_rdd, X_dense_rdd], columns=('X', 'Y'))

        result_local = local.fit_transform(X_dense)
        result_dist = dist.fit_transform(X_dense_rdd)
        assert_true(check_rdd_dtype(result_dist, (np.ndarray,)))
        assert_array_almost_equal(result_local, result_dist.toarray())

        result_local = local.fit_transform(X_sparse)
        result_dist = dist.fit_transform(X_sparse_rdd)
        assert_true(check_rdd_dtype(result_dist, (np.ndarray,)))
        assert_array_almost_equal(result_local, result_dist.toarray())

        result_dist = dist.fit_transform(Z_rdd)[:, 'X']
        assert_true(check_rdd_dtype(result_dist, (np.ndarray,)))
        assert_array_almost_equal(result_local, result_dist.toarray())


class TestSparseRandomProjection(SplearnTestCase):

    def test_same_components(self):
        local = SparseRandomProjection(n_components=20, random_state=42)
        dist = SparkSparseRandomProjection(n_components=20, random_state=42)

        shapes = [((1e3, 50), None),
                  ((1e4, 100), 600)]

        for shape, block_size in shapes:
            X_dense, X_dense_rdd = self.make_dense_rdd(shape, block_size)
            X_sparse, X_sparse_rdd = self.make_sparse_rdd(shape, block_size)
            Z = DictRDD([X_sparse_rdd, X_dense_rdd], columns=('X', 'Y'))

            local.fit(X_dense)
            dist.fit(X_dense_rdd)
            assert_array_almost_equal(local.components_.toarray(),
                                      dist.components_.toarray())

            local.fit(X_sparse)
            dist.fit(X_sparse_rdd)
            assert_array_almost_equal(local.components_.toarray(),
                                      dist.components_.toarray())

            dist.fit(Z)
            assert_array_almost_equal(local.components_.toarray(),
                                      dist.components_.toarray())

    def test_same_transform_result(self):
        local = SparseRandomProjection(n_components=4, random_state=42)
        dist = SparkSparseRandomProjection(n_components=4, random_state=42)

        X_dense, X_dense_rdd = self.make_dense_rdd()
        X_sparse, X_sparse_rdd = self.make_sparse_rdd()
        Z_rdd = DictRDD([X_sparse_rdd, X_dense_rdd], columns=('X', 'Y'))

        result_local = local.fit_transform(X_dense)
        result_dist = dist.fit_transform(X_dense_rdd)
        assert_true(check_rdd_dtype(result_dist, (np.ndarray,)))
        assert_array_almost_equal(result_local, result_dist.toarray())

        result_local = local.fit_transform(X_sparse)
        result_dist = dist.fit_transform(X_sparse_rdd)
        assert_true(check_rdd_dtype(result_dist, (sp.spmatrix,)))
        assert_array_almost_equal(result_local.toarray(), result_dist.toarray())

        result_dist = dist.fit_transform(Z_rdd)[:, 'X']
        assert_true(check_rdd_dtype(result_dist, (sp.spmatrix,)))
        assert_array_almost_equal(result_local.toarray(), result_dist.toarray())
