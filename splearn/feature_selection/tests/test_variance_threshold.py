import numpy as np
import scipy.sparse as sp
from sklearn.feature_selection import \
    VarianceThreshold as SklearnVarianceThreshold
from splearn.feature_selection import VarianceThreshold
from splearn.rdd import DictRDD
from splearn.utils.testing import (SplearnTestCase, assert_array_almost_equal,
                                   assert_true)
from splearn.utils.validation import check_rdd_dtype


class TestVarianceThreshold(SplearnTestCase):

    def test_state(self):
        scikit = SklearnVarianceThreshold()
        sparkit = VarianceThreshold()

        shapes = [((10, 5), -1),
                  ((1e3, 20), -1),
                  ((1e3, 20), 100),
                  ((1e4, 100), -1),
                  ((1e4, 100), 600)]

        for shape, block_size in shapes:
            X_dense, X_dense_rdd = self.make_dense_rdd(shape, block_size)
            X_sparse, X_sparse_rdd = self.make_sparse_rdd(shape, block_size)
            Z = DictRDD([X_sparse_rdd, X_dense_rdd], columns=('X', 'Y'))

            scikit.fit(X_dense)
            sparkit.fit(X_dense)
            assert_array_almost_equal(scikit.variances_, sparkit.variances_)

            scikit.fit(X_dense)
            sparkit.fit(X_dense_rdd)
            assert_array_almost_equal(scikit.variances_, sparkit.variances_)

            scikit.fit(X_sparse)
            sparkit.fit(X_sparse_rdd)
            assert_array_almost_equal(scikit.variances_, sparkit.variances_)

            sparkit.fit(Z)
            assert_array_almost_equal(scikit.variances_, sparkit.variances_)

    def test_dense(self):
        X_dense, X_dense_rdd = self.make_dense_rdd()

        for variance in [None, 0.03]:
            scikit = SklearnVarianceThreshold(variance)
            sparkit = VarianceThreshold(variance)

            T_true = scikit.fit_transform(X_dense)

            # Test Array input
            T_local = sparkit.fit_transform(X_dense)
            assert_array_almost_equal(T_true, T_local)

            # Test ArrayRDD input
            T_dist = sparkit.fit_transform(X_dense_rdd)
            assert_true(check_rdd_dtype(T_dist, (np.ndarray,)))
            assert_array_almost_equal(T_true, T_dist.toarray())

    def test_sparse(self):
        X_sparse, X_sparse_rdd = self.make_sparse_rdd()

        for variance in [None, 0.03]:
            scikit = SklearnVarianceThreshold(variance)
            sparkit = VarianceThreshold(variance)

            T_true = scikit.fit_transform(X_sparse).toarray()

            # Test Sparse matrix input
            T_local = sparkit.fit_transform(X_sparse)
            assert_true(sp.issparse(T_local))
            assert_array_almost_equal(T_true, T_local.toarray())

            # Test SparseRDD input
            T_dist = sparkit.fit_transform(X_sparse_rdd)
            assert_true(check_rdd_dtype(T_dist, (sp.spmatrix,)))
            assert_array_almost_equal(T_true, T_dist.toarray())

    def test_dict(self):
        X_dense, X_dense_rdd = self.make_dense_rdd()
        X_sparse, X_sparse_rdd = self.make_sparse_rdd()
        Z_rdd = DictRDD([X_sparse_rdd, X_dense_rdd], columns=('X', 'Y'))

        for variance in [None, 0.03]:
            scikit = SklearnVarianceThreshold(variance)
            sparkit = VarianceThreshold(variance)

            T_true = scikit.fit_transform(X_sparse).toarray()

            # Test DictRDD input
            T_dist = sparkit.fit_transform(Z_rdd)[:, 'X']
            assert_true(check_rdd_dtype(T_dist, (sp.spmatrix,)))
            assert_array_almost_equal(T_true, T_dist.toarray())
