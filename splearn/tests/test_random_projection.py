import numpy as np
import scipy.sparse as sp
from sklearn.random_projection import \
    GaussianRandomProjection as SklearnGaussianRandomProjection
from sklearn.random_projection import \
    SparseRandomProjection as SklearnSparseRandomProjection
from splearn.random_projection import (GaussianRandomProjection,
                                       SparseRandomProjection)
from splearn.rdd import DictRDD
from splearn.utils.testing import (SplearnTestCase, assert_array_almost_equal,
                                   assert_true)
from splearn.utils.validation import check_rdd_dtype


class TestGaussianRandomProjection(SplearnTestCase):

    def test_state(self):
        scikit = SklearnGaussianRandomProjection(n_components=20,
                                                 random_state=42)
        sparkit = GaussianRandomProjection(n_components=20, random_state=42)

        shapes = [((1e3, 50), -1),
                  ((1e4, 100), 600)]

        for shape, block_size in shapes:
            X_dense, X_dense_rdd = self.make_dense_rdd(shape, block_size)
            X_sparse, X_sparse_rdd = self.make_sparse_rdd(shape, block_size)
            Z = DictRDD([X_sparse_rdd, X_dense_rdd], columns=('X', 'Y'))

            scikit.fit(X_dense)
            sparkit.fit(X_dense)
            assert_array_almost_equal(scikit.components_, sparkit.components_)
            sparkit.fit(X_dense_rdd)
            assert_array_almost_equal(scikit.components_, sparkit.components_)

            scikit.fit(X_sparse)
            sparkit.fit(X_sparse)
            assert_array_almost_equal(scikit.components_, sparkit.components_)
            sparkit.fit(X_sparse_rdd)
            assert_array_almost_equal(scikit.components_, sparkit.components_)

            sparkit.fit(Z)
            assert_array_almost_equal(scikit.components_, sparkit.components_)

    def test_dense(self):
        X_dense, X_dense_rdd = self.make_dense_rdd()

        scikit = SklearnGaussianRandomProjection(n_components=4,
                                                 random_state=42)
        sparkit = GaussianRandomProjection(n_components=4, random_state=42)

        T_scikit = scikit.fit_transform(X_dense)

        T_sparkit = sparkit.fit_transform(X_dense)
        assert_array_almost_equal(T_scikit, T_sparkit)

        T_sparkit = sparkit.fit_transform(X_dense_rdd)
        assert_true(check_rdd_dtype(T_sparkit, (np.ndarray,)))
        assert_array_almost_equal(T_scikit, T_sparkit.toarray())

    def test_sparse(self):
        X_sparse, X_sparse_rdd = self.make_sparse_rdd()

        scikit = SklearnGaussianRandomProjection(n_components=4,
                                                 random_state=42)
        sparkit = GaussianRandomProjection(n_components=4, random_state=42)

        T_scikit = scikit.fit_transform(X_sparse)

        T_sparkit = sparkit.fit_transform(X_sparse)
        assert_array_almost_equal(T_scikit, T_sparkit)

        T_sparkit = sparkit.fit_transform(X_sparse_rdd)
        assert_true(check_rdd_dtype(T_sparkit, (np.ndarray,)))
        assert_array_almost_equal(T_scikit, T_sparkit.toarray())

    def test_dict(self):
        X_dense, X_dense_rdd = self.make_dense_rdd()
        X_sparse, X_sparse_rdd = self.make_sparse_rdd()
        Z_rdd = DictRDD([X_sparse_rdd, X_dense_rdd], columns=('X', 'Y'))

        scikit = SklearnGaussianRandomProjection(n_components=4,
                                                 random_state=42)
        sparkit = GaussianRandomProjection(n_components=4, random_state=42)

        T_scikit = scikit.fit_transform(X_sparse)
        T_sparkit = sparkit.fit_transform(Z_rdd)[:, 'X']
        assert_true(check_rdd_dtype(T_sparkit, (np.ndarray,)))
        assert_array_almost_equal(T_scikit, T_sparkit.toarray())


class TestSparseRandomProjection(SplearnTestCase):

    def test_state(self):
        scikit = SklearnSparseRandomProjection(n_components=20, random_state=42)
        sparkit = SparseRandomProjection(n_components=20, random_state=42)

        shapes = [((1e3, 50), -1),
                  ((1e4, 100), 600)]

        for shape, block_size in shapes:
            X_dense, X_dense_rdd = self.make_dense_rdd(shape, block_size)
            X_sparse, X_sparse_rdd = self.make_sparse_rdd(shape, block_size)
            Z = DictRDD([X_sparse_rdd, X_dense_rdd], columns=('X', 'Y'))

            scikit.fit(X_dense)
            sparkit.fit(X_dense)
            assert_array_almost_equal(scikit.components_.toarray(),
                                      sparkit.components_.toarray())
            sparkit.fit(X_dense_rdd)
            assert_array_almost_equal(scikit.components_.toarray(),
                                      sparkit.components_.toarray())

            scikit.fit(X_sparse)
            sparkit.fit(X_sparse)
            assert_array_almost_equal(scikit.components_.toarray(),
                                      sparkit.components_.toarray())
            sparkit.fit(X_sparse_rdd)
            assert_array_almost_equal(scikit.components_.toarray(),
                                      sparkit.components_.toarray())

            sparkit.fit(Z)
            assert_array_almost_equal(scikit.components_.toarray(),
                                      sparkit.components_.toarray())

    def test_dense(self):
        X_dense, X_dense_rdd = self.make_dense_rdd()

        scikit = SklearnSparseRandomProjection(n_components=4, random_state=42)
        sparkit = SparseRandomProjection(n_components=4, random_state=42)

        T_scikit = scikit.fit_transform(X_dense)

        T_sparkit = sparkit.fit_transform(X_dense)
        assert_array_almost_equal(T_scikit, T_sparkit)

        T_sparkit = sparkit.fit_transform(X_dense_rdd)
        assert_true(check_rdd_dtype(T_sparkit, (np.ndarray,)))
        assert_array_almost_equal(T_scikit, T_sparkit.toarray())

    def test_sparse(self):
        X_sparse, X_sparse_rdd = self.make_sparse_rdd()

        scikit = SklearnSparseRandomProjection(n_components=4, random_state=42)
        sparkit = SparseRandomProjection(n_components=4, random_state=42)

        T_scikit = scikit.fit_transform(X_sparse)

        T_sparkit = sparkit.fit_transform(X_sparse)
        assert_true(sp.issparse(T_sparkit))
        assert_array_almost_equal(T_scikit.toarray(), T_sparkit.toarray())

        T_sparkit = sparkit.fit_transform(X_sparse_rdd)
        assert_true(check_rdd_dtype(T_sparkit, (sp.spmatrix,)))
        assert_array_almost_equal(T_scikit.toarray(), T_sparkit.toarray())

    def test_dict(self):
        X_dense, X_dense_rdd = self.make_dense_rdd()
        X_sparse, X_sparse_rdd = self.make_sparse_rdd()
        Z_rdd = DictRDD([X_sparse_rdd, X_dense_rdd], columns=('X', 'Y'))

        scikit = SklearnSparseRandomProjection(n_components=4, random_state=42)
        sparkit = SparseRandomProjection(n_components=4, random_state=42)

        T_scikit = scikit.fit_transform(X_sparse)
        T_sparkit = sparkit.fit_transform(Z_rdd)[:, 'X']
        assert_true(check_rdd_dtype(T_sparkit, (sp.spmatrix,)))
        assert_array_almost_equal(T_scikit.toarray(), T_sparkit.toarray())
