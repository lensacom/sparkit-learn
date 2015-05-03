import shutil
import tempfile

import numpy as np
import scipy.linalg as ln
import scipy.sparse as sp
from common import SplearnTestCase
from numpy.testing import assert_array_almost_equal, assert_array_equal
from sklearn.decomposition import TruncatedSVD
from splearn.decomposition import SparkTruncatedSVD
from splearn.decomposition.truncated_svd import svd, svd_em
from splearn.rdd import ArrayRDD


def match_sign(a, b):
    a_sign = np.sign(a)
    b_sign = np.sign(b)
    if np.array_equal(a_sign, -b_sign):
        return -b
    elif np.array_equal(a_sign, b_sign):
        return b
    else:
        raise AssertionError("inconsistent matching of sign")


class SVDTestCase(SplearnTestCase):

    def setUp(self):
        super(SVDTestCase, self).setUp()
        self.outputdir = tempfile.mkdtemp()

    def tearDown(self):
        super(SVDTestCase, self).tearDown()
        shutil.rmtree(self.outputdir)

    def generate_dataset(self, shape=(1e3, 10), block_size=None):
        rng = np.random.RandomState(2)
        X = rng.randn(*shape)
        X_rdd = ArrayRDD(self.sc.parallelize(X, 4), block_size)
        return X, X_rdd

    def generate_sparse_dataset(self, shape=(1e3, 10), block_size=None):
        data = sp.rand(shape[0], shape[1], random_state=2, density=0.1).toarray()
        X = [sp.csr_matrix([row]) for row in data]
        X_rdd = ArrayRDD(self.sc.parallelize(X, 4), block_size)
        return data, X_rdd


class TestSVDFunctions(SVDTestCase):

    def test_svd(self):
        X, X_rdd = self.generate_dataset()
        u, s, v = svd(X_rdd, 1)
        u = np.squeeze(np.concatenate(np.array(u.collect()))).T
        u_true, s_true, v_true = ln.svd(X)
        assert_array_almost_equal(v[0], match_sign(v[0], v_true[0, :]))
        assert_array_almost_equal(s[0], s_true[0])
        assert_array_almost_equal(u, match_sign(u, u_true[:, 0]))

    def test_svd_em(self):
        X, X_rdd = self.generate_dataset((1e3, 4))
        u, s, v = svd_em(X_rdd, 1, seed=42, maxiter=50)
        u = np.squeeze(np.concatenate(np.array(u.collect()))).T
        u_true, s_true, v_true = ln.svd(X)
        tol = 1e-1
        assert(np.allclose(s[0], s_true[0], atol=tol))
        assert(np.allclose(+v, v_true[0, :], atol=tol) |
               np.allclose(-v, v_true[0, :], atol=tol))
        assert(np.allclose(+u, u_true[:, 0], atol=tol) |
               np.allclose(-u, u_true[:, 0], atol=tol))

    def test_svd_em_sparse(self):
        X_dense, X_rdd = self.generate_sparse_dataset((1e3, 4))
        u, s, v = svd_em(X_rdd, 1, seed=42, maxiter=50)
        u = np.squeeze(np.concatenate(np.array(u.collect()))).T
        u_true, s_true, v_true = ln.svd(X_dense)
        tol = 1e-1
        assert(np.allclose(s[0], s_true[0], atol=tol))
        assert(np.allclose(+v, v_true[0, :], atol=tol) |
               np.allclose(-v, v_true[0, :], atol=tol))
        assert(np.allclose(+u, u_true[:, 0], atol=tol) |
               np.allclose(-u, u_true[:, 0], atol=tol))


class TestTruncatedSVD(SVDTestCase):

    def test_same_components(self):
        X, X_rdd = self.generate_dataset((1e3, 10))

        n_components = 2
        random_state = 42
        tol = 1e-7
        local = TruncatedSVD(n_components, n_iter=5, tol=tol,
                             random_state=random_state)
        dist = SparkTruncatedSVD(n_components, n_iter=50, tol=tol,
                                 random_state=random_state)

        local.fit(X)
        dist.fit(X_rdd)

        v_true = local.components_
        v = dist.components_

        tol = 1e-1
        assert(np.allclose(+v[0], v_true[0, :], atol=tol) |
               np.allclose(-v[0], v_true[0, :], atol=tol))

    def test_same_fit_transforms(self):
        X, X_rdd = self.generate_dataset((1e3, 12))

        n_components = 4
        random_state = 42
        tol = 1e-7
        local = TruncatedSVD(n_components, n_iter=5, tol=tol,
                             random_state=random_state)
        dist = SparkTruncatedSVD(n_components, n_iter=50, tol=tol,
                                 random_state=random_state)

        Z_local = local.fit_transform(X)
        Z_dist = dist.fit_transform(X_rdd).toarray()

        tol = 1e-1
        assert_array_equal(Z_local.shape, Z_dist.shape)
        assert(np.allclose(+Z_dist[:, 0], Z_local[:, 0], atol=tol) |
               np.allclose(-Z_dist[:, 0], Z_local[:, 0], atol=tol))

        # assert_array_almost_equal(Z_dist[:, 0],
        #                           match_sign(Z_dist[:, 0], Z_local[:, 0]),
        #                           decimal=2)

        # Z_local = local.transform(X)
        # Z_dist = dist.transform(X_rdd).toarray()

        # assert_array_equal(Z_local.shape, Z_dist.shape)
        # assert_array_almost_equal(Z_dist[:, 0],
        #                           match_sign(Z_dist[:, 0], Z_local[:, 0]),
        #                           decimal=2)
