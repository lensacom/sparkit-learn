import shutil
import tempfile

import numpy as np
import scipy.linalg as ln
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


class TruncatedSVDTestCase(SplearnTestCase):

    def setUp(self):
        super(TruncatedSVDTestCase, self).setUp()
        self.outputdir = tempfile.mkdtemp()

    def tearDown(self):
        super(TruncatedSVDTestCase, self).tearDown()
        shutil.rmtree(self.outputdir)

    def generate_dataset(self, shape=(1e3, 10), block_size=None):
        rng = np.random.RandomState(42)
        X = rng.randn(*shape)
        X_rdd = ArrayRDD(self.sc.parallelize(X, 2), block_size)
        return X, X_rdd


class TestTruncatedSVD(TruncatedSVDTestCase):

    def test_svd(self):
        X, X_rdd = self.generate_dataset()
        u, s, v = svd(X_rdd, 1)
        u = np.squeeze(np.concatenate(np.array(u.collect()))).T
        u_true, s_true, v_true = ln.svd(X)
        assert_array_almost_equal(v[0], match_sign(v[0], v_true[0, :]))
        assert_array_almost_equal(s[0], s_true[0])
        assert_array_almost_equal(u, match_sign(u, u_true[:, 0]))

    def test_svd_em(self):
        X, X_rdd = self.generate_dataset((10, 3))
        u, s, v = svd_em(X_rdd, 1, seed=42)
        u = np.squeeze(np.concatenate(np.array(u.collect()))).T
        u_true, s_true, v_true = ln.svd(X)
        tol = 1
        assert_array_almost_equal(v[0], match_sign(v[0], v_true[0, :]), tol)
        assert_array_almost_equal(s[0], s_true[0], tol)
        assert_array_almost_equal(u, match_sign(u, u_true[:, 0]), tol)

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

        assert_array_equal(v.shape, v_true.shape)
        assert_array_almost_equal(v[0], match_sign(v[0], v_true[0, :]),
                                  decimal=2)

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

        assert_array_equal(Z_local.shape, Z_dist.shape)
        assert_array_almost_equal(Z_dist[:, 0],
                                  match_sign(Z_dist[:, 0], Z_local[:, 0]),
                                  decimal=2)

        Z_local = local.transform(X)
        Z_dist = dist.transform(X_rdd).toarray()

        assert_array_equal(Z_local.shape, Z_dist.shape)
        assert_array_almost_equal(Z_dist[:, 0],
                                  match_sign(Z_dist[:, 0], Z_local[:, 0]),
                                  decimal=2)
