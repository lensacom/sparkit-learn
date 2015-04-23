import shutil
import tempfile

import numpy as np
import scipy.linalg as ln
from common import SplearnTestCase
from numpy.testing import assert_array_almost_equal
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

    # def generate_dataset(self, classes, samples, blocks=None):
    #     X, y = make_classification(n_classes=classes,
    #                                n_samples=samples, n_features=4,
    #                                n_clusters_per_class=1,
    #                                random_state=42)
    #     X = np.abs(X)

    #     X_rdd = self.sc.parallelize(X, 4)
    #     y_rdd = self.sc.parallelize(y, 4)

    #     Z = DictRDD(X_rdd.zip(y_rdd), columns=('X', 'y'), block_size=blocks)

    #     return X, y, Z


class TestTruncatedSVD(TruncatedSVDTestCase):

    def test_svd(self):
        rng = np.random.RandomState(42)
        mat = rng.randn(1e3, 10)
        data = ArrayRDD(self.sc.parallelize(list(mat), 10))
        u, s, v = svd(data, 1)
        u = np.squeeze(np.concatenate(np.array(u.collect()))).T
        u_true, s_true, v_true = ln.svd(mat)
        assert_array_almost_equal(v[0], match_sign(v[0], v_true[0, :]))
        assert_array_almost_equal(s[0], s_true[0])
        assert_array_almost_equal(u, match_sign(u, u_true[:, 0]))

    def test_svd_em(self):
        rng = np.random.RandomState(42)
        mat = rng.randn(10, 3)
        data = ArrayRDD(self.sc.parallelize(list(mat), 2)).cache()
        u, s, v = svd_em(data, 1, seed=42)
        u = np.squeeze(np.concatenate(np.array(u.collect()))).T
        u_true, s_true, v_true = ln.svd(mat)
        tol = 1
        assert_array_almost_equal(v[0], match_sign(v[0], v_true[0, :]), tol)
        assert_array_almost_equal(s[0], s_true[0], tol)
        assert_array_almost_equal(u, match_sign(u, u_true[:, 0]), tol)
