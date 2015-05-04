import shutil
import tempfile

import numpy as np
import scipy.sparse as sp
from common import SplearnTestCase
from numpy.testing import assert_array_almost_equal
from sklearn.feature_selection import VarianceThreshold
from splearn.feature_selection import SparkVarianceThreshold
from splearn.rdd import ArrayRDD


class FeatureSelectionTestCase(SplearnTestCase):

    def setUp(self):
        super(FeatureSelectionTestCase, self).setUp()
        self.outputdir = tempfile.mkdtemp()

    def tearDown(self):
        super(FeatureSelectionTestCase, self).tearDown()
        shutil.rmtree(self.outputdir)

    def generate_dataset(self, shape=(1e3, 100), block_size=None):
        rng = np.random.RandomState(2)
        X = rng.randn(*shape)
        X_rdd = ArrayRDD(self.sc.parallelize(X, 4), block_size)
        return X, X_rdd

    def generate_sparse_dataset(self, shape=(1e3, 100), block_size=None):
        data = sp.rand(shape[0], shape[1], random_state=2, density=0.1).toarray()
        X = [sp.csr_matrix([row]) for row in data]
        X_rdd = ArrayRDD(self.sc.parallelize(X, 4), block_size)
        return sp.vstack(X), X_rdd


class TestVarianceThreshold(FeatureSelectionTestCase):

    def test_same_variances(self):
        local = VarianceThreshold()
        dist = SparkVarianceThreshold()

        shapes = [((10, 5), None),
                  ((1e3, 20), None),
                  ((1e3, 20), 100),
                  ((1e4, 100), None),
                  ((1e4, 100), 600)]

        for shape, block_size in shapes:
            X, X_rdd = self.generate_dataset(shape, block_size)
            local.fit(X)
            dist.fit(X_rdd)
            assert_array_almost_equal(local.variances_, dist.variances_)

            X, X_rdd = self.generate_sparse_dataset()
            local.fit(X)
            dist.fit(X_rdd)
            assert_array_almost_equal(local.variances_, dist.variances_)

    def test_same_transform_result(self):
        local = VarianceThreshold()
        dist = SparkVarianceThreshold()

        X, X_rdd = self.generate_dataset()
        result_local = local.fit_transform(X)
        result_dist = np.vstack(dist.fit_transform(X_rdd).collect())
        assert_array_almost_equal(result_local, result_dist)

        X, X_rdd = self.generate_sparse_dataset()
        result_local = local.fit_transform(X)
        result_dist = sp.vstack(dist.fit_transform(X_rdd).collect())
        assert_array_almost_equal(result_local.toarray(),
                                  result_dist.toarray())

    def test_same_transform_with_treshold(self):
        local = VarianceThreshold(.03)
        dist = SparkVarianceThreshold(.03)

        X, X_rdd = self.generate_dataset()
        result_local = local.fit_transform(X)
        result_dist = np.vstack(dist.fit_transform(X_rdd).collect())
        assert_array_almost_equal(result_local, result_dist)

        X, X_rdd = self.generate_sparse_dataset()
        result_local = local.fit_transform(X)
        print result_local.shape
        result_dist = sp.vstack(dist.fit_transform(X_rdd).collect())
        print result_dist.shape
        assert_array_almost_equal(result_local.toarray(),
                                  result_dist.toarray())
