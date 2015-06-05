import shutil
import tempfile

import numpy as np
from common import SplearnTestCase
from numpy.testing import assert_array_almost_equal
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from splearn.linear_model import SparkLinearRegression
from splearn.rdd import DictRDD


class LinearModelBaseTestCase(SplearnTestCase):

    def setUp(self):
        super(LinearModelBaseTestCase, self).setUp()
        self.outputdir = tempfile.mkdtemp()

    def tearDown(self):
        super(LinearModelBaseTestCase, self).tearDown()
        shutil.rmtree(self.outputdir)

    def generate_dataset(self, n_targets, n_samples, blocks=None):
        X, y = make_regression(n_targets=n_targets,
                               n_samples=n_samples, n_features=20,
                               n_informative=10, random_state=42)
        # X = np.abs(X)

        X_rdd = self.sc.parallelize(X)
        y_rdd = self.sc.parallelize(y)

        Z = DictRDD(X_rdd.zip(y_rdd), columns=('X', 'y'), bsize=blocks)

        return X, y, Z


class TestLinearRegression(LinearModelBaseTestCase):

    def test_same_coefs(self):
        X, y, Z = self.generate_dataset(1, 100000)

        local = LinearRegression()
        dist = SparkLinearRegression()

        local.fit(X, y)
        dist.fit(Z)

        assert_array_almost_equal(local.coef_, dist.coef_)
        assert_array_almost_equal(local.intercept_, dist.intercept_)

    def test_same_prediction(self):
        X, y, Z = self.generate_dataset(1, 100000)

        local = LinearRegression()
        dist = SparkLinearRegression()

        y_local = local.fit(X, y).predict(X)
        y_dist = dist.fit(Z).predict(Z[:, 'X'])

        assert_array_almost_equal(y_local, np.concatenate(y_dist.collect()))
