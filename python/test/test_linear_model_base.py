import shutil
import tempfile
import numpy as np
import scipy.sparse as sp

from nose.tools import assert_equal
from nose.tools import assert_true
from numpy.testing import assert_array_almost_equal

from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression

from common import SplearnTestCase
from splearn.rdd import ArrayRDD, DictRDD
from splearn.linear_model import SparkLinearRegression


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

        Z = DictRDD(X_rdd.zip(y_rdd), columns=('X', 'y'), block_size=blocks)

        return X, y, Z


class TestLinearRegression(LinearModelBaseTestCase):

    def test_same_prediction(self):
        X, y, Z = self.generate_dataset(1, 100000)

        local = LinearRegression()
        dist = SparkLinearRegression()

        y_local = local.fit(X, y).predict(X)
        y_dist = dist.fit(Z).predict(Z[:, 'X'])

        assert_array_almost_equal(y_local, np.concatenate(y_dist.collect()))
