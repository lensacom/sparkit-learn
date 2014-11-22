import shutil
import tempfile
import numpy as np
import scipy.sparse as sp

from nose.tools import assert_equal
from nose.tools import assert_true
from numpy.testing import assert_array_almost_equal

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from common import SplearnTestCase
from splearn.rdd import ArrayRDD, TupleRDD
from splearn.linear_model import SparkLogisticRegression


class LinearModelLogisticTestCase(SplearnTestCase):

    def setUp(self):
        super(LinearModelLogisticTestCase, self).setUp()
        self.outputdir = tempfile.mkdtemp()

    def tearDown(self):
        super(LinearModelLogisticTestCase, self).tearDown()
        shutil.rmtree(self.outputdir)

    def generate_dataset(self, n_classes, n_samples, blocks=None):
        X, y = make_classification(n_classes=n_classes,
                                   n_samples=n_samples, n_features=5,
                                   n_informative=4, n_redundant=0,
                                   n_clusters_per_class=1,
                                   random_state=42)
        # X = np.abs(X)

        X_rdd = self.sc.parallelize(X)
        y_rdd = self.sc.parallelize(y)

        Z = TupleRDD(X_rdd.zip(y_rdd), blocks)

        return X, y, Z


class TestLogisticRegression(LinearModelLogisticTestCase):

    def test_same_coefs(self):
        X, y, Z = self.generate_dataset(3, 10000)

        local = LogisticRegression()
        dist = SparkLogisticRegression()

        local.fit(X, y)
        dist.fit(Z, classes=np.unique(y))

        assert_array_almost_equal(local.coef_, dist.coef_)
