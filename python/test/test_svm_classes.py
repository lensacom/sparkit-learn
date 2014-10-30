import shutil
import tempfile
import numpy as np
import scipy.sparse as sp

from numpy.testing import assert_array_almost_equal

from sklearn.datasets import make_classification
from sklearn.svm import LinearSVC

from common import SplearnTestCase
from splearn.rdd import ArrayRDD, TupleRDD
from splearn.svm import SparkLinearSVC


class SVMClassesTestCase(SplearnTestCase):

    def setUp(self):
        super(SVMClassesTestCase, self).setUp()
        self.outputdir = tempfile.mkdtemp()

    def tearDown(self):
        super(SVMClassesTestCase, self).tearDown()
        shutil.rmtree(self.outputdir)

    def generate_dataset(self, classes, samples, blocks=None):
        X, y = make_classification(n_classes=classes,
                                   n_samples=samples, n_features=5,
                                   n_informative=4, n_redundant=0,
                                   n_clusters_per_class=1,
                                   random_state=42)

        X_rdd = self.sc.parallelize(X)
        y_rdd = self.sc.parallelize(y)

        Z = TupleRDD(X_rdd.zip(y_rdd), blocks)

        return X, y, Z


class TestLinearSVC(SVMClassesTestCase):

    def test_same_coefs(self):
        X, y, Z = self.generate_dataset(2, 100000)

        local = LinearSVC()
        dist = SparkLinearSVC()

        local.fit(X, y)
        dist.fit(Z, classes=np.unique(y))

        assert_array_almost_equal(local.coef_, dist.coef_, decimal=4)
