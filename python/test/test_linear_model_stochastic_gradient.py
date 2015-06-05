import shutil
import tempfile
import numpy as np
import scipy.sparse as sp

from nose.tools import assert_equal
from nose.tools import assert_true
from numpy.testing import assert_array_almost_equal

from sklearn.datasets import make_classification
from sklearn.linear_model import SGDClassifier

from common import SplearnTestCase
from splearn.rdd import DictRDD
from splearn.linear_model import SparkSGDClassifier


class LinearModelStochasticGradientTestCase(SplearnTestCase):

    def setUp(self):
        super(LinearModelStochasticGradientTestCase, self).setUp()
        self.outputdir = tempfile.mkdtemp()

    def tearDown(self):
        super(LinearModelStochasticGradientTestCase, self).tearDown()
        shutil.rmtree(self.outputdir)

    def generate_dataset(self, n_classes, n_samples, blocks=None):
        X, y = make_classification(n_classes=n_classes,
                                   n_samples=n_samples, n_features=10,
                                   n_informative=4, n_redundant=0,
                                   n_clusters_per_class=1,
                                   random_state=42)

        X_rdd = self.sc.parallelize(X, 4)
        y_rdd = self.sc.parallelize(y, 4)
        Z_rdd = X_rdd.zip(y_rdd)

        Z = DictRDD(Z_rdd, columns=('X', 'y'), block_size=blocks)

        return X, y, Z


class TestSGDClassifier(LinearModelStochasticGradientTestCase):

    def test_same_prediction(self):
        X, y, Z = self.generate_dataset(2, 80000)

        local = SGDClassifier(average=True)
        dist = SparkSGDClassifier(average=True)

        local.fit(X, y)
        dist.fit(Z, classes=np.unique(y))

        y_local = local.predict(X)
        y_dist = np.concatenate(dist.predict(Z[:, 'X']).collect())

        mismatch = y_local.shape[0] - np.count_nonzero(y_dist == y_local)
        mismatch_percent = float(mismatch) * 100 / y_local.shape[0]

        assert_true(mismatch_percent <= 1)
