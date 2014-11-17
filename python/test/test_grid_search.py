import shutil
import tempfile
import numpy as np
import scipy.sparse as sp

from numpy.testing import assert_array_almost_equal

from sklearn.datasets import make_classification
from sklearn.svm import LinearSVC

from common import SplearnTestCase
from splearn.rdd import ArrayRDD, DictRDD
from splearn.grid_search  import SparkGridSearchCV
from splearn.naive_bayes import SparkMultinomialNB

from sklearn.cross_validation import KFold


class GridSearchTestCase(SplearnTestCase):

    def setUp(self):
        super(GridSearchTestCase, self).setUp()
        self.outputdir = tempfile.mkdtemp()

    def tearDown(self):
        super(GridSearchTestCase, self).tearDown()
        shutil.rmtree(self.outputdir)

    def generate_dataset(self, classes, samples, blocks=None):
        X, y = make_classification(n_classes=classes,
                                   n_samples=samples, n_features=4,
                                   random_state=42)
        X = np.abs(X)

        X_rdd = self.sc.parallelize(X)
        y_rdd = self.sc.parallelize(y)
        Z = DictRDD(X_rdd.zip(y_rdd), columns=('X', 'y'), block_size=blocks)

        return X, y, Z


class TestGridSearchCV(GridSearchTestCase):

    def test_basic(self):
        X, y, Z = self.generate_dataset(2, 1000, 100)

        estimator = SparkMultinomialNB()
        parameters = {'alpha': [0.1, 1, 10]}
        fit_params = {'classes': np.unique(y)}
        grid = SparkGridSearchCV(estimator=estimator,
                                 param_grid=parameters,
                                 fit_params=fit_params)

        result = grid.fit(Z)
        print grid.grid_scores_
