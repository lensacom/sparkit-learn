import shutil
import tempfile
import numpy as np
import scipy.sparse as sp

from numpy.testing import assert_array_almost_equal

from sklearn.datasets import make_classification
from sklearn.svm import LinearSVC

from common import SplearnTestCase
from splearn.rdd import ArrayRDD, DictRDD
from splearn.grid import SparkGridSearchCV
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

        # cv = KFold(Z.count(), n_folds=3)

        # print Z.ix[0]
        # print Z.ix[5]
        # print Z.count()
        # print(Z.shape)
        cv = KFold(Z.count(), n_folds=3)
        print list(cv)
        # print Z.map(lambda (X, y): X.shape[0]).collect()

        # for train_index, test_index in cv:
        #     print("TRAIN:", train_index, "TEST:", test_index)
        #     print Z.ix[train_index].count()
            # X_train, X_test = X[train_index], X[test_index]
            # y_train, y_test = y[train_index], y[test_index]

        estimator = SparkMultinomialNB()
        parameters = {'alpha': [0.1, 1, 10]}
        fit_params = {'classes': np.unique(y)}
        grid = SparkGridSearchCV(estimator=estimator,
                                 param_grid=parameters,
                                 fit_params=fit_params)

        result = grid.fit(Z)

        print grid.grid_scores_

        print result

