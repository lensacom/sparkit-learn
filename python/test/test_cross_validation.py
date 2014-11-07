import shutil
import tempfile
import numpy as np
import scipy.sparse as sp

from numpy.testing import assert_array_almost_equal

from sklearn.datasets import make_classification
from sklearn.cross_validation import KFold

from common import SplearnTestCase
from splearn.rdd import ArrayRDD, TupleRDD
from splearn.cross_validation import SparkKFold


class CrossValidationTestCase(SplearnTestCase):

    def setUp(self):
        super(CrossValidationTestCase, self).setUp()
        self.outputdir = tempfile.mkdtemp()

    def tearDown(self):
        super(CrossValidationTestCase, self).tearDown()
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


class TestKFold(CrossValidationTestCase):

    def test_basic(self):

        # >>> from sklearn import cross_validation
        # >>> X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
        # >>> y = np.array([1, 2, 3, 4])
        # >>> kf = cross_validation.KFold(4, n_folds=2)
        # >>> len(kf)
        # 2
        # >>> print(kf)  # doctest: +NORMALIZE_WHITESPACE
        # sklearn.cross_validation.KFold(n=4, n_folds=2, shuffle=False,
        #                                random_state=None)
        # >>> for train_index, test_index in kf:
        # ...    print("TRAIN:", train_index, "TEST:", test_index)
        # ...    X_train, X_test = X[train_index], X[test_index]
        # ...    y_train, y_test = y[train_index], y[test_index]
        # TRAIN: [2 3] TEST: [0 1]
        # TRAIN: [0 1] TEST: [2 3]

        X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
        y = np.array([1, 2, 3, 4])


        kf = KFold(4, n_folds=2)
        len(kf)
        2
        print(kf)  # doctest: +NORMALIZE_WHITESPACE
        sklearn.cross_validation.KFold(n=4, n_folds=2, shuffle=False,
                                       random_state=None)
        for train_index, test_index in kf:
            print("TRAIN:", train_index, "TEST:", test_index)


