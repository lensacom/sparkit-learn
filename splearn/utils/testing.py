import unittest

import numpy as np
import scipy.sparse as sp
from nose.tools import assert_is_instance
from pyspark import SparkContext
from sklearn.datasets import make_blobs, make_classification, make_regression
from sklearn.feature_extraction.tests.test_text import ALL_FOOD_DOCS
from sklearn.utils.testing import (assert_almost_equal,
                                   assert_array_almost_equal,
                                   assert_array_equal, assert_equal,
                                   assert_raises, assert_true)
from splearn.rdd import ArrayRDD, DictRDD, SparseRDD


def assert_tuple_equal(tpl1, tpl2):
    assert_equal(len(tpl1), len(tpl2))
    for i in range(len(tpl1)):
        assert_array_equal(tpl1[i], tpl2[i])


def assert_multiple_tuples_equal(tpls1, tpls2):
    assert_equal(len(tpls1), len(tpls2))
    for i, tpl1 in enumerate(tpls1):
        assert_tuple_equal(tpl1, tpls2[i])


class SplearnTestCase(unittest.TestCase):

    def setUp(self):
        class_name = self.__class__.__name__
        self.sc = SparkContext('local[2]', class_name)
        self.sc._jvm.System.setProperty("spark.ui.showConsoleProgress", "false")
        log4j = self.sc._jvm.org.apache.log4j
        log4j.LogManager.getRootLogger().setLevel(log4j.Level.FATAL)

    def tearDown(self):
        self.sc.stop()
        # To avoid Akka rebinding to the same port, since it doesn't unbind
        # immediately on shutdown
        self.sc._jvm.System.clearProperty("spark.driver.port")

    def make_blobs(self, centers, n_samples, blocks=-1):
        X, y = make_blobs(n_samples=n_samples, centers=centers, random_state=42)
        X_rdd = ArrayRDD(self.sc.parallelize(X))
        return X, y, X_rdd

    def make_regression(self, n_targets, n_samples, blocks=-1):
        X, y = make_regression(n_targets=n_targets,
                               n_samples=n_samples, n_features=20,
                               n_informative=10, random_state=42)

        X_rdd = ArrayRDD(self.sc.parallelize(X))
        y_rdd = ArrayRDD(self.sc.parallelize(y))
        Z = DictRDD([X_rdd, y_rdd], columns=('X', 'y'), bsize=blocks)

        return X, y, Z

    def make_classification(self, n_classes, n_samples, blocks=-1,
                            nonnegative=False):
        X, y = make_classification(n_classes=n_classes,
                                   n_samples=n_samples, n_features=5,
                                   n_informative=4, n_redundant=0,
                                   n_clusters_per_class=1,
                                   random_state=42)
        if nonnegative:
            X = np.abs(X)

        X_rdd = ArrayRDD(self.sc.parallelize(X, 4))
        y_rdd = ArrayRDD(self.sc.parallelize(y, 4))
        Z = DictRDD([X_rdd, y_rdd], columns=('X', 'y'), bsize=blocks)

        return X, y, Z

    def make_text_rdd(self, blocks=-1):
        X = ALL_FOOD_DOCS
        X_rdd = ArrayRDD(self.sc.parallelize(X, 4), blocks)
        return X, X_rdd

    def make_dense_rdd(self, shape=(1e3, 10), block_size=-1):
        rng = np.random.RandomState(2)
        X = rng.randn(*shape)
        X_rdd = ArrayRDD(self.sc.parallelize(X, 4), bsize=block_size)
        return X, X_rdd

    def make_dense_range_rdd(self, shape=(1e3, 10), block_size=-1):
        X = np.arange(np.prod(shape)).reshape(shape)
        X_rdd = ArrayRDD(self.sc.parallelize(X, 4), bsize=block_size)
        return X, X_rdd

    def make_sparse_rdd(self, shape=(1e3, 10), block_size=-1):
        X = sp.rand(shape[0], shape[1], random_state=42, density=0.3)
        X_rows = [sp.csr_matrix([row]) for row in X.toarray()]
        X_rdd = SparseRDD(self.sc.parallelize(X_rows, 4), bsize=block_size)
        return X, X_rdd
