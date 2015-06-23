import numpy as np
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from splearn.naive_bayes import SparkGaussianNB, SparkMultinomialNB
from splearn.utils.testing import (SplearnTestCase, assert_array_almost_equal,
                                   assert_true)
from splearn.utils.validation import check_rdd_dtype


class TestGaussianNB(SplearnTestCase):

    def test_same_prediction(self):
        X, y, Z = self.make_classification(2, 800000, nonnegative=True)

        local = GaussianNB()
        dist = SparkGaussianNB()

        local_model = local.fit(X, y)
        dist_model = dist.fit(Z, classes=np.unique(y))

        # TODO: investigate the variance further!
        assert_array_almost_equal(local_model.sigma_, dist_model.sigma_, 2)
        assert_array_almost_equal(local_model.theta_, dist_model.theta_, 6)


class TestMultinomialNB(SplearnTestCase):

    def test_same_prediction(self):
        X, y, Z = self.make_classification(4, 100000, nonnegative=True)

        local = MultinomialNB()
        dist = SparkMultinomialNB()

        y_local = local.fit(X, y).predict(X)
        y_dist = dist.fit(Z, classes=np.unique(y)).predict(Z[:, 'X'])

        assert_true(check_rdd_dtype(y_dist, (np.ndarray,)))
        assert_array_almost_equal(y_local, y_dist.toarray())
