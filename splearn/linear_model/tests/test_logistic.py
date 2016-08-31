import numpy as np
from sklearn.linear_model import LogisticRegression
from splearn.linear_model import SparkLogisticRegression
from splearn.utils.testing import SplearnTestCase, assert_array_almost_equal


class TestLogisticRegression(SplearnTestCase):

    def test_same_coefs(self):
        X, y, Z = self.make_classification(2, 10000)

        local = LogisticRegression(tol=1e-4, C=10)
        dist = SparkLogisticRegression(tol=1e-4, C=10)

        local.fit(X, y)
        dist.fit(Z, classes=np.unique(y))
        converted = dist.to_scikit()

        assert_array_almost_equal(local.coef_, dist.coef_, decimal=1)
        assert_array_almost_equal(local.coef_, converted.coef_, decimal=1)

    def test_same_prediction(self):
        X, y, Z = self.make_classification(2, 100000)

        local = LogisticRegression(tol=1e-4, C=10)
        dist = SparkLogisticRegression(tol=1e-4, C=10)

        y_local = local.fit(X, y).predict(X)
        y_dist = dist.fit(Z, classes=local.classes_).predict(Z[:, 'X'])
        y_converted = dist.to_scikit().predict(X)

        assert (sum(y_local != y_dist.toarray()) < len(y_local) * 1. / 100.)
        assert (sum(y_local != y_converted) < len(y_local) * 1. / 100.)
