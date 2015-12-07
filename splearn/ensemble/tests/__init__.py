import numpy as np
from nose.tools import assert_true
from sklearn.ensemble import RandomForestClassifier
from splearn.ensemble import SparkPseudoRandomForestClassifier
from splearn.utils.testing import SplearnTestCase
from splearn.utils.validation import check_rdd_dtype


class TestSparkRandomForest(SplearnTestCase):

    def test_same_predictions(self):
        X, y, Z = self.make_classification(2, 10000)

        local = RandomForestClassifier()
        dist = SparkPseudoRandomForestClassifier()

        y_local = local.fit(X, y).predict(X)
        y_dist = dist.fit(Z, classes=np.unique(y)).predict(Z[:, 'X'])
        y_conv = dist.to_scikit().predict(X)

        assert_true(check_rdd_dtype(y_dist, (np.ndarray,)))
        assert(sum(y_local != y_dist.toarray()) < len(y_local) * 2./100.)
        assert(sum(y_local != y_conv) < len(y_local) * 2./100.)
