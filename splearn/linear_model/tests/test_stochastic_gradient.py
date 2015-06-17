import numpy as np
from sklearn.linear_model import SGDClassifier
from splearn.linear_model import SparkSGDClassifier
from splearn.utils.testing import SplearnTestCase, assert_true
from splearn.utils.validation import check_rdd_dtype


class TestSGDClassifier(SplearnTestCase):

    def test_same_prediction(self):
        X, y, Z = self.make_classification(2, 80000)

        local = SGDClassifier(average=True)
        dist = SparkSGDClassifier(average=True)

        local.fit(X, y)
        dist.fit(Z, classes=np.unique(y))

        y_local = local.predict(X)
        y_dist = dist.predict(Z[:, 'X'])

        mismatch = y_local.shape[0] - np.count_nonzero(y_dist.toarray() == y_local)
        mismatch_percent = float(mismatch) * 100 / y_local.shape[0]

        assert_true(mismatch_percent <= 1)
        assert_true(check_rdd_dtype(y_dist, (np.ndarray,)))
