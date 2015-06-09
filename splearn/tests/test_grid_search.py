import numpy as np
from sklearn.grid_search import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.utils.testing import assert_array_almost_equal
from splearn.grid_search import SparkGridSearchCV
from splearn.naive_bayes import SparkMultinomialNB
from splearn.utils.testing import SplearnTestCase


class TestGridSearchCV(SplearnTestCase):

    def test_same_result(self):
        X, y, Z = self.make_classification(2, 40000, nonnegative=True)

        parameters = {'alpha': [0.1, 1, 10]}
        fit_params = {'classes': np.unique(y)}

        local_estimator = MultinomialNB()
        local_grid = GridSearchCV(estimator=local_estimator,
                                  param_grid=parameters)

        estimator = SparkMultinomialNB()
        grid = SparkGridSearchCV(estimator=estimator,
                                 param_grid=parameters,
                                 fit_params=fit_params)

        local_grid.fit(X, y)
        grid.fit(Z)

        locscores = [r.mean_validation_score for r in local_grid.grid_scores_]
        scores = [r.mean_validation_score for r in grid.grid_scores_]

        assert_array_almost_equal(locscores, scores, decimal=2)
