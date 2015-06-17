# -*- coding: utf-8 -*-


from collections import Sized

import numpy as np
from sklearn.base import clone
from sklearn.externals.joblib import Parallel, delayed
from sklearn.grid_search import GridSearchCV, ParameterGrid, _CVScoreTuple
from sklearn.metrics.scorer import check_scoring
from splearn.base import SparkBaseEstimator
from splearn.cross_validation import _check_cv, _fit_and_score


class SparkGridSearchCV(GridSearchCV, SparkBaseEstimator):

    def _fit(self, Z, parameter_iterable):
        """Actual fitting,  performing the search over parameters."""
        self.scorer_ = check_scoring(self.estimator, scoring=self.scoring)

        cv = self.cv
        cv = _check_cv(cv, Z)

        if self.verbose > 0:
            if isinstance(parameter_iterable, Sized):
                n_candidates = len(parameter_iterable)
                print("Fitting {0} folds for each of {1} candidates, totalling"
                      " {2} fits".format(len(cv), n_candidates,
                                         n_candidates * len(cv)))

        base_estimator = clone(self.estimator)

        pre_dispatch = self.pre_dispatch

        out = Parallel(
            n_jobs=self.n_jobs, verbose=self.verbose,
            pre_dispatch=pre_dispatch, backend="threading"
        )(
            delayed(_fit_and_score)(clone(base_estimator), Z, self.scorer_,
                                    train, test, self.verbose, parameters,
                                    self.fit_params, return_parameters=True,
                                    error_score=self.error_score)
            for parameters in parameter_iterable
            for train, test in cv)

        # Out is a list of triplet: score, estimator, n_test_samples
        n_fits = len(out)
        n_folds = len(cv)

        scores = list()
        grid_scores = list()
        for grid_start in range(0, n_fits, n_folds):
            n_test_samples = 0
            score = 0
            all_scores = []
            for this_score, this_n_test_samples, _, parameters in \
                    out[grid_start:grid_start + n_folds]:
                all_scores.append(this_score)
                if self.iid:
                    this_score *= this_n_test_samples
                    n_test_samples += this_n_test_samples
                score += this_score
            if self.iid:
                score /= float(n_test_samples)
            else:
                score /= float(n_folds)
            scores.append((score, parameters))
            # TODO: shall we also store the test_fold_sizes?
            grid_scores.append(_CVScoreTuple(
                parameters,
                score,
                np.array(all_scores)))
        # Store the computed scores
        self.grid_scores_ = grid_scores

        # Find the best parameters by comparing on the mean validation score:
        # note that `sorted` is deterministic in the way it breaks ties
        best = sorted(grid_scores, key=lambda x: x.mean_validation_score,
                      reverse=True)[0]
        self.best_params_ = best.parameters
        self.best_score_ = best.mean_validation_score

        if self.refit:
            # fit the best estimator using the entire dataset
            # clone first to work around broken estimators
            best_estimator = clone(base_estimator).set_params(
                **best.parameters)
            best_estimator.fit(Z, **self.fit_params)
            self.best_estimator_ = best_estimator
        return self

    def fit(self, Z):
        return self._fit(Z, ParameterGrid(self.param_grid))
