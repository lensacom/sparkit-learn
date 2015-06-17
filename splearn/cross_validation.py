# -*- coding: utf-8 -*-

import numbers
import time
import warnings

from sklearn.cross_validation import FitFailedWarning, KFold
from sklearn.externals.joblib import logger
from sklearn.utils.validation import _num_samples


def _check_cv(cv, Z=None):
    # This exists for internal use while indices is being deprecated.
    if cv is None:
        cv = 3
    if isinstance(cv, numbers.Integral):
        n_samples = Z.count()
        cv = KFold(n_samples, cv, indices=True)
    if not getattr(cv, "_indices", True):
        raise ValueError("Sparse data and lists require indices-based cross"
                         " validation generator, got: %r", cv)
    return cv


def _score(estimator, Z_test, scorer):
    """Compute the score of an estimator on a given test set."""
    score = scorer(estimator, Z_test)
    if not isinstance(score, numbers.Number):
        raise ValueError("scoring must return a number, got %s (%s) instead."
                         % (str(score), type(score)))
    return score


def _fit_and_score(estimator, Z, scorer, train, test, verbose,
                   parameters, fit_params, return_train_score=False,
                   return_parameters=False, error_score='raise'):

    if verbose > 1:
        if parameters is None:
            msg = "no parameters to be set"
        else:
            msg = '%s' % (', '.join('%s=%s' % (k, v)
                                    for k, v in list(parameters.items())))
        print(("[CV] %s %s" % (msg, (64 - len(msg)) * '.')))

    fit_params = fit_params if fit_params is not None else {}

    if parameters is not None:
        estimator.set_params(**parameters)

    start_time = time.time()

    Z_train = Z[train]
    Z_test = Z[test]

    try:
        estimator.fit(Z_train, **fit_params)
    except Exception as e:
        if error_score == 'raise':
            raise
        elif isinstance(error_score, numbers.Number):
            test_score = error_score
            if return_train_score:
                train_score = error_score
            warnings.warn("Classifier fit failed. The score on this train-test"
                          " partition for these parameters will be set to %f. "
                          "Details: \n%r" % (error_score, e), FitFailedWarning)
        else:
            raise ValueError("error_score must be the string 'raise' or a"
                             " numeric value. (Hint: if using 'raise', please"
                             " make sure that it has been spelled correctly.)"
                             )
    else:
        test_score = _score(estimator, Z_test, scorer)
        if return_train_score:
            train_score = _score(estimator, Z_train, scorer)

    scoring_time = time.time() - start_time

    if verbose > 2:
        msg += ", score=%f" % test_score
    if verbose > 1:
        end_msg = "%s -%s" % (msg, logger.short_format_time(scoring_time))
        print(("[CV] %s %s" % ((64 - len(end_msg)) * '.', end_msg)))

    ret = [train_score] if return_train_score else []
    ret.extend([test_score, _num_samples(Z_test), scoring_time])
    if return_parameters:
        ret.append(parameters)
    return ret
