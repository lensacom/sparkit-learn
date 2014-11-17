# encoding: utf-8

import time

from sklearn.utils.validation import _num_samples
from sklearn.cross_validation import _score

import warnings
from itertools import chain, combinations
from math import ceil, floor, factorial
import numbers
import time
from abc import ABCMeta, abstractmethod

import numpy as np
import scipy.sparse as sp

from sklearn.base import is_classifier, clone
from sklearn.utils import indexable, check_random_state, safe_indexing
from sklearn.utils.validation import _num_samples, check_array
from sklearn.utils.multiclass import type_of_target
from sklearn.externals.joblib import Parallel, delayed, logger
from sklearn.externals.six import with_metaclass
from sklearn.externals.six.moves import zip
from sklearn.metrics.scorer import check_scoring

from sklearn.cross_validation import KFold, StratifiedKFold, FitFailedWarning


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
    print scorer
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
                          for k, v in parameters.items()))
        print("[CV] %s %s" % (msg, (64 - len(msg)) * '.'))

    # Adjust lenght of sample weights
    # n_samples = _num_samples(X)
    # fit_params = fit_params if fit_params is not None else {}
    # fit_params = dict([(k, np.asarray(v)[train]
    #                    if hasattr(v, '__len__') and len(v) == n_samples else v)
    #                    for k, v in fit_params.items()])

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
        print("[CV] %s %s" % ((64 - len(end_msg)) * '.', end_msg))

    ret = [train_score] if return_train_score else []
    ret.extend([test_score, _num_samples(Z_test), scoring_time])
    if return_parameters:
        ret.append(parameters)
    return ret
