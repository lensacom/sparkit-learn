# encoding: utf-8

from itertools import product

import numpy as np

from pyspark.rdd import RDD

from sklearn.grid_search import ParameterGrid, GridSearchCV
from sklearn.cross_validation import KFold

from splearn.rdd import ArrayRDD
from splearn.cross_validation import _fit_and_score

__all__ = ['SparkGridSearchCV']


class SparkParameterGrid(ParameterGrid):

    def __init__(self, *args, **kwargs):
        # assume: ArrayRDD(sc.parallelize(self.param_grid), spark_blocks)
        if not isinstance(kwargs['param_grid'], ArrayRDD):
            raise TypeError(
                'Expected {0}, got {1}.'.format(ArrayRDD, type(kwargs['param_grid']))
            )
        self.param_grid = kwargs['param_grid']

    def flatMap(self, func):
        def param_gen(params):
            param_grid = []
            for p in params:
                # Always sort the keys of a dictionary, for reproducibility
                items = sorted(p.items())
                if items:
                    keys, values = zip(*items)
                    param_grid += [dict(zip(keys, v)) for v in product(*values)]
            return param_grid
        return self.param_grid.flatMap(lambda x: param_gen(x)).flatMap(func)

    def __len__(self):
        return self.param_grid.count()


class SparkGridSearchCV(GridSearchCV):

    #def __add__(self, other):
    #    model = copy.deepcopy(self)
    #    model.grid_scores_ = model.grid_scores_.append(other.grid_scores_) if type(model.grid_scores_) == list else [model.grid_scores_, other.grid_scores_]
    #    model.best_params_ = model.best_params_.append(other.best_params_) if type(model.best_params_) == list else [model.best_params_, other.best_params_]
    #    model.best_score_ = model.best_score_.append(other.best_score_) if type(model.best_score_) == list else [model.best_score_, other.best_score_]
    #    model.best_estimator_ = model.best_estimator_.append(other.best_estimator_) if type(model.best_estimator_) == list else [model.best_estimator_, other.best_estimator_]
    #    return model

    #def __radd__(self, other):
    #    return self if other == 0 else self.__add__(other)


    def fit(self, Z):
        self.param_grid = ParameterGrid(self.param_grid)
        def cv_split(Z):
            kfold = KFold(Z.count())
            indexed = Z.zipWithIndex()
            return [(indexed.filter(lambda (Z, index): index in train).map(lambda (Z, index): Z),
                    indexed.filter(lambda (Z, index): index in test).map(lambda (Z, index): Z))
                    for train, test in list(kfold)]
        cvs = cv_split(Z)
        test_results = []
        for param in self.param_grid:
            for train, test in cvs:
                base_estimator = clone(self.estimator)
                base_estimator.set_params(**parameters)
                base_estimator.fit(train)
                error = base_estimator.predict(test.comlumn(0)) - test.coxlumn


    def fit_oldest(self, Z, blocks=2):
        def param_fit(Z, params):
            return Z.map(
                lambda (X,y): super(SparkGridSearchCV, self)._fit(X, y, ParameterGrid(params)))

        self.param_grid = ArrayRDD(self.param_grid, blocks)
        return self.param_grid.map(
            lambda params: param_fit(Z, params)
        ).sum()

    def fit_old(self, Z):
        def cv_split(Z):
            kfold = KFold(Z.count())
            indexed = Z.zipWithIndex()
            return [(indexed.filter(lambda (Z, index): index in train).map(lambda (Z, index): Z),
                    indexed.filter(lambda (Z, index): index in test).map(lambda (Z, index): Z))
                    for train, test in list(kfold)]
        cvs = cv_split(Z)
        self.fit_params['classes'] = len(np.unique(Z.column(1)))
        def train_model(train, parameters):
            base_estimator = clone(self.estimator)
            base_estimator.set_params(**parameters)
            base_estimator.fit(train)
            return base_estimator

        models = self.param_grid.flatMap(lambda params:
            [train_model(train, params) for train, test in cvs]).collect()
        return models

    def cross_fit(self, train, test, parameters):
        base_estimator = clone(self.estimator)

        out = _fit_and_score(clone(base_estimator), train, test, self.scorer_,
                             self.verbose, parameters,
                             self.fit_params, return_parameters=True,
                             error_score=self.error_score)
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


def _check_param_grid(param_grid):
    if not isinstance(param_grid, SparkParameterGrid):
        raise TypeError(
            'Assumed SparkParameterGrid, got {0}.'.format(type(param_grid))
        )

if __name__ == '__main__':
    from pyspark.context import SparkContext
    import splearn.naive_bayes as sparknb
    from splearn.rdd import ArrayRDD, TupleRDD

    sc = SparkContext()

    X = np.array([[-1, -1], [-2, -1], [-3, -2], [-1, -2], [1, 1], [2, 1], [3, 2], [1, 2]])
    y = np.array([1, 1, 1, 1, 2, 2, 2, 2])
    X_rdd = sc.parallelize(X)
    y_rdd = sc.parallelize(y)
    Z = TupleRDD(X_rdd.zip(y_rdd), 2)
    params = [{'classes': [len(np.unique(y))]}]
    dist_params = ArrayRDD(sc.parallelize(params))
    param_grid = SparkParameterGrid(param_grid=dist_params)

    grid = SparkGridSearchCV(
        estimator=sparknb.SparkGaussianNB(),
        param_grid=param_grid,
        verbose=0,
    )

    print grid.fit(Z)
