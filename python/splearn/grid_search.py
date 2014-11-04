# encoding: utf-8

from sklearn.grid_search import ParameterGrid, GridSearchCV

from splearn.rdd import ArrayRDD


__all__ = ['SparkGridSearchCV']

class __SparkParameterGrid(ParameterGrid):

    def __init__(self, spark_blocks=4, *args, **kwargs):
        super(SparkParameterGrid, self).__init__(*args, **kwargs)
        self.param_grid = ArrayRDD(sc.parallelize(self.param_grid), spark_blocks)

    def __iter__(self):
        def param_gen(params):
            param_grid = []
            for p in params:
                # Always sort the keys of a dictionary, for reproducibility
                items = sorted(p.items())
                if items:
                    keys, values = zip(*items)
                    param_grid += [dict(zip(keys, v)) for v in product(*values)]
            return param_grid
        return self.param_grid.flatMap(lambda x: param_gen(x)).toiter()

    def __len__(self):
        return self.param_grid.count()


class SparkGridSearchCV(GridSearchCV):

    def __add__(self, other):
        model = copy.deepcopy(self)
        model.grid_scores_ = model.grid_scores_.append(other.grid_scores_) if type(model.grid_scores_) == list else [model.grid_scores_, other.grid_scores_]
        model.best_params_ = model.best_params_.append(other.best_params_) if type(model.best_params_) == list else [model.best_params_, other.best_params_]
        model.best_score_ = model.best_score_.append(other.best_score_) if type(model.best_score_) == list else [model.best_score_, other.best_score_]
        model.best_estimator_ = model.best_estimator_.append(other.best_estimator_) if type(model.best_estimator_) == list else [model.best_estimator_, other.best_estimator_]
        return model

    def __radd__(self, other):
        return self if other == 0 else self.__add__(other)

    def fit(self, sc, Z, blocks=2):
        def param_fit(Z, params):
            return Z.map(
                lambda (X,y): super(SparkGridSearchCV, self)._fit(X, y, ParameterGrid(params)))

        dist_grid = ArrayRDD(sc.parallelize(self.param_grid), blocks)
        return dist_grid.map(
            lambda params: param_fit(Z, params)
        ).sum()


if __name__ == '__main__':
    import numpy as np
    import splearn.naive_bayes as sparknb
    X = np.array([[-1, -1], [-2, -1], [-3, -2], [-1, -2], [1, 1], [2, 1], [3, 2], [1, 2]])
    y = np.array([1, 1, 1, 1, 2, 2, 2, 2])
    X_rdd = sc.parallelize(X)
    y_rdd = sc.parallelize(y)
    Z = TupleRDD(X_rdd.zip(y_rdd), 4)
    grid = SparkGridSearchCV(estimator=sparknb.SparkGaussianNB(), param_grid={}, verbose=0)
    result = grid.fit(sc, Z, 2)
