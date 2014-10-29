# encoding: utf-8

from sklearn.base import copy
from sklearn.linear_model.base import LinearRegression

from ..rdd import ArrayRDD


class SparkLinearModelMixin(object):

    def __add__(self, other):
        """
        Add method for Linear models with coef and intercept attributes.

        Parameters
        ----------
        other : fitted sklearn linear model
            Model to add.

        Returns
        -------
        model : Linear model
            Model with updated coefficients.
        """
        model = copy.deepcopy(self)
        model.coef_ += other.coef_
        model.intercept_ += other.intercept_
        return model

    def __radd__(self, other):
        return self if other == 0 else self.__add__(other)

    def __div__(self, other):
        self.coef_ /= other
        self.intercept_ /= other
        return self

    def _spark_fit(self, cls, Z, *args, **kwargs):
        mapper = lambda (X, y): super(cls, self).fit(X, y, *args, **kwargs)
        models = Z.map(mapper)
        avg = models.sum() / models.count()
        self.__dict__.update(avg.__dict__)
        return self

    def _spark_predict(self, cls, X, *args, **kwargs):
        return X.map(lambda X: super(cls, self).predict(X, *args, **kwargs))


class SparkLinearRegression(LinearRegression, SparkLinearModelMixin):

    def fit(self, Z):
        return self._spark_fit(SparkLinearRegression, Z)

    def predict(self, X):
        return self._spark_predict(SparkLinearRegression, X)
