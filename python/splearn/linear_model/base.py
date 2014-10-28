# encoding: utf-8

from sklearn.linear_model.base import LinearRegression
from distributed import DistributedTrainMixin

class SparkLinearRegression (LinearRegression, DistributedTrainMixin):

    def _fit(self, X, y, classes):
        return super(SparkLinearRegression, self).fit(X, y)

    def fit(self, Z, n_iter=10, classes=None):
        avg = super(SparkLinearRegression, self).parallel_train(self, Z, classes, n_iter)
        self.__dict__.update(avg.__dict__)
        return self