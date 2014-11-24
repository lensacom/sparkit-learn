# encoding: utf-8

from sklearn.linear_model.stochastic_gradient import SGDClassifier

from splearn.rdd import ArrayRDD, DictRDD
from splearn.linear_model.base import SparkLinearModelMixin


class SparkSGDClassifier(SGDClassifier, SparkLinearModelMixin):

    def fit(self, Z):
        return self._spark_fit(SparkSGDClassifier, Z)

    def predict(self, X):
        return self._spark_predict(SparkSGDClassifier, X)
