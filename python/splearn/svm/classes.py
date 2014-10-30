# encoding: utf-8

import numpy as np

from sklearn.svm import LinearSVC

from splearn.linear_model.base import SparkLinearModelMixin


class SparkLinearSVC(LinearSVC, SparkLinearModelMixin):

    @property
    def classes_(self):
        return self._classes_

    @classes_.setter
    def classes_(self, value):
        pass

    def fit(self, Z, classes=None):
        self._classes_ = np.unique(classes)
        return self._spark_fit(SparkLinearSVC, Z)

    def predict(self, X):
        return self._spark_predict(SparkLinearSVC, X)
