# encoding: utf-8

import numpy as np

from .base import SparkLinearModelMixin

from sklearn.linear_model import SGDClassifier


class SparkSGDClassifier(SGDClassifier, SparkLinearModelMixin):

    def __init__(self, *args, **kwargs):
        super(SparkSGDClassifier, self).__init__(*args, **kwargs)
        self.average = True  # force averaging

    @property
    def classes_(self):
        return self._classes_

    @classes_.setter
    def classes_(self, value):
        pass

    def fit(self, Z, classes=None):
        self._classes_ = np.unique(classes)
        return self._spark_fit(SparkSGDClassifier, Z)

    def predict(self, X):
        return self._spark_predict(SparkSGDClassifier, X)
