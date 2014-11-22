# encoding: utf-8

import numpy as np

from .base import SparkLinearModelMixin

from sklearn.linear_model import SGDClassifier


class SparkSGDClassifier(SGDClassifier, SparkLinearModelMixin):

    def fit(self, Z, classes=None):
        models = Z.map(lambda (X, y): self.partial_fit(X, y, classes))
        avg = models.sum() / models.count()
        self.__dict__.update(avg.__dict__)
        return self
