# -*- coding: utf-8 -*-

from sklearn.base import ClassifierMixin
from sklearn.metrics import accuracy_score


class SparkClassifierMixin(ClassifierMixin):
    """Mixin class for all classifiers in sparkit-learn."""

    def score(self, Z):
        X, y, w = Z[:, 'X'], Z[:, 'y'], None
        if 'w' in Z.columns:
            w = Z[:, 'w']
        return accuracy_score(y.toarray(),
                              self.predict(X).toarray(),
                              sample_weight=w)
