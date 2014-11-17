from sklearn.base import ClassifierMixin

from .metrics import accuracy_score


class SparkClassifierMixin(ClassifierMixin):
    """Mixin class for all classifiers in scikit-learn."""

    def score(self, Z, sample_weight=None):
        # from .metrics import accuracy_score
        X, y = Z[:, 'X'], Z[:, 'y']
        return accuracy_score(y.toarray(), self.predict(X).toarray(), sample_weight=sample_weight)
