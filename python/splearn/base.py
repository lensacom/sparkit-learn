# -*- coding: utf-8 -*-

from pyspark.broadcast import Broadcast
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.metrics import accuracy_score


class SparkBroadcasterMixin(object):

    # def __getstate__(self):
    #     return {k: v.value if isinstance(v, Broadcast) else v
    #             for k, v in self.__dict__.iteritems()}

    # def _broadcast(self, sc, key, value):
    #     key = "_broadcasted_{0}".format(key)
    #     if hasattr(self, key):
    #         return getattr(self, key)

    #     setattr(self, key, sc.broadcast(value() if callable(value) else value))
    #     return getattr(self, key)

    # def _unbroadcast(self, key):
    #     key = "_broadcasted_{0}".format(key)
    #     if hasattr(self, key):
    #         delattr(self, key)

    def _broadcast(self, sc, key, value=None):
        if value is None:
            value = getattr(self, key)
        if not isinstance(value, Broadcast):
            setattr(self, key, sc.broadcast(value))
            # setattr(self, key + "_", value)


class SparkBaseEstimator(BaseEstimator):
    pass


class SparkClassifierMixin(ClassifierMixin):

    """Mixin class for all classifiers in sparkit-learn."""

    def score(self, Z):
        X, y, w = Z[:, 'X'], Z[:, 'y'], None
        if 'w' in Z.columns:
            w = Z[:, 'w']
        return accuracy_score(y.toarray(),
                              self.predict(X).toarray(),
                              sample_weight=w)


class SparkTransformerMixin(TransformerMixin):

    """Mixin class for all transformers in sparkit-learn."""

    def fit_transform(self, Z, **fit_params):
        """Fit to data, then transform it.

        Fits transformer to Z with optional parameters fit_params
        and returns a transformed version of Z.

        Parameters
        ----------
        Z : ArrayRDD or DictRDD
            Training set.

        Returns
        -------
        Z_new : ArrayRDD or DictRDD
            Transformed array.

        """
        # non-optimized default implementation; override when a better
        # method is possible
        return self.fit(Z, **fit_params).transform(Z)
