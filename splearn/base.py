# -*- coding: utf-8 -*-

from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.metrics import accuracy_score


class SparkBroadcasterMixin(object):

    # TODO: consider caching in case of streaming
    def broadcast(self, func, context):
        bcvars = {name: context.broadcast(getattr(self, name))
                  for name in self.__transient__}

        def func_wrapper(*args, **kwargs):
            for k, v in bcvars.items():
                setattr(func.__self__, k, v.value)
            return func(*args, **kwargs)
        return func_wrapper


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
