# -*- coding: utf-8 -*-

from sklearn.base import BaseEstimator as SklearnBaseEstimator
from sklearn.base import ClassifierMixin as SklearnClassifierMixin
from sklearn.base import TransformerMixin as SklearnTransformerMixin
from sklearn.metrics import accuracy_score

from .rdd import BlockRDD, DictRDD


class BroadcasterMixin(object):

    # TODO: consider caching in case of streaming
    def broadcast(self, func, context):
        bcvars = {name: context.broadcast(getattr(self, name))
                  for name in self.__transient__}

        def func_wrapper(*args, **kwargs):
            for k, v in bcvars.items():
                setattr(func.__self__, k, v.value)
            return func(*args, **kwargs)
        return func_wrapper


class BaseEstimator(SklearnBaseEstimator):
    pass


class ClassifierMixin(SklearnClassifierMixin):

    """Mixin class for all classifiers in sparkit-learn."""

    def score(self, Z):
        X, y, w = Z[:, 'X'], Z[:, 'y'], None
        if 'w' in Z.columns:
            w = Z[:, 'w']
        return accuracy_score(y.toarray(),
                              self.predict(X).toarray(),
                              sample_weight=w)


class TransformerMixin(SklearnTransformerMixin):

    """Mixin class for all transformers in sparkit-learn."""

    def fit(self, X, y=None, **fit_params):
        if isinstance(X, BlockRDD):
            return self.spark_fit(X, **fit_params)
        else:
            return super(TransformerMixin, self).fit(X, y, **fit_params)

    def transform(self, X):
        if isinstance(X, BlockRDD):
            return self.spark_transform(X)
        else:
            return super(TransformerMixin, self).transform(X)

    def fit_transform(self, X, y=None, **fit_params):
        if isinstance(X, BlockRDD):
            return self.spark_fit_transform(X, **fit_params)
        else:
            return super(TransformerMixin, self).fit_transform(
                X, y, **fit_params)

    def spark_fit_transform(self, Z, **fit_params):
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
        return self.spark_fit(Z, **fit_params).spark_transform(Z)
