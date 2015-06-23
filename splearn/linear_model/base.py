# encoding: utf-8

import numpy as np
import scipy.sparse as sp
from sklearn.base import copy
from sklearn.linear_model.base import LinearRegression

from ..utils.validation import check_rdd


class SparkLinearModelMixin(object):

    def __add__(self, other):
        """Add method for Linear models with coef and intercept attributes.

        Parameters
        ----------
        other : fitted sklearn linear model
            Model to add.

        Returns
        -------
        model : Linear model
            Model with updated coefficients.
        """
        model = copy.deepcopy(self)
        model.coef_ += other.coef_
        model.intercept_ += other.intercept_
        return model

    def __radd__(self, other):
        """Reverse add method for Linear models.

        Parameters
        ----------
        other : fitted sklearn linear model
            Model to add.

        Returns
        -------
        model : Linear model
            Model with updated coefficients.
        """
        return self if other == 0 else self.__add__(other)

    def __div__(self, other):
        """Division method for Linear models. Used for averaging.

        Parameters
        ----------
        other : integer
            Integer to divide with.

        Returns
        -------
        model : Linear model
            Model with updated coefficients.
        """
        self.coef_ /= other
        self.intercept_ /= other
        return self

    __truediv__ = __div__

    def _spark_fit(self, cls, Z, *args, **kwargs):
        """Wraps a Scikit-learn Linear model's fit method to use with RDD
        input.

        Parameters
        ----------
        cls : class object
            The sklearn linear model's class to wrap.
        Z : TupleRDD or DictRDD
            The distributed train data in a DictRDD.

        Returns
        -------
        self: the wrapped class
        """
        mapper = lambda X_y: super(cls, self).fit(
            X_y[0], X_y[1], *args, **kwargs
        )
        models = Z.map(mapper)
        avg = models.sum() / models.count()
        self.__dict__.update(avg.__dict__)
        return self

    def _spark_predict(self, cls, X, *args, **kwargs):
        """Wraps a Scikit-learn Linear model's predict method to use with RDD
        input.

        Parameters
        ----------
        cls : class object
            The sklearn linear model's class to wrap.
        Z : ArrayRDD
            The distributed data to predict in a DictRDD.

        Returns
        -------
        self: the wrapped class
        """
        return X.map(lambda X: super(cls, self).predict(X, *args, **kwargs))


class SparkLinearRegression(LinearRegression, SparkLinearModelMixin):

    """Distributed implementation of sklearn's Linear Regression.

    Parameters
    ----------
    fit_intercept : boolean, optional
        whether to calculate the intercept for this model. If set
        to false, no intercept will be used in calculations
        (e.g. data is expected to be already centered).
    normalize : boolean, optional, default False
        If True, the regressors X will be normalized before regression.
    copy_X : boolean, optional, default True
        If True, X will be copied; else, it may be overwritten.
    n_jobs : The number of jobs to use for the computation.
        If -1 all CPUs are used. This will only provide speedup for
        n_targets > 1 and sufficient large problems.

    Attributes
    ----------
    coef_ : array, shape (n_features, ) or (n_targets, n_features)
        Estimated coefficients for the linear regression problem.
        If multiple targets are passed during the fit (y 2D), this
        is a 2D array of shape (n_targets, n_features), while if only
        one target is passed, this is a 1D array of length n_features.
    intercept_ : array
        Independent term in the linear model.

    """

    def fit(self, Z):
        """
        Fit linear model.

        Parameters
        ----------
        Z : DictRDD with (X, y) values
            X containing numpy array or sparse matrix - The training data
            y containing the target values

        Returns
        -------
        self : returns an instance of self.
        """
        check_rdd(Z, {'X': (sp.spmatrix, np.ndarray)})
        return self._spark_fit(SparkLinearRegression, Z)

    def predict(self, X):
        """Distributed method to predict class labels for samples in X.

        Parameters
        ----------
        X : ArrayRDD containing {array-like, sparse matrix}
            Samples.

        Returns
        -------
        C : ArrayRDD
            Predicted class label per sample.
        """
        check_rdd(X, (sp.spmatrix, np.ndarray))
        return self._spark_predict(SparkLinearRegression, X)
