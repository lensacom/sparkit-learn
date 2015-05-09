# -*- coding: utf-8 -*-

from sklearn.externals import six
from sklearn.pipeline import Pipeline


class SparkPipeline(Pipeline):

    """Distributed implementation of sklearn's pipeline node.

    Pipeline of transforms with a final estimator.

    Sequentially apply a list of transforms and a final estimator.
    Intermediate steps of the pipeline must be 'transforms', that is, they
    must implement fit and transform methods.
    The final estimator only needs to implement fit.

    The purpose of the pipeline is to assemble several steps that can be
    cross-validated together while setting different parameters.
    For this, it enables setting parameters of the various steps using their
    names and the parameter name separated by a '__', as in the example below.

    Parameters
    ----------
    steps: list
        List of (name, transform) tuples (implementing fit/transform) that are
        chained, in the order in which they are chained, with the last object
        an estimator.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.base import BaseEstimator
    >>> from splearn.rdd import DictRDD
    >>> from splearn.linear_model.base import SparkLinearRegression
    >>> from splearn.pipeline import SparkPipeline

    >>> class Noiser(BaseEstimator):
    >>>     def __init__(self, random_seed=42):
    >>>         np.random.seed(random_seed)
    >>>     def fit(self, Z):
    >>>         return self
    >>>     def transform(self, Z):
    >>>         f = lambda X: X + np.random.rand(*X.shape)
    >>>         if isinstance(Z, DictRDD):
    >>>             return Z.transform(f, column='X')
    >>>         else:
    >>>             return Z.transform(f)

    >>> X = np.arange(100)[:, np.newaxis]
    >>> y = np.arange(100)
    >>> X_rdd = sc.parallelize(X)
    >>> y_rdd = sc.parallelize(y)
    >>> rdd = X_rdd.zip(y_rdd)
    >>> Z = DictRDD(rdd, ('X', 'y'), 25)

    >>> pipe = SparkPipeline([('noise', Noiser()),
    >>>                       ('reg', SparkLinearRegression())])
    >>> pipe.fit(Z)
    SparkPipeline(steps=[
        ('noise', Noiser(random_seed=None)),
        ('reg', SparkLinearRegression(copy_X=True,
                                      fit_intercept=True,
                                      n_jobs=1,
                                      normalize=False)
        )])

    >>> pipe.predict(Z[:, 'X']).collect()
    [array([ 1.51878876, 2.50336579, 3.20260105, 4.41610508, 5.52531787]),
     array([ 5.56329829, 6.54787532, 7.24711057, 8.46061461, 9.5698274 ]),
     array([ 9.60780781, 10.59238484, 11.2916201, 12.50512413, 13.61433693]),
     array([ 13.65231734, 14.63689437, 15.33612963, 16.54963366, 17.65884645])]
    """

    def _pre_transform(self, Z, **fit_params):
        fit_params_steps = dict((step, {}) for step, _ in self.steps)
        for pname, pval in six.iteritems(fit_params):
            step, param = pname.split('__', 1)
            fit_params_steps[step][param] = pval
        Zt = Z
        for name, transform in self.steps[:-1]:
            Zt_tmp = Zt.persist()
            if hasattr(transform, "fit_transform"):
                Zt = transform.fit_transform(Zt_tmp, **fit_params_steps[name])
            else:
                Zt = transform.fit(Zt_tmp, **fit_params_steps[name]) \
                              .transform(Zt)
            Zt_tmp.unpersist()
        return Zt, fit_params_steps[self.steps[-1][0]]

    def fit(self, Z, **fit_params):
        """Fit all the transforms one after the other and transform the
        data, then fit the transformed data using the final estimator.

        Parameters
        ----------
        Z : ArrayRDD, TupleRDD or DictRDD
            Input data in blocked distributed format.

        Returns
        -------
        self : SparkPipeline
        """
        Zt, fit_params = self._pre_transform(Z, **fit_params)
        self.steps[-1][-1].fit(Zt, **fit_params)
        return self

    def fit_transform(self, Z, **fit_params):
        """Fit all the transforms one after the other and transform the
        data, then use fit_transform on transformed data using the final
        estimator."""
        Zt, fit_params = self._pre_transform(Z, **fit_params)
        if hasattr(self.steps[-1][-1], 'fit_transform'):
            return self.steps[-1][-1].fit_transform(Zt, **fit_params)
        else:
            return self.steps[-1][-1].fit(Zt, **fit_params).transform(Zt)

    def score(self, Z):
        """Applies transforms to the data, and the score method of the
        final estimator. Valid only if the final estimator implements
        score."""
        Zt = Z
        for name, transform in self.steps[:-1]:
            Zt = transform.transform(Zt)
        return self.steps[-1][-1].score(Zt)
