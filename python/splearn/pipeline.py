# -*- coding: utf-8 -*-

from sklearn.pipeline import Pipeline
from sklearn.externals import six


class SparkPipeline(Pipeline):

    def _pre_transform(self, Z, **fit_params):
        fit_params_steps = dict((step, {}) for step, _ in self.steps)
        for pname, pval in six.iteritems(fit_params):
            step, param = pname.split('__', 1)
            fit_params_steps[step][param] = pval
        Zt = Z
        for name, transform in self.steps[:-1]:
            if hasattr(transform, "fit_transform"):
                Zt = transform.fit_transform(Zt, **fit_params_steps[name])
            else:
                Zt = transform.fit(Zt, **fit_params_steps[name]) \
                              .transform(Zt)
        return Zt, fit_params_steps[self.steps[-1][0]]

    def fit(self, Z, **fit_params):
        """Fit all the transforms one after the other and transform the
        data, then fit the transformed data using the final estimator.
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
