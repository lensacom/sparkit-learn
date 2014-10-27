# encoding: utf-8

import numpy as np
from sklearn.cluster import KMeans

class SparkKMeans(KMeans):

    def fit(self, X, mode='clus', y=None):
        if mode == 'avg':
            return self._fit_avg(X, y)
        return self._fit_clus(X, y)

    def _fit_clus(self, X, y=None):
        models = X.map(
            lambda X: super(SparkKMeans, self).fit(X))
        models = models.map(lambda x: x.cluster_centers_).collect()
        C = np.concatenate(tuple(model for model in models))
        return super(SparkKMeans, self).fit(C)

    def _fit_avg(self, X, y=None):
        models = X.map(
            lambda X: super(SparkKMeans, self).fit(X)).collect()
        avg = np.mean([model.cluster_centers_ for model in models], axis=0)
        avgmodel = models[0]
        avgmodel.cluster_centers_ = avg
        self.__dict__.update(avgmodel.__dict__)
        return self

