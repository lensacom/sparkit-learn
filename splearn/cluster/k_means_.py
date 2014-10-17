# encoding: utf-8

import numpy as np
from sklearn.cluster import KMeans

class SparkKMeans(KMeans):

    # TODO: docs, decide which fit method is faster
    def fit(self, X, y=None):
        models = X.map(
            lambda X: super(SparkKMeans, self).fit(X))
        models = models.map(lambda x: x.cluster_centers_).collect()
        C = np.concatenate(tuple(model for model in models))
        return super(SparkKMeans, self).fit(C)

    def fit2(self, X, y=None):
        models = X.map(
            lambda X: super(SparkKMeans, self).fit(X))
        model_centers = models.map(lambda x: x.cluster_centers_).collect()
        avg = np.mean(model_centers, axis=0)
        avgmodel = models.collect()[0]
        avgmodel.cluster_centers_ = avg
        self.__dict__.update(avgmodel.__dict__)
        return self

