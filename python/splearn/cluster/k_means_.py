# encoding: utf-8

import numpy as np
from sklearn.cluster import KMeans

class SparkKMeans(KMeans):

    def fit(self, X, y=None):
        models = X.map(
            lambda X: super(SparkKMeans, self).fit(X))
        models = models.map(lambda x: x.cluster_centers_).collect()
        C = np.concatenate(tuple(model for model in models))
        return super(SparkKMeans, self).fit(C)
