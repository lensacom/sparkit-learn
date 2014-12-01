# encoding: utf-8

import numpy as np

from ..rdd import ArrayRDD, DictRDD

from sklearn.cluster import KMeans
from pyspark.mllib.clustering import KMeans as MLlibKMeans


class SparkKMeans(KMeans):

    def __init__(self, *args, **kwargs):
        super(SparkKMeans, self).__init__(*args, **kwargs)

    def fit(self, Z):
        X = Z[:, 'X'] if isinstance(Z, DictRDD) else Z
        if self.init == 'k-means||':
            self._mllib_model = MLlibKMeans.train(
                X.unblock()._rdd,
                self.n_clusters,
                maxIterations=self.max_iter,
                initializationMode="k-means||")
            self.cluster_centers_ = self._mllib_model.centers
        else:
            models = X.map(lambda X: super(SparkKMeans, self).fit(X))
            models = models.map(lambda model: model.cluster_centers_).collect()
            return super(SparkKMeans, self).fit(np.concatenate(models))

    def predict(self, X):
        if hasattr(self, '_mllib_model'):
            if isinstance(X, ArrayRDD):
                X = X.tolist()
            return self._mllib_model.predict(X)
        else:
            return X.map(lambda X: super(SparkKMeans, self).predict(X))
