# encoding: utf-8

import numpy as np
import scipy.sparse as sp
from pyspark.mllib.clustering import KMeans as MLlibKMeans
from sklearn.cluster import KMeans

from ..rdd import ArrayRDD, DictRDD
from ..utils.validation import check_rdd


class SparkKMeans(KMeans):

    """Distributed K-Means clustering

    Parameters
    ----------
    n_clusters : int, optional, default: 8
        The number of clusters to form as well as the number of
        centroids to generate.
    max_iter : int, default: 300
        Maximum number of iterations of the k-means algorithm for a
        single run.
    n_init : int, default: 10
        Number of time the k-means algorithm will be run with different
        centroid seeds. The final results will be the best output of
        n_init consecutive runs in terms of inertia.
    init : {'k-means++', 'random' or an ndarray}
        Method for initialization, defaults to 'k-means++':
        'k-means++' : selects initial cluster centers for k-mean
        clustering in a smart way to speed up convergence. See section
        Notes in k_init for more details.
        'random': choose k observations (rows) at random from data for
        the initial centroids.
        If an ndarray is passed, it should be of shape (n_clusters, n_features)
        and gives the initial centers.
    precompute_distances : {'auto', True, False}
        Precompute distances (faster but takes more memory).
        'auto' : do not precompute distances if n_samples * n_clusters > 12
        million. This corresponds to about 100MB overhead per job using
        double precision.
        True : always precompute distances
        False : never precompute distances
    tol : float, default: 1e-4
        Relative tolerance with regards to inertia to declare convergence
    n_jobs : int, default: 1
        The number of jobs to use for the computation. This works by breaking
        down the pairwise matrix into n_jobs even slices and computing them in
        parallel.
        If -1 all CPUs are used. If 1 is given, no parallel computing code is
        used at all, which is useful for debugging. For n_jobs below -1,
        (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all CPUs but one
        are used.
    random_state : integer or numpy.RandomState, optional
        The generator used to initialize the centers. If an integer is
        given, it fixes the seed. Defaults to the global numpy random
        number generator.

    Attributes
    ----------
    cluster_centers_ : array, [n_clusters, n_features]
        Coordinates of cluster centers
    labels_ :
        Labels of each point
    inertia_ : float
        Sum of distances of samples to their closest cluster center.

    """

    def __init__(self, *args, **kwargs):
        super(SparkKMeans, self).__init__(*args, **kwargs)

    def fit(self, Z):
        """Compute k-means clustering.

        Parameters
        ----------
        Z : ArrayRDD or DictRDD containing array-like or sparse matrix
            Train data.

        Returns
        -------
        self
        """
        X = Z[:, 'X'] if isinstance(Z, DictRDD) else Z
        check_rdd(X, (np.ndarray, sp.spmatrix))
        if self.init == 'k-means||':
            self._mllib_model = MLlibKMeans.train(
                X.unblock(),
                self.n_clusters,
                maxIterations=self.max_iter,
                initializationMode="k-means||")
            self.cluster_centers_ = self._mllib_model.centers
        else:
            models = X.map(lambda X: super(SparkKMeans, self).fit(X))
            models = models.map(lambda model: model.cluster_centers_).collect()
            return super(SparkKMeans, self).fit(np.concatenate(models))

    def predict(self, X):
        """Predict the closest cluster each sample in X belongs to.

        In the vector quantization literature, `cluster_centers_` is called
        the code book and each value returned by `predict` is the index of
        the closest code in the code book.

        Parameters
        ----------
        X : ArrayRDD containing array-like, sparse matrix
            New data to predict.

        Returns
        -------
        labels : ArrayRDD with predictions
            Index of the cluster each sample belongs to.

        """
        check_rdd(X, (np.ndarray, sp.spmatrix))
        if hasattr(self, '_mllib_model'):
            if isinstance(X, ArrayRDD):
                X = X.tolist()
            return self._mllib_model.predict(X)
        else:
            rdd = X.map(lambda X: super(SparkKMeans, self).predict(X))
            return ArrayRDD(rdd)
