import shutil
import tempfile

import numpy as np
from common import SplearnTestCase
from numpy.testing import assert_array_almost_equal
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from splearn.cluster import SparkKMeans
from splearn.rdd import ArrayRDD


class ClusterKMeansTestCase(SplearnTestCase):

    def setUp(self):
        super(ClusterKMeansTestCase, self).setUp()
        self.outputdir = tempfile.mkdtemp()

    def tearDown(self):
        super(ClusterKMeansTestCase, self).tearDown()
        shutil.rmtree(self.outputdir)

    def generate_dataset(self, centers, n_samples, blocks=None):
        X, y = make_blobs(n_samples=n_samples, centers=centers,
                          random_state=42)
        X_rdd = ArrayRDD(self.sc.parallelize(X))
        return X, y, X_rdd


class TestKMeans(ClusterKMeansTestCase):

    def test_same_centroids(self):
        X, y, X_rdd = self.generate_dataset(centers=4, n_samples=200000)

        local = KMeans(n_clusters=4, init='k-means++', random_state=42)
        dist = SparkKMeans(n_clusters=4, init='k-means++', random_state=42)

        local.fit(X)
        dist.fit(X_rdd)

        local_centers = np.sort(local.cluster_centers_, axis=0)
        dist_centers = np.sort(dist.cluster_centers_, axis=0)

        assert_array_almost_equal(local_centers, dist_centers, decimal=4)

    def test_kmeans_parallel(self):
        X, y, X_rdd = self.generate_dataset(centers=4, n_samples=200000)

        local = KMeans(n_clusters=4, init='k-means++', random_state=42)
        dist = SparkKMeans(n_clusters=4, init='k-means||', random_state=42)

        local.fit(X)
        dist.fit(X_rdd)

        local_centers = np.sort(local.cluster_centers_, axis=0)
        dist_centers = np.sort(dist.cluster_centers_, axis=0)

        assert_array_almost_equal(local_centers, dist_centers, decimal=4)
