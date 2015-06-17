import numpy as np
from sklearn.cluster import KMeans
from splearn.cluster import SparkKMeans
from splearn.utils.testing import SplearnTestCase, assert_array_almost_equal


class TestKMeans(SplearnTestCase):

    def test_same_centroids(self):
        X, y, X_rdd = self.make_blobs(centers=4, n_samples=200000)

        local = KMeans(n_clusters=4, init='k-means++', random_state=42)
        dist = SparkKMeans(n_clusters=4, init='k-means++', random_state=42)

        local.fit(X)
        dist.fit(X_rdd)

        local_centers = np.sort(local.cluster_centers_, axis=0)
        dist_centers = np.sort(dist.cluster_centers_, axis=0)

        assert_array_almost_equal(local_centers, dist_centers, decimal=4)
