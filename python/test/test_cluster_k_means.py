import shutil
import tempfile
import numpy as np
import scipy.sparse as sp

from nose.tools import assert_equal
from nose.tools import assert_true
from numpy.testing import assert_array_almost_equal

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

from common import SplearnTestCase
from splearn.rdd import ArrayRDD, TupleRDD
from splearn.cluster import SparkKMeans


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

        local = KMeans(n_clusters=4, init='k-means++')
        dist = SparkKMeans(n_clusters=4, init='k-means++')

        local.fit(X)
        dist.fit(X_rdd)

        local_centers = np.sort(local.cluster_centers_, axis=0)
        dist_centers = np.sort(dist.cluster_centers_, axis=0)

        assert_array_almost_equal(local_centers, dist_centers, decimal=4)

    def test_kmeans_parallel(self):
        X, y, X_rdd = self.generate_dataset(centers=4, n_samples=200000)

        local = KMeans(n_clusters=4, init='k-means++')
        dist = SparkKMeans(n_clusters=4, init='k-means||')

        local.fit(X)
        dist.fit(X_rdd)

        local_centers = np.sort(local.cluster_centers_, axis=0)
        dist_centers = np.sort(dist.cluster_centers_, axis=0)

        assert_array_almost_equal(local_centers, dist_centers, decimal=4)
