import shutil
import tempfile
import numpy as np
import scipy.sparse as sp

from nose.tools import assert_equal
from nose.tools import assert_true
from numpy.testing import assert_array_almost_equal
from numpy.testing import assert_array_equal

from sklearn.datasets import make_classification, fetch_20newsgroups
from sklearn.feature_extraction.tests.test_text import ALL_FOOD_DOCS
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from common import SplearnTestCase
from splearn.rdd import ArrayRDD, DictRDD
from splearn.feature_extraction.text import SparkHashingVectorizer
from splearn.feature_extraction.text import SparkCountVectorizer
from splearn.feature_extraction.text import SparkTfidfTransformer


class FeatureExtractionTextTestCase(SplearnTestCase):

    def setUp(self):
        super(FeatureExtractionTextTestCase, self).setUp()
        self.outputdir = tempfile.mkdtemp()

    def tearDown(self):
        super(FeatureExtractionTextTestCase, self).tearDown()
        shutil.rmtree(self.outputdir)

    def generate_dataset(self, classes, samples, blocks=None):
        X, _ = make_classification(n_classes=classes,
                                   n_samples=samples, n_features=20,
                                   n_informative=10, n_redundant=0,
                                   n_clusters_per_class=1,
                                   random_state=42)
        X = np.abs(X)
        X_rdd = ArrayRDD(self.sc.parallelize(X, 4), blocks)

        return X, X_rdd

    def generate_text_dataset(self, blocks=None):
        X = ALL_FOOD_DOCS
        X_rdd = ArrayRDD(self.sc.parallelize(X, 4), blocks)
        return X, X_rdd


class TestCountVectorizer(FeatureExtractionTextTestCase):

    def test_same_output(self):
        X, X_rdd = self.generate_text_dataset()
        local = CountVectorizer()
        dist = SparkCountVectorizer()

        result_local = local.fit_transform(X)
        result_dist = sp.vstack(dist.fit_transform(X_rdd).collect())

        assert_equal(local.vocabulary_, dist.vocabulary_)
        assert_array_equal(result_local.toarray(), result_dist.toarray())


class TestHashingVectorizer(FeatureExtractionTextTestCase):

    def test_same_output(self):
        X, X_rdd = self.generate_text_dataset()
        local = HashingVectorizer()
        dist = SparkHashingVectorizer()

        result_local = local.transform(X)
        result_dist = sp.vstack(dist.transform(X_rdd).collect())
        assert_array_equal(result_local.toarray(), result_dist.toarray())


class TestTfidfTransformer(FeatureExtractionTextTestCase):

    def test_same_idf_diag(self):
        X, X_rdd = self.generate_dataset(4, 1000, None)

        local = TfidfTransformer()
        dist = SparkTfidfTransformer()

        local.fit(X)
        dist.fit(X_rdd)

        assert_array_almost_equal(local._idf_diag.toarray(),
                                  dist._idf_diag.toarray())

    def test_same_transform_result(self):
        X, X_rdd = self.generate_dataset(4, 1000, None)

        local = TfidfTransformer()
        dist = SparkTfidfTransformer()

        Z_local = local.fit_transform(X)
        Z_dist = sp.vstack(dist.fit_transform(X_rdd).collect())

        assert_array_almost_equal(Z_local.toarray(),
                                  Z_dist.toarray())
