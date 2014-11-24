import shutil
import tempfile
import numpy as np
import scipy.sparse as sp

from nose.tools import assert_equal
from nose.tools import assert_true
from numpy.testing import assert_array_almost_equal
from numpy.testing import assert_array_equal

from common import SplearnTestCase
from splearn.rdd import ArrayRDD, DictRDD
from splearn.feature_extraction import SparkDictVectorizer

from sklearn.feature_extraction import DictVectorizer


class FeatureExtractionDictVectorizerTestCase(SplearnTestCase):

    def setUp(self):
        super(FeatureExtractionDictVectorizerTestCase, self).setUp()
        self.outputdir = tempfile.mkdtemp()

    def tearDown(self):
        super(FeatureExtractionDictVectorizerTestCase, self).tearDown()
        shutil.rmtree(self.outputdir)

    def generate_dict_dataset(self, blocks=None):
        X = [{"foo": 1, "bar": 3},
             {"bar": 4, "baz": 2},
             {"bar": 6, "baz": 1},
             {"bar": 4, "ewo": "ok"},
             {"bar": 4, "baz": 2},
             {"bar": 9, "ewo": "fail"},
             {"bar": 4, "baz": 2},
             {"bar": 1, "quux": 1, "quuux": 2}]
        X_rdd = ArrayRDD(self.sc.parallelize(X, 4), blocks)
        return X, X_rdd


class TestDictVectorizer(FeatureExtractionDictVectorizerTestCase):

    def test_same_output(self):
        X, X_rdd = self.generate_dict_dataset()
        local = DictVectorizer()
        dist = SparkDictVectorizer()

        result_local = local.fit_transform(X)
        result_dist = sp.vstack(dist.fit_transform(X_rdd).collect())

        assert_equal(local.vocabulary_, dist.vocabulary_)
        assert_array_equal(result_local.toarray(), result_dist.toarray())
