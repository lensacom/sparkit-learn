import scipy.sparse as sp
import numpy as np
from sklearn.feature_extraction.text import (CountVectorizer,
                                             HashingVectorizer,
                                             TfidfTransformer)
from splearn.feature_extraction.text import (SparkCountVectorizer,
                                             SparkHashingVectorizer,
                                             SparkTfidfTransformer)
from splearn.utils.testing import (SplearnTestCase, assert_array_almost_equal,
                                   assert_array_equal, assert_equal, assert_true)
from splearn.utils.validation import check_rdd_dtype


class TestCountVectorizer(SplearnTestCase):

    def test_same_output(self):
        X, X_rdd = self.make_text_rdd()
        local = CountVectorizer()
        dist = SparkCountVectorizer()

        result_local = local.fit_transform(X).toarray()
        result_dist = dist.fit_transform(X_rdd).toarray()

        assert_equal(local.vocabulary_, dist.vocabulary_)
        assert_array_equal(result_local, result_dist)

    def test_limit_features(self):
        X, X_rdd = self.make_text_rdd()

        params = [{'min_df': .5},
                  {'min_df': 2, 'max_df': .9},
                  {'min_df': 1, 'max_df': .6},
                  {'min_df': 2, 'max_features': 3}]

        for paramset in params:
            local = CountVectorizer(**paramset)
            dist = SparkCountVectorizer(**paramset)

            result_local = local.fit_transform(X).toarray()
            result_dist = dist.fit_transform(X_rdd).toarray()

            assert_equal(local.vocabulary_, dist.vocabulary_)
            assert_array_equal(result_local, result_dist)

            result_dist = dist.transform(X_rdd).toarray()
            assert_array_equal(result_local, result_dist)


class TestHashingVectorizer(SplearnTestCase):

    def test_same_output(self):
        X, X_rdd = self.make_text_rdd()
        local = HashingVectorizer()
        dist = SparkHashingVectorizer()

        result_local = local.transform(X).toarray()
        result_dist = dist.transform(X_rdd).toarray()
        assert_array_equal(result_local, result_dist)

    def test_dummy_analyzer(self):
        X, X_rdd = self.make_text_rdd()

        def splitter(x):
            return x.split()
        X = list(map(splitter, X))
        X_rdd = X_rdd.map(lambda x: list(map(splitter, x)))

        local = HashingVectorizer(analyzer=lambda x: x)
        dist = SparkHashingVectorizer(analyzer=lambda x: x)

        result_local = local.transform(X).toarray()
        result_dist = dist.transform(X_rdd).toarray()
        assert_array_equal(result_local, result_dist)

        result_local = local.fit_transform(X).toarray()
        result_dist = dist.fit_transform(X_rdd).toarray()
        assert_array_equal(result_local, result_dist)


class TestTfidfTransformer(SplearnTestCase):

    def test_same_transform_result(self):
        X, y, Z_rdd = self.make_classification(4, 1000, -1)
        X_rdd = Z_rdd[:, 'X']

        local = TfidfTransformer()
        dist = SparkTfidfTransformer()

        Z_local = local.fit_transform(X)
        Z_dist = dist.fit_transform(X_rdd)

        assert_true(check_rdd_dtype(Z_dist, sp.spmatrix))
        assert_array_almost_equal(Z_local.toarray(),
                                  Z_dist.toarray())
