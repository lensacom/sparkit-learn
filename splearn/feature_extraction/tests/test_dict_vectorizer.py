import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction import DictVectorizer as SklearnDictVectorizer
from splearn.feature_extraction import DictVectorizer
from splearn.rdd import ArrayRDD
from splearn.utils.testing import (SplearnTestCase, assert_array_equal,
                                   assert_equal, assert_true)
from splearn.utils.validation import check_rdd_dtype


class TestDictVectorizer(SplearnTestCase):

    def make_dict_dataset(self, blocks=-1):
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

    def test_same_output_dense(self):
        X, X_rdd = self.make_dict_dataset()
        scikit = SklearnDictVectorizer(sparse=False)
        sparkit = DictVectorizer(sparse=False)

        result_true = scikit.fit_transform(X)
        result_local = sparkit.fit_transform(X)
        result_dist = sparkit.fit_transform(X_rdd)

        assert_true(check_rdd_dtype(result_dist, (np.ndarray,)))
        assert_equal(scikit.vocabulary_, sparkit.vocabulary_)

        assert_array_equal(result_true, result_local)
        assert_array_equal(result_true, result_dist.toarray())

    def test_same_output_sparse(self):
        X, X_rdd = self.make_dict_dataset()
        scikit = SklearnDictVectorizer(sparse=True)
        sparkit = DictVectorizer(sparse=True)

        result_true = scikit.fit_transform(X).toarray()
        result_local = sparkit.fit_transform(X).toarray()
        result_dist = sparkit.fit_transform(X_rdd)

        assert_true(check_rdd_dtype(result_dist, (sp.spmatrix,)))
        assert_equal(scikit.vocabulary_, sparkit.vocabulary_)

        assert_array_equal(result_true, result_local)
        assert_array_equal(result_true, result_dist.toarray())
