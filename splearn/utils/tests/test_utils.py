import numpy as np
import scipy.sparse as sp
from sklearn.utils.testing import assert_false, assert_raises, assert_true
from splearn.rdd import ArrayRDD, DictRDD, SparseRDD
from splearn.utils.testing import SplearnTestCase
from splearn.utils.validation import check_rdd, check_rdd_dtype


class TestUtilities(SplearnTestCase):

    def test_check_rdd_dtype(self):
        array = np.ndarray
        spmat = sp.spmatrix

        dense, dense_rdd = self.make_dense_rdd(block_size=5)
        sparse, sparse_rdd = self.make_sparse_rdd(block_size=5)

        dict_rdd = DictRDD(
            (dense_rdd, sparse_rdd),
            columns=('X', 'y'),
            bsize=5
        )

        assert_true(check_rdd_dtype(dense_rdd, array))
        assert_true(check_rdd_dtype(dense_rdd, (array, spmat)))
        assert_true(check_rdd_dtype(sparse_rdd, spmat))
        assert_true(check_rdd_dtype(dict_rdd, {'X': array}))
        assert_true(check_rdd_dtype(dict_rdd, {'y': spmat}))
        assert_true(check_rdd_dtype(dict_rdd, {'X': array, 'y': spmat}))
        assert_true(check_rdd_dtype(dict_rdd, {'X': (array, spmat), 'y': spmat}))

        assert_false(check_rdd_dtype(dense_rdd, spmat))
        assert_false(check_rdd_dtype(sparse_rdd, (array,)))
        assert_false(check_rdd_dtype(dict_rdd, {'X': spmat}))

        assert_raises(TypeError, check_rdd_dtype, (dict_rdd, (tuple,)))
        assert_raises(TypeError, check_rdd_dtype, (np.arange(20), (array,)))

    def test_check_rdd(self):
        array = np.ndarray
        spmat = sp.spmatrix

        dense, dense_rdd = self.make_dense_rdd(block_size=5)
        sparse, sparse_rdd = self.make_sparse_rdd(block_size=5)

        dict_rdd = DictRDD(
            (dense_rdd, sparse_rdd),
            columns=('X', 'y'),
            bsize=5
        )

        assert_raises(TypeError, check_rdd, (dense_rdd, spmat))
        assert_raises(TypeError, check_rdd, (sparse_rdd, (array,)))
        assert_raises(TypeError, check_rdd, (dict_rdd, {'X': (array, spmat)}))
        assert_raises(TypeError, check_rdd, (dict_rdd, (tuple,)))
        assert_raises(TypeError, check_rdd, (np.arange(20), (array,)))
