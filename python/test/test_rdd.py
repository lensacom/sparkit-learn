import shutil
import tempfile

import numpy as np
import scipy.sparse as sp
from common import SplearnTestCase
from nose.tools import (assert_equal, assert_is_instance, assert_raises,
                        assert_true)
from numpy.testing import assert_array_equal
from sklearn.utils.testing import assert_array_almost_equal
from splearn.rdd import ArrayRDD, TupleRDD, block


def assert_equal_tuple(tpl1, tpl2):
    assert_equal(len(tpl1), len(tpl2))
    for i in range(len(tpl1)):
        assert_array_equal(tpl1[i], tpl2[i])


def assert_equal_multiple_tuples(tpls1, tpls2):
    assert_equal(len(tpls1), len(tpls2))
    for i, tpl1 in enumerate(tpls1):
        assert_equal_tuple(tpl1, tpls2[i])


class RDDTestCase(SplearnTestCase):

    def setUp(self):
        super(RDDTestCase, self).setUp()
        self.outputdir = tempfile.mkdtemp()

    def tearDown(self):
        super(RDDTestCase, self).tearDown()
        shutil.rmtree(self.outputdir)


class TestBlockRDD(RDDTestCase):

    def test_block_rdd_tuple(self):
        n_partitions = 10
        n_samples = 100
        sparse_row = sp.csr_matrix([[0, 0, 1, 0, 1]])
        data = self.sc.parallelize(
            [(np.array([1., 2.]), 0, sparse_row) for i in range(n_samples)],
            n_partitions)
        blocked_data = block(data)

        expected_first_block = np.array([[1., 2.]] * 10)
        expected_second_block = np.zeros(10, dtype=np.int)
        expected_third_block = sp.vstack([sparse_row] * 10)

        first_block_tuple = blocked_data.first()
        assert_array_almost_equal(expected_first_block, first_block_tuple[0])
        assert_array_almost_equal(expected_second_block, first_block_tuple[1])
        assert_array_almost_equal(expected_third_block.toarray(),
                                  first_block_tuple[2].toarray())

        tuple_blocks = blocked_data.collect()
        assert_equal(len(tuple_blocks), n_partitions)
        assert_equal(sum(len(b[0]) for b in tuple_blocks), n_samples)
        assert_equal(sum(len(b[1]) for b in tuple_blocks), n_samples)

    def test_block_rdd_sp_matrix(self):
        n_partitions = 10
        n_samples = 100
        sparse_row = sp.csr_matrix([[0, 0, 1, 0, 1]])
        data = self.sc.parallelize([sparse_row for i in range(n_samples)],
                                   n_partitions)
        blocked_data = block(data)
        assert_true(sp.issparse(blocked_data.first()))

        expected_block = sp.vstack([sparse_row] * 10)
        assert_array_almost_equal(expected_block.toarray(),
                                  blocked_data.first().toarray())

    def test_block_rdd_array(self):
        n_partitions = 10
        n_samples = 100
        data = self.sc.parallelize([np.array([1]) for i in range(n_samples)],
                                   n_partitions)
        blocked_data = block(data)
        assert_array_almost_equal(np.ones((10, 1)), blocked_data.first())
        blocks = blocked_data.collect()
        assert_equal(len(blocks), n_partitions)
        assert_array_almost_equal(np.ones((10, 1)), blocks[-1])
        assert_equal(sum(len(b) for b in blocks), n_samples)

        n_partitions = 17
        data = self.sc.parallelize([np.array([1]) for i in range(n_samples)],
                                   n_partitions)
        blocked_data = block(data)
        assert_array_almost_equal(np.ones((n_samples / n_partitions, 1)),
                                  blocked_data.first())
        blocks = blocked_data.collect()
        assert_equal(len(blocks), n_partitions)
        assert_equal(sum(len(b) for b in blocks), n_samples)

    def test_block_rdd_array_block_size(self):
        n_partitions = 10
        n_samples = 107
        data = self.sc.parallelize([np.array([1]) for i in range(n_samples)],
                                   n_partitions)

        block_data_5 = block(data, block_size=5)
        blocks = block_data_5.collect()
        assert_true(all(len(b) <= 5 for b in blocks))

        block_data_10 = block(data, block_size=10)
        blocks = block_data_10.collect()
        assert_true(all(len(b) <= 10 for b in blocks))

    def test_block_empty_rdd(self):
        n_partitions = 3
        empty_data = self.sc.parallelize([], n_partitions)
        assert_raises(ValueError, block, empty_data)

    def test_block_rdd_dict(self):
        n_partitions = 3
        n_samples = 57
        dicts = [{'a': i, 'b': float(i) ** 2} for i in range(n_samples)]
        data = self.sc.parallelize(dicts, n_partitions)

        block_data_5 = block(data, block_size=5)
        blocks = block_data_5.collect()
        assert_true(all(len(b) <= 5 for b in blocks))
        assert_array_almost_equal(blocks[0][0], np.arange(5))
        assert_array_almost_equal(blocks[0][1],
                                  np.arange(5, dtype=np.float) ** 2)


class TestArrayRDD(RDDTestCase):

    def test_initialization(self):
        n_partitions = 4
        n_samples = 100

        data = [np.array([1, 2]) for i in range(n_samples)]
        rdd = self.sc.parallelize(data, n_partitions)

        assert_raises(TypeError, ArrayRDD, data)
        assert_raises(TypeError, ArrayRDD, data, False)
        assert_raises(TypeError, ArrayRDD, data, 10)

        assert_is_instance(ArrayRDD(rdd), ArrayRDD)
        assert_is_instance(ArrayRDD(rdd, 10), ArrayRDD)
        assert_is_instance(ArrayRDD(rdd, None), ArrayRDD)

    def test_partitions_number(self):
        data = np.arange(400).reshape((100, 4))
        rdd = self.sc.parallelize(data, 4)
        assert_equal(ArrayRDD(rdd, 5).partitions, 4)
        assert_equal(ArrayRDD(rdd, 10).partitions, 4)
        assert_equal(ArrayRDD(rdd, 20).partitions, 4)

        data = np.arange(400).reshape((100, 4))
        rdd = self.sc.parallelize(data, 7)
        assert_equal(ArrayRDD(rdd, 5).partitions, 7)
        assert_equal(ArrayRDD(rdd, 10).partitions, 7)
        assert_equal(ArrayRDD(rdd, 20).partitions, 7)

    def test_blocks_number(self):
        n_partitions = 10
        n_samples = 1000

        data = [np.array([1, 2]) for i in range(n_samples)]
        rdd = self.sc.parallelize(data, n_partitions)

        assert_equal(1000, ArrayRDD(rdd, False).blocks)
        assert_equal(10, ArrayRDD(rdd).blocks)
        assert_equal(20, ArrayRDD(rdd, 50).blocks)
        assert_equal(20, ArrayRDD(rdd, 66).blocks)
        assert_equal(10, ArrayRDD(rdd, 100).blocks)
        assert_equal(10, ArrayRDD(rdd, 300).blocks)
        assert_equal(200, ArrayRDD(rdd, 5).blocks)
        assert_equal(100, ArrayRDD(rdd, 10).blocks)

    def test_blocks_size(self):
        n_partitions = 10
        n_samples = 1000

        data = [np.array([1, 2]) for i in range(n_samples)]
        rdd = self.sc.parallelize(data, n_partitions)

        shapes = ArrayRDD(rdd).map(lambda x: x.shape[0]).collect()
        assert_true(all(np.array(shapes) == 100))
        shapes = ArrayRDD(rdd, 5).map(lambda x: x.shape[0]).collect()
        assert_true(all(np.array(shapes) == 5))
        shapes = ArrayRDD(rdd, 50).map(lambda x: x.shape[0]).collect()
        assert_true(all(np.array(shapes) == 50))
        shapes = ArrayRDD(rdd, 250).map(lambda x: x.shape[0]).collect()
        assert_true(all(np.array(shapes) == 100))
        shapes = ArrayRDD(rdd, 66).map(lambda x: x.shape[0]).collect()
        assert_true(all(np.in1d(shapes, [66, 34])))

    def test_shape(self):
        data = np.arange(4000)
        shapes = [(1000, 4),
                  (200, 20),
                  (100, 40),
                  (2000, 2)]
        for shape in shapes:
            rdd = self.sc.parallelize(data.reshape(shape))
            assert_equal(ArrayRDD(rdd).shape, shape)

    def test_unblocking_rdd(self):
        data = np.arange(400)
        rdd = self.sc.parallelize(data, 4)
        X = ArrayRDD(rdd, 5)
        X_unblocked = X.unblock()
        assert_is_instance(X_unblocked, ArrayRDD)
        assert_array_equal(X_unblocked.take(12), np.arange(12))

    def test_convert_tolist(self):
        data = np.arange(400)
        rdd = self.sc.parallelize(data, 4)
        X = ArrayRDD(rdd, 5)
        X_list = X.tolist()
        assert_is_instance(X_list, list)
        assert_equal(X_list, data.tolist())

        data = [2, 3, 5, 1, 6, 7, 9, 9]
        rdd = self.sc.parallelize(data, 2)
        X = ArrayRDD(rdd)
        X_list = X.tolist()
        assert_is_instance(X_list, list)
        assert_equal(X_list, data)

    def test_get_single_item(self):
        data = np.arange(400).reshape((100, 4))
        rdd = self.sc.parallelize(data, 4)
        X = ArrayRDD(rdd, 5)

        expected = np.arange(0, 20).reshape((5, 4))
        assert_array_equal(X.first(), expected)
        assert_array_equal(X[0].first(), expected)
        assert_array_equal(X.ix(0).first(), expected)

        expected = np.arange(20, 40).reshape((5, 4))
        assert_array_equal(X[1].first(), expected)
        assert_array_equal(X.ix(1).first(), expected)

        expected = np.arange(380, 400).reshape((5, 4))
        assert_array_equal(X[19].first(), expected)
        assert_array_equal(X.ix(19).first(), expected)
        assert_array_equal(X[-1].first(), expected)
        assert_array_equal(X.ix(-1).first(), expected)

        expected = np.arange(340, 360).reshape((5, 4))
        assert_array_equal(X[17].first(), expected)
        assert_array_equal(X.ix(17).first(), expected)
        assert_array_equal(X[-3].first(), expected)
        assert_array_equal(X.ix(-3).first(), expected)

    def test_get_multiple_item(self):
        data = np.arange(400).reshape((100, 4))
        rdd = self.sc.parallelize(data, 4)
        X = ArrayRDD(rdd, 5)

        exp0th = np.arange(0, 20).reshape((5, 4))
        exp1st = np.arange(20, 40).reshape((5, 4))
        exp2nd = np.arange(40, 60).reshape((5, 4))
        exp7th = np.arange(140, 160).reshape((5, 4))
        exp18th = np.arange(360, 380).reshape((5, 4))
        exp19th = np.arange(380, 400).reshape((5, 4))

        assert_array_equal(X[[0, 1]].collect(), [exp0th, exp1st])
        assert_array_equal(X[[0, 2]].collect(), [exp0th, exp2nd])
        assert_array_equal(X[[0, -1]].collect(), [exp0th, exp19th])
        assert_array_equal(X[[0, -2]].collect(), [exp0th, exp18th])
        assert_array_equal(X[[1, -2]].collect(), [exp1st, exp18th])
        assert_array_equal(X[[7, 0]].collect(), [exp7th, exp0th])
        assert_array_equal(X[[1, 2, 7, 19]].collect(),
                           [exp1st, exp2nd, exp7th, exp19th])

    def test_array_slice_syntax(self):
        data = np.arange(400).reshape((100, 4))
        rdd = self.sc.parallelize(data, 4)
        X = ArrayRDD(rdd, 5)

        exp0th = np.arange(0, 20).reshape((5, 4))
        exp1st = np.arange(20, 40).reshape((5, 4))
        exp7th = np.arange(140, 160).reshape((5, 4))
        exp8th = np.arange(160, 180).reshape((5, 4))
        exp9th = np.arange(180, 200).reshape((5, 4))
        exp18th = np.arange(360, 380).reshape((5, 4))
        exp19th = np.arange(380, 400).reshape((5, 4))

        assert_array_equal(X[:1].collect(), [exp0th])
        assert_array_equal(X[:2].collect(), [exp0th, exp1st])
        assert_array_equal(X[18:].collect(), [exp18th, exp19th])
        assert_array_equal(X[-1:].collect(), [exp19th])
        assert_array_equal(X[-2:].collect(), [exp18th, exp19th])
        assert_array_equal(X[7:10].collect(), [exp7th, exp8th, exp9th])
        assert_array_equal(X[7:10:2].collect(), [exp7th, exp9th])
        assert_array_equal(X[::9].collect(), [exp0th, exp9th, exp18th])
        assert_array_equal(X[::-10].collect(), [exp19th, exp9th])
        assert_array_equal(X[-1:1].collect(), [])

    def test_transform(self):
        data = np.arange(400).reshape((100, 4))
        rdd = self.sc.parallelize(data, 4)
        X = ArrayRDD(rdd, 5)

        fn = lambda x: x**2
        X1 = map(fn, X.collect())
        X2 = X.transform(fn).collect()

        assert_array_equal(X1, X2)


class TestTupleRDD(RDDTestCase):

    def test_initialization(self):
        n_partitions = 4
        n_samples = 100

        data = [(1, 2) for i in range(n_samples)]
        rdd = self.sc.parallelize(data, n_partitions)

        assert_raises(TypeError, TupleRDD, data)
        assert_raises(TypeError, TupleRDD, data, False)
        assert_raises(TypeError, TupleRDD, data, 10)

        assert_is_instance(TupleRDD(rdd), TupleRDD)
        assert_is_instance(TupleRDD(rdd), ArrayRDD)
        assert_is_instance(TupleRDD(rdd, 10), TupleRDD)
        assert_is_instance(TupleRDD(rdd), ArrayRDD)
        assert_is_instance(TupleRDD(rdd, None), TupleRDD)
        assert_is_instance(TupleRDD(rdd), ArrayRDD)

    def test_shape(self):
        data = np.arange(4000)
        shapes = [(1000, 4),
                  (200, 20),
                  (100, 40),
                  (2000, 2)]
        for shape in shapes:
            rdd = self.sc.parallelize(data.reshape(shape))
            rdd1 = rdd.zipWithIndex()
            rdd2 = rdd.map(lambda x: (x, 1, 2, 3, 4, True))
            assert_equal(TupleRDD(rdd1).shape, (shape[0], 2))
            assert_equal(TupleRDD(rdd2).shape, (shape[0], 6))

    def test_get_single_tuple(self):
        x, y = np.arange(80).reshape((40, 2)), np.arange(40)
        x_rdd = self.sc.parallelize(x, 2)
        y_rdd = self.sc.parallelize(y, 2)
        z_rdd = x_rdd.zip(y_rdd)
        z = TupleRDD(z_rdd, 5)

        expected = np.arange(0, 10).reshape((5, 2)), np.arange(5)
        for tpl in [z.first(), z.ix(0).first(), z[0].first()]:
            assert_equal_tuple(tpl, expected)

        expected = np.arange(30, 40).reshape((5, 2)), np.arange(15, 20)
        for tpl in [z.ix(3).first(), z[3].first(), z[-5].first()]:
            assert_equal_tuple(tpl, expected)

        expected = np.arange(70, 80).reshape((5, 2)), np.arange(35, 40)
        for tpl in [z.ix(7).first(), z[7].first(), z[-1].first()]:
            assert_equal_tuple(tpl, expected)

    def test_get_single_item(self):
        x, y = np.arange(80).reshape((40, 2)), np.arange(40)
        x_rdd = self.sc.parallelize(x, 2)
        y_rdd = self.sc.parallelize(y, 2)
        z_rdd = x_rdd.zip(y_rdd)
        z = TupleRDD(z_rdd, 5)

        assert_array_equal(z[0, 0].first(), np.arange(0, 10).reshape((5, 2)))
        assert_array_equal(z[0, 1].first(), np.arange(5))

        assert_array_equal(z[3, 0].first(), np.arange(30, 40).reshape((5, 2)))
        assert_array_equal(z[3, 1].first(), np.arange(15, 20))
        assert_array_equal(z[3, -1].first(), np.arange(15, 20))

        assert_array_equal(z[7, 0].first(), np.arange(70, 80).reshape((5, 2)))
        assert_array_equal(z[-1, 0].first(), np.arange(70, 80).reshape((5, 2)))
        assert_array_equal(z[7, 1].first(), np.arange(35, 40))
        assert_array_equal(z[-1, -1].first(), np.arange(35, 40))

    def test_get_multiple_tuples(self):
        x, y = np.arange(80).reshape((40, 2)), np.arange(40)
        x_rdd = self.sc.parallelize(x, 2)
        y_rdd = self.sc.parallelize(y, 2)
        z_rdd = x_rdd.zip(y_rdd)
        z = TupleRDD(z_rdd, 5)

        expected = [(np.arange(0, 10).reshape((5, 2)), np.arange(0, 5)),
                    (np.arange(10, 20).reshape((5, 2)), np.arange(5, 10))]
        assert_equal_multiple_tuples(z[:2].collect(), expected)
        assert_equal_multiple_tuples(z[:2, :].collect(), expected)
        assert_equal_multiple_tuples(z[[0, 1]].collect(), expected)
        assert_equal_multiple_tuples(z[[0, 1], :].collect(), expected)
        assert_equal_multiple_tuples(z[[1, 0]].collect(), expected[::-1])

        expected = [(np.arange(50, 60).reshape((5, 2)), np.arange(25, 30)),
                    (np.arange(60, 70).reshape((5, 2)), np.arange(30, 35)),
                    (np.arange(70, 80).reshape((5, 2)), np.arange(35, 40))]
        assert_equal_multiple_tuples(z[-3:].collect(), expected)
        assert_equal_multiple_tuples(z[-3:, :].collect(), expected)
        assert_equal_multiple_tuples(z[[5, 6, 7]].collect(), expected)
        assert_equal_multiple_tuples(z[[5, 6, 7], :].collect(), expected)
        assert_equal_multiple_tuples(z[[7, 6, 5]].collect(), expected[::-1])
        assert_equal_multiple_tuples(z[[7, 6, 5], :].collect(), expected[::-1])
        assert_equal_multiple_tuples(z[[5, 7, 6]].collect(),
                                     [expected[0], expected[2], expected[1]])

    def test_get_multiple_items(self):
        x, y = np.arange(80).reshape((40, 2)), np.arange(40)
        x_rdd = self.sc.parallelize(x, 2)
        y_rdd = self.sc.parallelize(y, 2)
        z_rdd = x_rdd.zip(y_rdd)
        z = TupleRDD(z_rdd, 5)

        expected = [(np.arange(0, 10).reshape((5, 2)), np.arange(0, 5)),
                    (np.arange(10, 20).reshape((5, 2)), np.arange(5, 10))]
        assert_array_equal(z[:2, 1].collect(),
                           [expected[0][1], expected[1][1]])
        assert_array_equal(z[[0, 1], 0].collect(),
                           [expected[0][0], expected[1][0]])
        assert_equal_multiple_tuples(z[[0, 1], -1:].collect(),
                                     [(expected[0][1],),
                                      (expected[1][1],)])
        assert_equal_multiple_tuples(z[[0, 1], -1:].collect(),
                                     [(expected[0][1],),
                                      (expected[1][1],)])
        assert_equal_multiple_tuples(z[[1, 0], [1, 0]].collect(),
                                     [expected[1][::-1], expected[0][::-1]])


class TestDictRDD(RDDTestCase):
    pass
