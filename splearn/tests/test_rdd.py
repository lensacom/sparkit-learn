import numpy as np
import scipy.sparse as sp
from pyspark import RDD
from splearn.rdd import ArrayRDD, BlockRDD, DictRDD, SparseRDD, block
from splearn.utils.testing import (SplearnTestCase, assert_almost_equal,
                                   assert_array_almost_equal,
                                   assert_array_equal, assert_equal,
                                   assert_is_instance,
                                   assert_multiple_tuples_equal, assert_raises,
                                   assert_true, assert_tuple_equal)
from splearn.utils.validation import check_rdd_dtype


class TestBlocking(SplearnTestCase):

    def test_empty(self):
        n_partitions = 3
        empty_data = self.sc.parallelize([], n_partitions)
        assert_raises(ValueError, block, empty_data)

    def test_dtype(self):
        n_partitions = 10
        n_samples = 100
        data = self.sc.parallelize(["lorem" for i in range(n_samples)],
                                   n_partitions)
        blocked_data = block(data, dtype=list)
        assert_array_equal(["lorem"] * 10, blocked_data.first())
        blocks = blocked_data.collect()
        assert_equal(len(blocks), n_partitions)
        assert_array_equal(["lorem"] * 10, blocks[-1])
        assert_equal(sum(len(b) for b in blocks), n_samples)

        n_partitions = 17
        data = self.sc.parallelize([1 for i in range(n_samples)],
                                   n_partitions)
        blocked_data = block(data, dtype=tuple)
        assert_array_equal(tuple([1] * (n_samples // n_partitions)),
                           blocked_data.first())
        blocks = blocked_data.collect()
        assert_equal(len(blocks), n_partitions)
        assert_equal(sum(len(b) for b in blocks), n_samples)

    def test_array(self):
        n_partitions = 10
        n_samples = 100
        data = self.sc.parallelize([np.array([1]) for i in range(n_samples)],
                                   n_partitions)
        blocked_data = block(data)
        assert_array_equal(np.ones((10, 1)), blocked_data.first())
        blocks = blocked_data.collect()
        assert_equal(len(blocks), n_partitions)
        assert_array_equal(np.ones((10, 1)), blocks[-1])
        assert_equal(sum(len(b) for b in blocks), n_samples)

        n_partitions = 17
        data = self.sc.parallelize([np.array([1]) for i in range(n_samples)],
                                   n_partitions)
        blocked_data = block(data)
        assert_array_equal(np.ones((n_samples // n_partitions, 1)),
                           blocked_data.first())
        blocks = blocked_data.collect()
        assert_equal(len(blocks), n_partitions)
        assert_equal(sum(len(b) for b in blocks), n_samples)

    def test_array_bsize(self):
        n_partitions = 10
        n_samples = 107
        data = self.sc.parallelize([np.array([1]) for i in range(n_samples)],
                                   n_partitions)

        block_data_5 = block(data, bsize=5)
        blocks = block_data_5.collect()
        assert_true(all(len(b) <= 5 for b in blocks))

        block_data_10 = block(data, bsize=10)
        blocks = block_data_10.collect()
        assert_true(all(len(b) <= 10 for b in blocks))

    def test_sparse_matrix(self):
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

    def test_block_rdd_dict(self):
        n_partitions = 3
        n_samples = 57
        dicts = [{'a': i, 'b': float(i) ** 2} for i in range(n_samples)]
        data = self.sc.parallelize(dicts, n_partitions)

        block_data_5 = block(data, bsize=5)
        blocks = block_data_5.collect()
        assert_true(all(len(b) <= 5 for b in blocks))
        assert_array_almost_equal(blocks[0][0], np.arange(5))
        assert_array_almost_equal(blocks[0][1],
                                  np.arange(5, dtype=np.float) ** 2)


class TestBlockRDD(SplearnTestCase):

    def generate(self, n_samples=100, n_partitions=10):
        return self.sc.parallelize(list(range(n_samples)), n_partitions)

    def test_creation(self):
        rdd = self.generate()

        blocked = BlockRDD(rdd)
        assert_is_instance(blocked, BlockRDD)
        expected = tuple(range(10))
        assert_equal(blocked.first(), expected)
        expected = [tuple(v) for v in np.arange(100).reshape(10, 10)]
        assert_equal(blocked.collect(), expected)

        blocked = BlockRDD(rdd, bsize=4)
        assert_is_instance(blocked, BlockRDD)
        expected = tuple(range(4))
        assert_equal(blocked.first(), expected)
        expected = [4, 4, 2] * 10
        assert_equal([len(x) for x in blocked.collect()], expected)

    def test_dtypes(self):
        rdd = self.generate()
        blocked = BlockRDD(rdd, dtype=list)
        assert_is_instance(blocked.first(), list)
        blocked = BlockRDD(rdd, dtype=tuple)
        assert_is_instance(blocked.first(), tuple)
        blocked = BlockRDD(rdd, dtype=set)
        assert_is_instance(blocked.first(), set)
        blocked = BlockRDD(rdd, dtype=np.array)
        assert_is_instance(blocked.first(), np.ndarray)

    def test_length(self):
        blocked = BlockRDD(self.generate(1000))
        assert_equal(len(blocked), 1000)
        blocked = BlockRDD(self.generate(100))
        assert_equal(len(blocked), 100)
        blocked = BlockRDD(self.generate(79))
        assert_equal(len(blocked), 79)
        blocked = BlockRDD(self.generate(89))
        assert_equal(len(blocked), 89)
        blocked = BlockRDD(self.generate(62))
        assert_equal(len(blocked), 62)

    def test_blocks_number(self):
        blocked = BlockRDD(self.generate(1000), bsize=50)
        assert_equal(blocked.blocks, 20)
        blocked = BlockRDD(self.generate(621), bsize=45)
        assert_equal(blocked.blocks, 20)
        blocked = BlockRDD(self.generate(100), bsize=4)
        assert_equal(blocked.blocks, 30)
        blocked = BlockRDD(self.generate(79, 2), bsize=9)
        assert_equal(blocked.blocks, 10)
        blocked = BlockRDD(self.generate(89, 2), bsize=5)
        assert_equal(blocked.blocks, 18)

    def test_partition_number(self):
        blocked = BlockRDD(self.generate(1000, 5), bsize=50)
        assert_equal(blocked.partitions, 5)
        blocked = BlockRDD(self.generate(621, 3), bsize=45)
        assert_equal(blocked.partitions, 3)
        blocked = BlockRDD(self.generate(100, 10))
        assert_equal(blocked.partitions, 10)

    def test_unblock(self):
        blocked = BlockRDD(self.generate(1000, 5))
        unblocked = blocked.unblock()
        assert_is_instance(blocked, BlockRDD)
        assert_equal(unblocked.collect(), list(range(1000)))

        blocked = BlockRDD(self.generate(1000, 5), dtype=tuple)
        unblocked = blocked.unblock()
        assert_is_instance(blocked, BlockRDD)
        assert_equal(unblocked.collect(), list(range(1000)))

    def test_tolist(self):
        blocked = BlockRDD(self.generate(1000, 5))
        unblocked = blocked.tolist()
        assert_is_instance(blocked, BlockRDD)
        assert_equal(unblocked, list(range(1000)))

        blocked = BlockRDD(self.generate(1000, 5), dtype=tuple)
        unblocked = blocked.tolist()
        assert_is_instance(blocked, BlockRDD)
        assert_equal(unblocked, list(range(1000)))

        blocked = BlockRDD(self.generate(1000, 5), dtype=np.array)
        unblocked = blocked.tolist()
        assert_is_instance(blocked, BlockRDD)
        assert_equal(unblocked, list(range(1000)))


class TestArrayRDD(SplearnTestCase):

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

        assert_equal(1000, ArrayRDD(rdd, noblock=True, bsize=1).blocks)
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

    def test_ndim(self):
        data = np.arange(4000)
        shapes = [(4000),
                  (1000, 4),
                  (200, 10, 2),
                  (100, 10, 2, 2)]
        for shape in shapes:
            reshaped = data.reshape(shape)
            rdd = self.sc.parallelize(reshaped)
            assert_equal(ArrayRDD(rdd).ndim, reshaped.ndim)

    def test_shape(self):
        data = np.arange(4000)
        shapes = [(1000, 4),
                  (200, 20),
                  (100, 40),
                  (2000, 2)]
        for shape in shapes:
            reshaped = data.reshape(shape)
            rdd = self.sc.parallelize(reshaped)
            assert_equal(ArrayRDD(rdd).shape, shape)

    def test_size(self):
        data = np.arange(4000)
        shapes = [(1000, 4),
                  (200, 20),
                  (100, 40),
                  (2000, 2)]
        for shape in shapes:
            reshaped = data.reshape(shape)
            rdd = self.sc.parallelize(reshaped)
            size = ArrayRDD(rdd).map(lambda x: x.size).sum()
            assert_equal(size, reshaped.size)
            assert_equal(ArrayRDD(rdd).size, reshaped.size)

    def test_unblocking_rdd(self):
        data = np.arange(400)
        rdd = self.sc.parallelize(data, 4)
        X = ArrayRDD(rdd, 5)
        X_unblocked = X.unblock()
        assert_is_instance(X_unblocked, RDD)
        assert_array_equal(X_unblocked.take(12), np.arange(12).tolist())

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

    def test_convert_toarray(self):
        data = np.arange(400)
        rdd = self.sc.parallelize(data, 4)
        X = ArrayRDD(rdd, 5)
        X_array = X.toarray()
        assert_array_equal(X_array, data)

        data = [2, 3, 5, 1, 6, 7, 9, 9]
        rdd = self.sc.parallelize(data, 2)
        X = ArrayRDD(rdd)
        X_array = X.toarray()
        assert_array_equal(X_array, np.array(data))

    def test_get_single_item(self):
        data = np.arange(400).reshape((100, 4))
        rdd = self.sc.parallelize(data, 4)
        X = ArrayRDD(rdd, 5)

        expected = np.arange(0, 20).reshape((5, 4))
        assert_array_equal(X.first(), expected)
        assert_array_equal(X[0].first(), expected)

        expected = np.arange(20, 40).reshape((5, 4))
        assert_array_equal(X[1].first(), expected)

        expected = np.arange(380, 400).reshape((5, 4))
        assert_array_equal(X[19].first(), expected)
        assert_array_equal(X[-1].first(), expected)

        expected = np.arange(340, 360).reshape((5, 4))
        assert_array_equal(X[17].first(), expected)
        assert_array_equal(X[-3].first(), expected)

    def test_get_multiple_item(self):
        X, X_rdd = self.make_dense_range_rdd((100, 4), block_size=5)

        exp0th = np.arange(0, 20).reshape((5, 4))
        exp1st = np.arange(20, 40).reshape((5, 4))
        exp2nd = np.arange(40, 60).reshape((5, 4))
        exp7th = np.arange(140, 160).reshape((5, 4))
        exp18th = np.arange(360, 380).reshape((5, 4))
        exp19th = np.arange(380, 400).reshape((5, 4))

        assert_array_equal(X_rdd[[0, 1]].collect(), [exp0th, exp1st])
        assert_array_equal(X_rdd[[0, 2]].collect(), [exp0th, exp2nd])
        assert_array_equal(X_rdd[[0, -1]].collect(), [exp0th, exp19th])
        assert_array_equal(X_rdd[[0, -2]].collect(), [exp0th, exp18th])
        assert_array_equal(X_rdd[[1, -2]].collect(), [exp1st, exp18th])
        assert_array_equal(X_rdd[[7, 0]].collect(), [exp7th, exp0th])
        assert_array_equal(X_rdd[[1, 2, 7, 19]].collect(),
                           [exp1st, exp2nd, exp7th, exp19th])

    def test_array_slice_syntax(self):
        X, X_rdd = self.make_dense_range_rdd((100, 4), block_size=5)

        exp0th = np.arange(0, 20).reshape((5, 4))
        exp1st = np.arange(20, 40).reshape((5, 4))
        exp7th = np.arange(140, 160).reshape((5, 4))
        exp8th = np.arange(160, 180).reshape((5, 4))
        exp9th = np.arange(180, 200).reshape((5, 4))
        exp18th = np.arange(360, 380).reshape((5, 4))
        exp19th = np.arange(380, 400).reshape((5, 4))

        assert_array_equal(X_rdd[:1].collect(), [exp0th])
        assert_array_equal(X_rdd[:2].collect(), [exp0th, exp1st])
        assert_array_equal(X_rdd[18:].collect(), [exp18th, exp19th])
        assert_array_equal(X_rdd[-1:].collect(), [exp19th])
        assert_array_equal(X_rdd[-2:].collect(), [exp18th, exp19th])
        assert_array_equal(X_rdd[7:10].collect(), [exp7th, exp8th, exp9th])
        assert_array_equal(X_rdd[7:10:2].collect(), [exp7th, exp9th])
        assert_array_equal(X_rdd[::9].collect(), [exp0th, exp9th, exp18th])
        assert_array_equal(X_rdd[::-10].collect(), [exp19th, exp9th])
        assert_array_equal(X_rdd[-1:1].collect(), [])

    def test_transform(self):
        X, X_rdd = self.make_dense_rdd((100, 4))

        fn = lambda x: x ** 2
        X1 = list(map(fn, X_rdd.collect()))
        X2 = X_rdd.transform(fn).collect()

        assert_array_equal(X1, X2)

    def test_transform_dtype(self):
        X, X_rdd = self.make_dense_rdd((100, 4))

        rdd = X_rdd.transform(lambda x: x)
        assert_is_instance(rdd, ArrayRDD)
        rdd = X_rdd.transform(lambda x: x.tolist(), dtype=list)
        assert_is_instance(rdd, BlockRDD)
        rdd = X_rdd.transform(lambda x: sp.lil_matrix(x), dtype=sp.spmatrix)
        assert_is_instance(rdd, SparseRDD)


class TestDenseMath(SplearnTestCase):

    def _test_func_on_axis(self, func):
        X, X_rdd = self.make_dense_rdd(block_size=100)
        assert_almost_equal(getattr(X_rdd, func)(), getattr(X, func)())
        for axes in (0, 1):
            assert_array_almost_equal(getattr(X_rdd, func)(axis=axes),
                                      getattr(X, func)(axis=axes))

        X, X_rdd = self.make_dense_rdd((100, 3, 2))
        assert_almost_equal(getattr(X_rdd, func)(), getattr(X, func)())
        for axes in (0, 1, 2):
            assert_array_almost_equal(getattr(X_rdd, func)(axis=axes),
                                      getattr(X, func)(axis=axes))

    def test_min(self):
        self._test_func_on_axis('min')

    def test_max(self):
        self._test_func_on_axis('max')

    def test_sum(self):
        self._test_func_on_axis('sum')

    def test_prod(self):
        self._test_func_on_axis('prod')

    def test_mean(self):
        self._test_func_on_axis('mean')

    def test_dot(self):
        A, A_rdd = self.make_dense_rdd((20, 10))
        B, B_rdd = self.make_dense_rdd((10, 20))
        assert_array_equal(A_rdd.dot(B).toarray(), A.dot(B))
        assert_array_equal(B_rdd.dot(A).toarray(), B.dot(A))

    def test_add(self):
        A, A_rdd = self.make_dense_rdd((8, 3))
        B, B_rdd = self.make_dense_rdd((1, 3))
        np_res = A + B
        assert_array_equal(
            A_rdd.add(B).toarray(), np_res
        )
        assert_array_equal((A_rdd + B).toarray(), np_res)
        A_rdd += B
        assert_array_equal(A_rdd.toarray(), np_res)

    def test_subtract(self):
        A, A_rdd = self.make_dense_rdd((8, 3))
        B, B_rdd = self.make_dense_rdd((1, 3))
        np_res = A - B
        assert_array_equal(
            A_rdd.subtract(B).toarray(), np_res
        )
        assert_array_equal((A_rdd - B).toarray(), np_res)
        A_rdd -= B
        assert_array_equal(A_rdd.toarray(), np_res)

    def test_multiply(self):
        A, A_rdd = self.make_dense_rdd((8, 3))
        B, B_rdd = self.make_dense_rdd((1, 3))
        np_res = A * B
        assert_array_equal(
            A_rdd.multiply(B).toarray(), np_res
        )
        assert_array_equal((A_rdd * B).toarray(), np_res)
        A_rdd *= B
        assert_array_equal(A_rdd.toarray(), np_res)

    def test_divide(self):
        A, A_rdd = self.make_dense_rdd((8, 3))
        B, B_rdd = self.make_dense_rdd((1, 3))
        np_res = A / B
        assert_array_equal(
            A_rdd.divide(B).toarray(), np_res
        )
        assert_array_equal((A_rdd / B).toarray(), np_res)
        A_rdd /= B
        assert_array_equal(A_rdd.toarray(), np_res)

    def test_power(self):
        A, A_rdd = self.make_dense_rdd((8, 3))
        B, B_rdd = self.make_dense_rdd((1, 3))
        np_res = A ** B
        assert_array_equal(
            A_rdd.power(B).toarray(), np_res
        )
        assert_array_equal((A_rdd ** B).toarray(), np_res)
        A_rdd **= B
        assert_array_equal(A_rdd.toarray(), np_res)

    def test_floor_divide(self):
        A, A_rdd = self.make_dense_rdd((8, 3))
        B, B_rdd = self.make_dense_rdd((1, 3))
        np_res = A // B
        assert_array_equal(
            A_rdd.floor_divide(B).toarray(), np_res
        )
        assert_array_equal((A_rdd // B).toarray(), np_res)
        A_rdd //= B
        assert_array_equal(A_rdd.toarray(), np_res)

    def test_true_divide(self):
        A, A_rdd = self.make_dense_rdd((8, 3))
        B, B_rdd = self.make_dense_rdd((1, 3))
        np_res = A / B
        assert_array_equal(
            A_rdd.true_divide(B).toarray(), np_res
        )

    def test_mod(self):
        A, A_rdd = self.make_dense_rdd((8, 3))
        B, B_rdd = self.make_dense_rdd((1, 3))
        np_res = A % B
        assert_array_equal(
            A_rdd.mod(B).toarray(), np_res
        )
        assert_array_equal((A_rdd % B).toarray(), np_res)
        A_rdd %= B
        assert_array_equal(A_rdd.toarray(), np_res)

    def test_fmod(self):
        A, A_rdd = self.make_dense_rdd((8, 3))
        B, B_rdd = self.make_dense_rdd((1, 3))
        np_res = np.fmod(A, B)
        assert_array_equal(
            A_rdd.fmod(B).toarray(), np_res
        )

    def test_remainder(self):
        A, A_rdd = self.make_dense_rdd((8, 3))
        B, B_rdd = self.make_dense_rdd((1, 3))
        np_res = np.remainder(A, B)
        assert_array_equal(
            A_rdd.remainder(B).toarray(), np_res
        )

    def test_flatten(self):
        X, X_rdd = self.make_dense_rdd((100, 3, 2))
        X = X.flatten()
        X_rdd = X_rdd.flatten()
        assert_array_equal(X_rdd.toarray(), X)


class TestSparseMath(SplearnTestCase):

    def _test_func_on_axis(self, func, toarray=True):
        X, X_rdd = self.make_sparse_rdd(block_size=100)
        assert_almost_equal(getattr(X_rdd, func)(), getattr(X, func)())
        for axes in (0, 1):
            if toarray:
                assert_array_almost_equal(
                    getattr(X_rdd, func)(axis=axes).toarray(),
                    getattr(X, func)(axis=axes).toarray())
            else:
                assert_array_almost_equal(
                    getattr(X_rdd, func)(axis=axes),
                    getattr(X, func)(axis=axes))

    def test_min(self):
        self._test_func_on_axis('min')

    def test_max(self):
        self._test_func_on_axis('max')

    def test_sum(self):
        self._test_func_on_axis('sum', toarray=False)

    def test_mean(self):
        self._test_func_on_axis('mean', toarray=False)

    def test_dot(self):
        A, A_rdd = self.make_sparse_rdd((20, 10))
        B, B_rdd = self.make_sparse_rdd((10, 20))
        assert_array_almost_equal(A_rdd.dot(B).toarray(), A.dot(B).toarray())
        assert_array_almost_equal(B_rdd.dot(A).toarray(), B.dot(A).toarray())


class TestDictRDD(SplearnTestCase):

    def test_initialization(self):
        n_partitions = 4
        n_samples = 100

        data = [(1, 2) for i in range(n_samples)]
        rdd = self.sc.parallelize(data, n_partitions)

        assert_raises(TypeError, DictRDD, data)
        assert_raises(TypeError, DictRDD, data, bsize=False)
        assert_raises(TypeError, DictRDD, data, bsize=10)

        assert_is_instance(DictRDD(rdd), DictRDD)
        assert_is_instance(DictRDD(rdd), BlockRDD)
        assert_is_instance(DictRDD(rdd, bsize=10), DictRDD)
        assert_is_instance(DictRDD(rdd), BlockRDD)
        assert_is_instance(DictRDD(rdd, bsize=None), DictRDD)
        assert_is_instance(DictRDD(rdd), BlockRDD)

    def test_creation_from_zipped_rdd(self):
        x = np.arange(80).reshape((40, 2))
        y = range(40)
        x_rdd = self.sc.parallelize(x, 4)
        y_rdd = self.sc.parallelize(y, 4)
        zipped_rdd = x_rdd.zip(y_rdd)

        expected = (np.arange(20).reshape(10, 2), tuple(range(10)))

        rdd = DictRDD(zipped_rdd)
        assert_tuple_equal(rdd.first(), expected)
        rdd = DictRDD(zipped_rdd, columns=('x', 'y'))
        assert_tuple_equal(rdd.first(), expected)
        rdd = DictRDD(zipped_rdd, dtype=(np.ndarray, list))
        first = rdd.first()
        assert_tuple_equal(first, expected)
        assert_is_instance(first[1], list)

    def test_creation_from_rdds(self):
        x = np.arange(80).reshape((40, 2))
        y = np.arange(40)
        z = list(range(40))
        x_rdd = self.sc.parallelize(x, 4)
        y_rdd = self.sc.parallelize(y, 4)
        z_rdd = self.sc.parallelize(z, 4)

        expected = (
            np.arange(20).reshape(10, 2),
            np.arange(10), list(range(10))
        )
        rdd = DictRDD([x_rdd, y_rdd, z_rdd])
        assert_tuple_equal(rdd.first(), expected)
        rdd = DictRDD([x_rdd, y_rdd, z_rdd], columns=('x', 'y', 'z'))
        assert_tuple_equal(rdd.first(), expected)
        rdd = DictRDD([x_rdd, y_rdd, z_rdd],
                      dtype=(np.ndarray, np.ndarray, list))
        first = rdd.first()
        assert_tuple_equal(first, expected)
        assert_is_instance(first[2], list)

    def test_creation_from_blocked_rdds(self):
        x = np.arange(80).reshape((40, 2))
        y = np.arange(40)
        z = list(range(40))
        x_rdd = ArrayRDD(self.sc.parallelize(x, 4))
        y_rdd = ArrayRDD(self.sc.parallelize(y, 4))
        z_rdd = BlockRDD(self.sc.parallelize(z, 4), dtype=list)

        expected = (
            np.arange(20).reshape(10, 2),
            np.arange(10), list(range(10))
        )
        rdd = DictRDD([x_rdd, y_rdd, z_rdd])
        assert_tuple_equal(rdd.first(), expected)
        rdd = DictRDD([x_rdd, y_rdd, z_rdd], columns=('x', 'y', 'z'))
        assert_tuple_equal(rdd.first(), expected)
        rdd = DictRDD([x_rdd, y_rdd, z_rdd], dtype=(None, None, list))
        first = rdd.first()
        assert_tuple_equal(first, expected)
        assert_is_instance(first[2], list)

    def test_auto_dtype(self):
        x = np.arange(80).reshape((40, 2))
        y = tuple(range(40))
        z = list(range(40))
        x_rdd = self.sc.parallelize(x, 4)
        y_rdd = self.sc.parallelize(y, 4)
        z_rdd = self.sc.parallelize(z, 4)

        expected = (np.arange(20).reshape(10, 2), tuple(range(10)),
                    list(range(10)))

        rdd = DictRDD([x_rdd, y_rdd, z_rdd])
        assert_tuple_equal(rdd.first(), expected)
        assert_equal(rdd.dtype, (np.ndarray, tuple, tuple))
        assert_true(check_rdd_dtype(rdd, {0: np.ndarray, 1: tuple, 2: tuple}))

        rdd = DictRDD([x_rdd, y_rdd, z_rdd], columns=('x', 'y', 'z'))
        assert_tuple_equal(rdd.first(), expected)
        assert_equal(rdd.dtype, (np.ndarray, tuple, tuple))
        assert_true(check_rdd_dtype(rdd, {'x': np.ndarray, 'y': tuple,
                                          'z': tuple}))

    def test_get_single_tuple(self):
        x, y = np.arange(80).reshape((40, 2)), np.arange(40)
        x_rdd = self.sc.parallelize(x, 2)
        y_rdd = self.sc.parallelize(y, 2)
        z_rdd = x_rdd.zip(y_rdd)
        z = DictRDD(z_rdd, bsize=5)

        expected = np.arange(0, 10).reshape((5, 2)), np.arange(5)
        for tpl in [z.first(), z[0].first(), z[0].first()]:
            assert_tuple_equal(tpl, expected)

        expected = np.arange(30, 40).reshape((5, 2)), np.arange(15, 20)
        for tpl in [z[3].first(), z[3].first(), z[-5].first()]:
            assert_tuple_equal(tpl, expected)

        expected = np.arange(70, 80).reshape((5, 2)), np.arange(35, 40)
        for tpl in [z[7].first(), z[7].first(), z[-1].first()]:
            assert_tuple_equal(tpl, expected)

    def test_get_single_item(self):
        x, y = np.arange(80).reshape((40, 2)), np.arange(40)
        x_rdd = self.sc.parallelize(x, 2)
        y_rdd = self.sc.parallelize(y, 2)
        z_rdd = x_rdd.zip(y_rdd)
        z = DictRDD(z_rdd, bsize=5)

        assert_array_equal(z[0, 0].first(), np.arange(0, 10).reshape((5, 2)))
        assert_array_equal(z[0, 1].first(), np.arange(5))

        assert_array_equal(z[3, 0].first(), np.arange(30, 40).reshape((5, 2)))
        assert_array_equal(z[3, 1].first(), np.arange(15, 20))
        # assert_array_equal(z[3, -1].first(), np.arange(15, 20))

        assert_array_equal(z[7, 0].first(), np.arange(70, 80).reshape((5, 2)))
        assert_array_equal(z[-1, 0].first(), np.arange(70, 80).reshape((5, 2)))
        assert_array_equal(z[7, 1].first(), np.arange(35, 40))
        # assert_array_equal(z[-1, -1].first(), np.arange(35, 40))

    def test_get_multiple_tuples(self):
        x, y = np.arange(80).reshape((40, 2)), np.arange(40)
        x_rdd = self.sc.parallelize(x, 2)
        y_rdd = self.sc.parallelize(y, 2)
        z_rdd = x_rdd.zip(y_rdd)
        z = DictRDD(z_rdd, bsize=5)

        expected = [(np.arange(0, 10).reshape((5, 2)), np.arange(0, 5)),
                    (np.arange(10, 20).reshape((5, 2)), np.arange(5, 10))]
        assert_multiple_tuples_equal(z[:2].collect(), expected)
        assert_multiple_tuples_equal(z[:2, :].collect(), expected)
        assert_multiple_tuples_equal(z[[0, 1]].collect(), expected)
        assert_multiple_tuples_equal(z[[0, 1], :].collect(), expected)
        assert_multiple_tuples_equal(z[[1, 0]].collect(), expected[::-1])

        expected = [(np.arange(50, 60).reshape((5, 2)), np.arange(25, 30)),
                    (np.arange(60, 70).reshape((5, 2)), np.arange(30, 35)),
                    (np.arange(70, 80).reshape((5, 2)), np.arange(35, 40))]
        assert_multiple_tuples_equal(z[-3:].collect(), expected)
        assert_multiple_tuples_equal(z[-3:, :].collect(), expected)
        assert_multiple_tuples_equal(z[[5, 6, 7]].collect(), expected)
        assert_multiple_tuples_equal(z[[5, 6, 7], :].collect(), expected)
        assert_multiple_tuples_equal(z[[7, 6, 5]].collect(), expected[::-1])
        assert_multiple_tuples_equal(z[[7, 6, 5], :].collect(), expected[::-1])
        assert_multiple_tuples_equal(z[[5, 7, 6]].collect(),
                                     [expected[0], expected[2], expected[1]])

    def test_get_multiple_items(self):
        x, y = np.arange(80).reshape((40, 2)), np.arange(40)
        x_rdd = self.sc.parallelize(x, 2)
        y_rdd = self.sc.parallelize(y, 2)
        z_rdd = x_rdd.zip(y_rdd)
        z = DictRDD(z_rdd, bsize=5)

        expected = [(np.arange(0, 10).reshape((5, 2)), np.arange(0, 5)),
                    (np.arange(10, 20).reshape((5, 2)), np.arange(5, 10))]
        assert_array_equal(z[:2, 1].collect(),
                           [expected[0][1], expected[1][1]])
        assert_array_equal(z[[0, 1], 0].collect(),
                           [expected[0][0], expected[1][0]])
        assert_multiple_tuples_equal(z[[0, 1], [1]].collect(),
                                     [(expected[0][1],),
                                      (expected[1][1],)])
        assert_multiple_tuples_equal(z[[0, 1], -1:].collect(),
                                     [(expected[0][1],),
                                      (expected[1][1],)])
        assert_multiple_tuples_equal(z[[1, 0], [1, 0]].collect(),
                                     [expected[1][::-1], expected[0][::-1]])

    def test_transform(self):
        data1 = np.arange(400).reshape((100, 4))
        data2 = np.arange(200).reshape((100, 2))
        rdd1 = self.sc.parallelize(data1, 4)
        rdd2 = self.sc.parallelize(data2, 4)

        X = DictRDD(rdd1.zip(rdd2), bsize=5)

        X1 = [(x[0], x[1] ** 2) for x in X.collect()]
        X2 = X.transform(lambda a, b: (a, b ** 2))
        assert_multiple_tuples_equal(X1, X2.collect())

        X1 = [(x[0], x[1] ** 2) for x in X.collect()]
        X2 = X.transform(lambda x: x ** 2, column=1)
        assert_multiple_tuples_equal(X1, X2.collect())

        X1 = [(x[0] ** 2, x[1]) for x in X.collect()]
        X2 = X.transform(lambda x: x ** 2, column=0)
        assert_multiple_tuples_equal(X1, X2.collect())

        X1 = [(x[0] ** 2, x[1] ** 0.5) for x in X.collect()]
        X2 = X.transform(lambda a, b: (a ** 2, b ** 0.5), column=[0, 1])
        assert_multiple_tuples_equal(X1, X2.collect())

        X1 = [(x[0] ** 2, x[1] ** 0.5) for x in X.collect()]
        X2 = X.transform(lambda b, a: (b ** 0.5, a ** 2), column=[1, 0])
        assert_multiple_tuples_equal(X1, X2.collect())

    def test_transform_with_dtype(self):
        data1 = np.arange(400).reshape((100, 4))
        data2 = np.arange(200).reshape((100, 2))
        rdd1 = self.sc.parallelize(data1, 4)
        rdd2 = self.sc.parallelize(data2, 4)

        X = DictRDD(rdd1.zip(rdd2), bsize=5)

        X2 = X.transform(lambda x: x ** 2, column=0)
        assert_equal(X2.dtype, (np.ndarray, np.ndarray))

        X2 = X.transform(lambda x: tuple((x ** 2).tolist()), column=0,
                         dtype=tuple)
        assert_equal(X2.dtype, (tuple, np.ndarray))
        assert_true(check_rdd_dtype(X2, {0: tuple, 1: np.ndarray}))

        X2 = X.transform(lambda x: x ** 2, column=1, dtype=list)
        assert_equal(X2.dtype, (np.ndarray, list))
        assert_true(check_rdd_dtype(X2, {0: np.ndarray, 1: list}))

        X2 = X.transform(lambda a, b: (a ** 2, (b ** 0.5).tolist()),
                         column=[0, 1], dtype=(np.ndarray, list))
        assert_true(check_rdd_dtype(X2, {0: np.ndarray, 1: list}))

        X2 = X.transform(lambda b, a: ((b ** 0.5).tolist(), a ** 2),
                         column=[1, 0], dtype=(list, np.ndarray))
        assert_equal(X2.dtype, (np.ndarray, list))
        assert_true(check_rdd_dtype(X2, {0: np.ndarray, 1: list}))
