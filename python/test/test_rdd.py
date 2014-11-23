import shutil
import tempfile
import numpy as np
import scipy.sparse as sp

from nose.tools import assert_true
from nose.tools import assert_raises
from nose.tools import assert_equal

from numpy.testing import assert_array_less

from sklearn.utils.testing import assert_array_almost_equal

from common import SplearnTestCase
from splearn.rdd import block
from splearn.rdd import ArrayRDD, TupleRDD, DictRDD


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
        assert_equal(sum(len(b[0]) for b in tuple_blocks),  n_samples)
        assert_equal(sum(len(b[1]) for b in tuple_blocks),  n_samples)

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
        assert_equal(sum(len(b) for b in blocks),  n_samples)

        n_partitions = 17
        data = self.sc.parallelize([np.array([1]) for i in range(n_samples)],
                                   n_partitions)
        blocked_data = block(data)
        assert_array_almost_equal(np.ones((n_samples / n_partitions, 1)),
                                  blocked_data.first())
        blocks = blocked_data.collect()
        assert_equal(len(blocks), n_partitions)
        assert_equal(sum(len(b) for b in blocks),  n_samples)

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
        blocks = block(empty_data).collect()
        assert_equal(len(blocks), 0)

    # def test_block_rdd_dict(self):
    #     n_partitions = 3
    #     n_samples = 57
    #     dicts = [{'a': i, 'b': float(i) ** 2} for i in range(n_samples)]
    #     data = self.sc.parallelize(dicts, n_partitions)

    #     block_data_5 = block(data, block_size=5)
    #     blocks = block_data_5.collect()
    #     assert_true(all(len(b) <= 5 for b in blocks))
    #     assert_array_almost_equal(blocks[0][0], np.arange(5))
    #     assert_array_almost_equal(blocks[0][1],
    #                               np.arange(5, dtype=np.float) ** 2)


class TestArrayRDD(RDDTestCase):

    def test_initialization(self):
        n_partitions = 10
        n_samples = 1000

        data = [np.array([1, 2]) for i in range(n_samples)]
        rdd = self.sc.parallelize(data, n_partitions)

        assert_raises(TypeError, ArrayRDD, data)
        assert_raises(TypeError, ArrayRDD, data, False)
        assert_raises(TypeError, ArrayRDD, data, 10)

        assert_true(isinstance(ArrayRDD(rdd), ArrayRDD))
        assert_true(isinstance(ArrayRDD(rdd, 10), ArrayRDD))
        assert_true(isinstance(ArrayRDD(rdd, None), ArrayRDD))

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

    def test_get_single_item(self):
        pass

    def test_get_multiple_item(self):
        pass

    def test_array_slice_syntax(self):
        pass


class TestTupleRDD(RDDTestCase):
    pass


class TestDictRDD(RDDTestCase):
    pass
