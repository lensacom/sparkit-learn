# -*- coding: utf-8 -*-

from pyspark.rdd import RDD

import numpy as np
import pandas as pd
import scipy.sparse as sp
import operator


def _pack_accumulated(accumulated):
    if len(accumulated) > 0 and sp.issparse(accumulated[0]):
        return sp.vstack(accumulated)
    else:
        return np.array(accumulated)


def _block_tuple(iterator, block_size=None):
    """Pack rdd of tuples as tuples of arrays or scipy.sparse matrices."""
    i = 0
    blocked_tuple = None
    for tuple_i in iterator:
        if blocked_tuple is None:
            blocked_tuple = tuple([] for _ in range(len(tuple_i)))

        if block_size is not None and i >= block_size:
            yield tuple(_pack_accumulated(x) for x in blocked_tuple)
            blocked_tuple = tuple([] for _ in range(len(tuple_i)))
            i = 0
        for x_j, x in zip(tuple_i, blocked_tuple):
            x.append(x_j)
        i += 1
    yield tuple(_pack_accumulated(x) for x in blocked_tuple)


def _block_collection(iterator, collection_type, block_size=None):
    """Pack rdd with a specific collection constructor."""
    i = 0
    accumulated = []
    for a in iterator:
        if block_size is not None and i >= block_size:
            yield collection_type(accumulated)
            accumulated = []
            i = 0
        accumulated.append(a)
        i += 1
    yield collection_type(accumulated)


def block(rdd, block_size=None):
    """Block an RDD

    Parameters
    ----------

    rdd : RDD
        RDD of data points to block into either numpy arrays,
        scipy sparse matrices, or pandas data frames.
        Type of data point will be automatically inferred
        and blocked accordingly.

    block_size : int, optional, default None
        Size of each block (number of elements), if None all data points
        from each partition will be combined in a block.

    """
    try:
        entry = rdd.first()
    except IndexError:
        # empty RDD: do not block
        return rdd

    # do different kinds of block depending on the type
    if isinstance(entry, tuple):
        return TupleRDD(rdd, block_size)
    elif isinstance(entry, dict):
        return DataFrameRDD(rdd, block_size)
    elif sp.issparse(entry):
        return MatrixRDD(rdd, block_size)
    else:  # Fallback to array packing
        return ArrayRDD(rdd, block_size)


class ArrayRDD(object):

    def __init__(self, rdd, block_size=None):
        if isinstance(rdd, ArrayRDD):
            self._rdd = rdd._rdd
        elif isinstance(rdd, RDD):
            if block_size is False:
                self._rdd = rdd
            else:
                self._rdd = self._block(rdd, block_size)
        else:
            pass  # raise exception

    def _block(self, rdd, block_size):
        return rdd.mapPartitions(
            lambda x: _block_collection(x, np.array, block_size))

    def __getattr__(self, attr):
        def bypass(*args, **kwargs):
            result = getattr(self._rdd, attr)(*args, **kwargs)
            if isinstance(result, RDD):
                if result is not self._rdd:
                    return self.__class__(result, False)
                else:
                    return self
            return result

        if not hasattr(self._rdd, attr):
            raise AttributeError("{0} object has no attribute {1}".format(
                                 self.__class__, attr))
        return bypass

    def __repr__(self):
        return "{0} from {1}".format(self.__class__, repr(self._rdd))

    def partitions(self):
        return self._rdd.getNumPartitions()

    @property
    def shape(self):
        first = self.first().shape
        shape = self._rdd.map(lambda x: x.shape[0]).reduce(operator.add)
        return (shape,) + first[1:]


class MatrixRDD(ArrayRDD):

    def _block(self, rdd, block_size):
        return rdd.mapPartitions(
            lambda x: _block_collection(x, sp.vstack, block_size))


class DataFrameRDD(ArrayRDD):

    def _block(self, rdd, block_size):
        return rdd.mapPartitions(
            lambda x: _block_collection(x, pd.DataFrame, block_size))


class TupleRDD(ArrayRDD):

    def _block(self, rdd, block_size):
        return rdd.mapPartitions(
            lambda x: _block_tuple(x, block_size))

    def column(self, col):
        # check first element
        return ArrayRDD(self._rdd.map(lambda x: x[col]))
