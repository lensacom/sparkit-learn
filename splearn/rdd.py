# -*- coding: utf-8 -*-

import functools

import numpy as np
import scipy.sparse as sp
from pyspark import RDD
from sklearn.externals import six


def _auto_dtype(elem):
    if sp.issparse(elem):
        return sp.spmatrix
    elif isinstance(elem, np.ndarray):
        return np.ndarray
    else:
        return tuple


def _check_dtype(block, dtype):
    if not isinstance(block, dtype):
        raise ValueError("Type mismatch. Expected {0}, {1} given.".format(
            dtype, type(block)))


def _pack_accumulated(accumulated, dtype):
    if dtype is np.ndarray:
        return np.array(accumulated)
    elif dtype is sp.spmatrix:
        return sp.vstack(accumulated)
    else:
        return dtype(accumulated)


def _block_collection(iterator, dtype, bsize=-1):
    """Pack rdd with a specific collection constructor."""
    i = 0
    accumulated = []
    for a in iterator:
        if (bsize > 0) and (i >= bsize):
            yield _pack_accumulated(accumulated, dtype)
            accumulated = []
            i = 0
        accumulated.append(a)
        i += 1
    if i > 0:
        yield _pack_accumulated(accumulated, dtype)


def _block_tuple(iterator, dtypes, bsize=-1):
    """Pack rdd of tuples as tuples of arrays or scipy.sparse matrices."""
    i = 0
    blocked_tuple = None
    for tuple_i in iterator:
        if blocked_tuple is None:
            blocked_tuple = tuple([] for _ in range(len(tuple_i)))

        if (bsize > 0) and (i >= bsize):
            yield tuple(_pack_accumulated(x, dtype)
                        for x, dtype in zip(blocked_tuple, dtypes))
            blocked_tuple = tuple([] for _ in range(len(tuple_i)))
            i = 0

        for x_j, x in zip(tuple_i, blocked_tuple):
            x.append(x_j)
        i += 1
    if i > 0:
        yield tuple(_pack_accumulated(x, dtype)
                    for x, dtype in zip(blocked_tuple, dtypes))


def block(rdd, bsize=-1, dtype=None):
    """Block an RDD

    Parameters
    ----------

    rdd : RDD
        RDD of data points to block into either numpy arrays,
        scipy sparse matrices, or pandas data frames.
        Type of data point will be automatically inferred
        and blocked accordingly.

    bsize : int, optional, default None
        Size of each block (number of elements), if None all data points
        from each partition will be combined in a block.

    Returns
    -------

    rdd : ArrayRDD or TupleRDD or DictRDD
        The transformed rdd with added functionality
    """
    try:
        entry = rdd.first()
    except IndexError:
        # empty RDD: do not block
        return rdd

    # do different kinds of block depending on the type
    if isinstance(entry, dict):
        rdd = rdd.map(lambda x: list(x.values()))
        return DictRDD(rdd, list(entry.keys()), bsize, dtype)
    elif isinstance(entry, tuple):
        return DictRDD(rdd, bsize=bsize, dtype=dtype)
    elif sp.issparse(entry):
        return SparseRDD(rdd, bsize)
    elif isinstance(entry, np.ndarray):
        return ArrayRDD(rdd, bsize)
    else:
        return BlockRDD(rdd, bsize, dtype)


class BlockRDD(object):

    def __init__(self, rdd, bsize=-1, dtype=tuple, noblock=False):
        if isinstance(rdd, BlockRDD):
            _check_dtype(rdd.first(), dtype)
            self._rdd = rdd._rdd
            bsize = rdd.bsize
        elif isinstance(rdd, RDD):
            if not noblock:
                rdd = self._block(rdd, bsize, dtype)
            self._rdd = rdd
        else:
            raise TypeError(
                "Unexpected type {0} for parameter rdd".format(type(rdd)))

        self.dtype = dtype
        self.bsize = bsize

    def _block(self, rdd, bsize, dtype):
        """Execute the blocking process on the given rdd.

        Parameters
        ----------
        rdd : pyspark.rdd.RDD
            Distributed data to block
        bsize : int or None
            The desired size of the blocks

        Returns
        -------
        rdd : pyspark.rdd.RDD
            Blocked rdd.
        """
        return rdd.mapPartitions(
            lambda x: _block_collection(x, dtype, bsize))

    def get_params(self):
        return {'bsize': self.bsize,
                'dtype': self.dtype}

    def __repr__(self):
        """Returns a string representation of the ArrayRDD.
        """
        return "{0} from {1}".format(self.__class__, repr(self._rdd))

    def __getattr__(self, attr):
        """Access pyspark.rdd.RDD methods or attributes.

        Parameters
        ----------
        attr : method or attribute
            The method/attribute to access

        Returns
        -------
        result : method or attribute
            The result of the requested method or attribute casted to ArrayRDD
            if method/attribute is available, else raises AttributeError.
        """
        def bypass(*args, **kwargs):
            result = getattr(self._rdd, attr)(*args, **kwargs)
            if isinstance(result, RDD):
                if result is self._rdd:
                    return self
                else:
                    return BlockRDD(result, bsize=self.bsize, dtype=self.dtype,
                                    noblock=True)
            return result

        if not hasattr(self._rdd, attr):
            raise AttributeError("{0} object has no attribute {1}".format(
                                 self.__class__, attr))
        return bypass

    def __getitem__(self, index):
        """Returns the selected blocks defined by index parameter.

        Parameter
        ---------
        index : int or slice
            The key of the block or the range of the blocks.

        Returns
        -------
        block : ArrayRDD
            The selected block(s).
        """
        if isinstance(index, tuple):
            raise IndexError("Too many indices for BlockRDD")
        elif isinstance(index, slice) and index == slice(None, None, None):
            return self

        indices = np.arange(self.blocks)[index]
        indexed = self._rdd.zipWithIndex()
        if isinstance(index, slice):
            ascending = index.step is None or index.step > 0
            rdd = indexed.filter(lambda x_i: x_i[1] in indices)
            if not ascending:
                rdd = rdd.sortBy(lambda x_i: x_i[1], ascending)
        elif hasattr(index, "__iter__"):
            # TODO: check monotoniticity to avoid unnunnecessary sorting
            arg = indices.tolist()
            rdd = indexed.filter(lambda x_i: x_i[1] in indices) \
                         .sortBy(lambda x_i: arg.index(x_i[1]))
        elif isinstance(index, int):
            rdd = indexed.filter(lambda x_i: x_i[1] == indices)
        else:
            raise KeyError("Unexpected type of index: {0}".format(type(index)))

        rdd = rdd.map(lambda x_i: x_i[0])
        return self.__class__(rdd, noblock=True, **self.get_params())

    def __len__(self):
        """Returns the number of elements in all blocks."""
        return self._rdd.map(len).sum()

    @property
    def context(self):
        return self._rdd.ctx

    @property
    def blocks(self):  # numblocks?
        """Returns the number of blocks."""
        return self._rdd.count()

    @property
    def partitions(self):  # numpart?
        """Returns the number of partitions in the rdd."""
        return self._rdd.getNumPartitions()

    def unblock(self):
        """Flattens the blocks. Returns as list."""
        return self._rdd.flatMap(lambda x: list(x))

    def tolist(self):
        """Flattens the blocks. Returns as list."""
        return self.unblock().collect()

    def toarray(self):
        """Returns the data as numpy.array from each partition."""
        rdd = self._rdd.map(lambda x: np.array(x))
        return np.concatenate(rdd.collect())

    def transform(self, fn, dtype=None, *args, **kwargs):
        """Equivalent to map, compatibility purpose only.
        Column parameter ignored.
        """
        rdd = self._rdd.map(fn)

        if dtype is None:
            return self.__class__(rdd, noblock=True, **self.get_params())
        if dtype is np.ndarray:
            return ArrayRDD(rdd, bsize=self.bsize, noblock=True)
        elif dtype is sp.spmatrix:
            return SparseRDD(rdd, bsize=self.bsize, noblock=True)
        else:
            return BlockRDD(rdd, bsize=self.bsize, dtype=dtype, noblock=True)

    # def cartesian(self, other):
    #     return TupleRDD(self._rdd.cartesian(other._rdd), False)


class ArrayLikeRDDMixin(object):

    def __getitem__(self, key):
        """Access a specified block.

        Parameters
        ----------
        key : int or slice
            The key of the block or the range of the blocks.

        Returns
        -------
        block : ArrayRDD
            The selected block(s).
        """
        if isinstance(key, tuple):
            index, mask = key[0], key[1:]
            return super(ArrayRDD, self).__getitem__(index) \
                                        .map(lambda x: x[mask])
        else:
            return super(ArrayRDD, self).__getitem__(key)

    @property
    def ndim(self):
        return self._rdd.first().ndim

    @property
    def shape(self):
        """Returns the shape of the data."""
        # TODO cache
        first = self.first().shape
        shape = self._rdd.map(lambda x: x.shape[0]).sum()
        return (shape,) + first[1:]

    @property
    def size(self):
        """Returns the shape of the data.
        """
        return np.prod(self.shape)

    def sum(self, axis=None):
        if axis in (None, 0):
            return self._rdd.map(lambda x: x.sum(axis=axis)).sum()
        else:
            return self._rdd.map(lambda x: x.sum(axis=axis)) \
                            .reduce(lambda a, b: np.concatenate([a, b]))

    def mean(self, axis=None):
        def mapper(x):
            """Calculate statistics for every numpy or scipy blocks."""
            cnt = np.prod(x.shape) if axis is None else x.shape[axis]
            return cnt, x.mean(axis=axis)

        def reducer(a, b):
            """Calculate the combined statistics."""
            n_a, mean_a = a
            n_b, mean_b = b
            n_ab = n_a + n_b
            if axis in (None, 0):
                mean_ab = ((mean_a * n_a) + (mean_b * n_b)) / n_ab
            else:
                mean_ab = np.concatenate([mean_a, mean_b])
            return n_ab, mean_ab

        n, mean = self._rdd.map(mapper).treeReduce(reducer)
        return mean


class ArrayRDD(BlockRDD, ArrayLikeRDDMixin):

    """A distributed array data structure.

    Stores distributed numpy.arrays. It provides a transparent interface to the
    underlying pyspark.rdd.RDD and also extends it with numpy.array like
    methods.

    Parameters
    ----------
    rdd : pyspark.rdd.RDD
        A parallelized data container
    bsize : {int, None, False} default to None
        The number of entries to block together. If None one block will be
        created in every partition. If False, no blocking executed. Useful when
        casting already blocked rdds.

    Attributes
    ----------
    partitions : int
        The number of partitions in the rdd
    blocks : int
        The number of blocks present in the rdd
    shape : tuple
        The dimensionality of the dataset
    _rdd : pyspark.rdd.RDD
        The underlying distributed data container

    Examples
    --------
    >>> from splearn.rdd import ArrayRDD
    >>> rdd = sc.parallelize(range(20), 2)
    >>> X = ArrayRDD(rdd, bsize=5)
    >>> X
    <class 'splearn.rdd.ArrayRDD'> from PythonRDD...

    >>> X.collect()
    [array([0, 1, 2, 3, 4]),
     array([5, 6, 7, 8, 9]),
     array([10, 11, 12, 13, 14]),
     array([15, 16, 17, 18, 19])]

    >>> X[:-1:2].collect()
    [array([0, 1, 2, 3, 4]), array([10, 11, 12, 13, 14])]

    """

    def __init__(self, rdd, bsize=-1, dtype=np.ndarray, noblock=False):
        if dtype is not np.ndarray:
            raise ValueError("Only supported type for ArrayRDD is np.ndarray!")
        super(ArrayRDD, self).__init__(rdd, bsize, dtype, noblock)

    def __mul__(self, other):
        return self.multiply(other)

    def __add__(self, other):
        return self.add(other)

    def __sub__(self, other):
        return self.subtract(other)

    def __div__(self, other):
        return self.divide(other)

    __truediv__ = __div__

    def __pow__(self, other):
        return self.power(other)

    def __floordiv__(self, other):
        return self.floor_divide(other)

    def __mod__(self, other):
        return self.mod(other)

    def tosparse(self):
        return SparseRDD(self._rdd.map(lambda x: sp.csr_matrix(x)))

    def flatten(self):
        return self.map(lambda x: x.flatten())

    def min(self, axis=None):
        return self._on_axis('min', axis)

    def max(self, axis=None):
        return self._on_axis('max', axis)

    def prod(self, axis=None):
        return self._on_axis('prod', axis)

    def dot(self, other):
        return self._on_other('dot', other)

    def add(self, other):
        return self._on_other('add', other)

    def subtract(self, other):
        return self._on_other('subtract', other)

    def multiply(self, other):
        return self._on_other('multiply', other)

    def divide(self, other):
        return self._on_other('divide', other)

    def power(self, other):
        return self._on_other('power', other)

    def floor_divide(self, other):
        return self._on_other('floor_divide', other)

    def true_divide(self, other):
        return self._on_other('true_divide', other)

    def mod(self, other):
        return self._on_other('mod', other)

    def fmod(self, other):
        return self._on_other('fmod', other)

    def remainder(self, other):
        return self._on_other('remainder', other)

    def _on_other(self, func, other):
        rdd = self._rdd.map(lambda x: getattr(np, func)(x, other))
        return ArrayRDD(rdd, noblock=True)

    def _on_axis(self, func, axis=None):
        rdd = self._rdd.map(lambda x: getattr(x, func)(axis=axis))

        if axis is None:
            return getattr(np.array(rdd.collect()), func)()
        elif axis == 0:
            return rdd.reduce(
                lambda a, b: getattr(np.array((a, b)), func)(axis=0))
        else:
            return rdd.reduce(lambda a, b: np.concatenate((a, b)))


class SparseRDD(BlockRDD, ArrayLikeRDDMixin):

    def __init__(self, rdd, bsize=-1, dtype=sp.spmatrix, noblock=False):
        if dtype is not sp.spmatrix:
            raise ValueError("Only supported type for SparseRDD is"
                             " sp.spmatrix!")
        super(SparseRDD, self).__init__(rdd, bsize, dtype, noblock)

    def toarray(self):
        """Returns the data as numpy.array from each partition."""
        rdd = self._rdd.map(lambda x: x.toarray())
        return np.concatenate(rdd.collect())

    def todense(self):
        rdd = self._rdd.map(lambda x: x.toarray())
        return ArrayRDD(rdd, noblock=True)

    def dot(self, other):
        rdd = self._rdd.map(lambda x: x.dot(other))
        if sp.issparse(other):
            return SparseRDD(rdd, bsize=self.bsize, noblock=True)
        else:
            return ArrayRDD(rdd, bsize=self.bsize, noblock=True)

    def min(self, axis=None):
        return self._on_axis('min', axis)

    def max(self, axis=None):
        return self._on_axis('max', axis)

    def _on_axis(self, func, axis=None):
        rdd = self._rdd.map(lambda x: getattr(x, func)(axis=axis))

        if axis is None:
            return getattr(np.array(rdd.collect()), func)()
        elif axis == 0:
            return rdd.reduce(
                lambda a, b: getattr(sp.vstack((a, b)), func)(axis=0))
        else:
            return rdd.reduce(lambda a, b: sp.vstack((a, b)))


class DictRDD(BlockRDD):

    """Distributed named tuple data structure.

    The tuple is stored as a tuple of numpy.arrays in each block. It works like
    a column based data structure, each column can be transformed and accessed
    independently and identified by a column name. Very useful for training:
    In splearn the fit methods expects a DictRDD with 'X', 'y' and an optional
    'w' columns.
        'X' - training data,
        'y' - labels,
        'w' - weights


    Parameters
    ----------
    rdd : pyspark.rdd.RDD
        A parallelized data container
    bsize : {int, None, False} default to None
        The number of entries to block together. If None, one block will be
        created in every partition. If False, no blocking executed. Useful when
        casting already blocked rdds.

    Attributes
    ----------
    columns : tuple
        Name of columns in the rdd as a tuple
    partitions : int
        The number of partitions in the rdd
    blocks : int
        The number of blocks present in the rdd
    shape : tuple
        The dimensionality of the dataset
    _rdd : pyspark.rdd.RDD
        The underlying distributed data container

    Examples
    --------
    >>> import numpy as np
    >>> from splearn.rdd import DictRDD
    >>> data = np.array([range(20), range(2)*10])
    >>> Z = DictRDD(sc.parallelize(data.T), columns=('X', 'y'), bsize=5)
    >>> Z
    <class 'splearn.rdd.DictRDD'> from PythonRDD...

    >>> Z.collect()
    [(array([0, 1, 2, 3, 4]), array([0, 1, 0, 1, 0])),
     (array([5, 6, 7, 8, 9]), array([1, 0, 1, 0, 1])),
     (array([10, 11, 12, 13, 14]), array([0, 1, 0, 1, 0])),
     (array([15, 16, 17, 18, 19]), array([1, 0, 1, 0, 1]))]

    >>> Z.columns
    ('X', 'y')

    >>> Z[1:-1, 'X'].collect()
    [array([5, 6, 7, 8, 9]), array([10, 11, 12, 13, 14])]
    """

    def __init__(self, rdd, columns=None, bsize=-1, dtype=None,
                 noblock=False):
        if hasattr(rdd, '__iter__'):
            def zipper(a_b):
                (a, b) = a_b
                if not isinstance(a, tuple):
                    a = (a,)
                return a + (b,)

            if columns is None:
                columns = list(range(len(rdd)))
            elif len(rdd) != len(columns):
                raise ValueError("The number of RDDs must be equal to columns!")
            if all(isinstance(r, BlockRDD) for r in rdd):
                if any(r.bsize != rdd[0].bsize for r in rdd):
                    raise ValueError("All RDD must have the same block size!")
                dtype = [r.dtype for r in rdd]
                rdd = [r._rdd for r in rdd]
                noblock = True
            elif not all(isinstance(r, RDD) for r in rdd):
                raise TypeError("All element must be an instance of RDD or"
                                " BlockRDD!")
            rdd = functools.reduce(lambda t, x: t.zip(x).map(zipper), rdd) \
                           .persist()
        elif isinstance(rdd, (RDD, BlockRDD)):
            if columns is None:
                columns = list(range(len(rdd.first())))
        else:
            raise TypeError("Rdd Must be an instance of RDD or BlockRDD!")

        if not len(columns) == len(set(columns)):
            raise ValueError("Column names must be unique!")
        elif not hasattr(columns, "__iter__"):
            raise ValueError("Columns parameter must be iterable!")

        if dtype is None:
            dtype = [_auto_dtype(elem) for elem in rdd.first()]
        elif len(columns) != len(dtype):
            raise ValueError("Columns and dtype lengths must be equal!")

        self.columns = tuple(columns)
        super(DictRDD, self).__init__(rdd, bsize, tuple(dtype), noblock)

    def _block(self, rdd, bsize, dtype):
        """Execute the blocking process on the given rdd.

        Parameters
        ----------
        rdd : pyspark.rdd.RDD
            Distributed data to block
        bsize : int or None
            The desired size of the blocks

        Returns
        -------
        rdd : pyspark.rdd.RDD
            Blocked rdd.
        """
        return rdd.mapPartitions(lambda x: _block_tuple(x, dtype, bsize))

    def get_params(self):
        return {'bsize': self.bsize,
                'dtype': self.dtype,
                'columns': self.columns}

    def ix(self, key):
        """Returns the selected blocks defined by index parameter.
        Parameter
        ---------
        index : int or slice
            The key of the block or the range of the blocks.
        Returns
        -------
        block : DictRDD
            The selected block(s).
        """
        return super(DictRDD, self).__getitem__(key)

    def get(self, key):
        if isinstance(key, tuple):
            raise IndexError("Too many indices for DictRDD")
        elif isinstance(key, slice):
            if key == slice(None, None, None):
                return self
            rdd = self._rdd.map(lambda x: x[key])
            return DictRDD(rdd, bsize=self.bsize, columns=self.dtype[key],
                           dtype=self.dtype[key], noblock=True)
        elif hasattr(key, "__iter__") and not isinstance(key, six.string_types):
            if tuple(key) == self.columns:
                return self
            indices = [self.columns.index(k) for k in key]
            dtype = [self.dtype[i] for i in indices]
            rdd = self._rdd.map(lambda x: tuple(x[i] for i in indices))
            return DictRDD(rdd, bsize=self.bsize, columns=key, dtype=dtype,
                           noblock=True)
        else:
            index = self.columns.index(key)
            dtype = self.dtype[index]
            bsize = self.bsize
            rdd = self._rdd.map(lambda x: x[index])
            if dtype is np.ndarray:
                return ArrayRDD(rdd, bsize=bsize, noblock=True)
            elif dtype is sp.spmatrix:
                return SparseRDD(rdd, bsize=bsize, noblock=True)
            else:
                return BlockRDD(rdd, bsize=bsize, dtype=dtype, noblock=True)

    def __getitem__(self, key):
        """Access a specified block.

        Parameters
        ----------
        key : int or slice
            The key of the block or the range of the blocks.

        Returns
        -------
        block : ArrayRDD or TupleRDD
            The selected block(s).
        """
        if isinstance(key, tuple):  # get first index
            index, key = key
            return super(DictRDD, self).__getitem__(index).get(key)
        else:
            return super(DictRDD, self).__getitem__(key)

    def __contains__(self, key):
        return key in self.columns

    def unblock(self):
        """Flattens the blocks.
        """
        return self._rdd.flatMap(lambda cols: list(zip(*cols)))

    def transform(self, fn, column=None, dtype=None):
        """Execute a transformation on a column or columns. Returns the modified
        DictRDD.

        Parameters
        ----------
        f : function
            The function to execute on the columns.
        column : {str, list or None}
            The column(s) to transform. If None is specified the method is
            equivalent to map.
        column : {str, list or None}
            The dtype of the column(s) to transform.

        Returns
        -------
        result : DictRDD
            DictRDD with transformed column(s).

        TODO: optimize
        """
        dtypes = self.dtype
        if column is None:
            indices = list(range(len(self.columns)))
        else:
            if not type(column) in (list, tuple):
                column = [column]
            indices = [self.columns.index(c) for c in column]

        if dtype is not None:
            if not type(dtype) in (list, tuple):
                dtype = [dtype]
            dtypes = [dtype[indices.index(i)] if i in indices else t
                      for i, t in enumerate(self.dtype)]

        def mapper(values):
            result = fn(*[values[i] for i in indices])

            if len(indices) == 1:
                result = (result,)
            elif not isinstance(result, (tuple, list)):
                raise ValueError("Transformer function must return an"
                                 " iterable!")
            elif len(result) != len(indices):
                raise ValueError("Transformer result's length must be"
                                 " equal to the given columns length!")

            return tuple(result[indices.index(i)] if i in indices else v
                         for i, v in enumerate(values))

        return DictRDD(self._rdd.map(mapper),
                       columns=self.columns, dtype=dtypes,
                       bsize=self.bsize, noblock=True)
