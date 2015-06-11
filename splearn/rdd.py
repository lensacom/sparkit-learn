# -*- coding: utf-8 -*-

import itertools

import numpy as np
import scipy.sparse as sp
try:
    from pyspark.rdd import RDD
except ImportError:
    raise ImportError("pyspark home needs to be added to PYTHONPATH.\n"
                      "export PYTHONPATH=$PYTHONPATH:$SPARK_HOME/python:../")


def _pack_accumulated(accumulated, dtype=None):
    if dtype is not None:
        return dtype(accumulated)
    elif len(accumulated) > 0 and sp.issparse(accumulated[0]):
        return sp.vstack(accumulated)
    else:
        return np.array(accumulated)


def _unpack_blocks(blocks, dtype=None):
    if dtype is not None:
        dtype(itertools.chain(*blocks))
    elif sp.issparse(blocks[0]):
        return sp.vstack(blocks)
    else:
        return np.concatenate(blocks)


def _block_tuple(iterator, bsize=None, dtypes=None):
    """Pack rdd of tuples as tuples of arrays or scipy.sparse matrices."""
    i = 0
    blocked_tuple = None
    for tuple_i in iterator:
        if blocked_tuple is None:
            blocked_tuple = tuple([] for _ in range(len(tuple_i)))
            dtypes = dtypes or [None] * len(blocked_tuple)

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


def _block_collection(iterator, bsize=None, dtype=None):
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


def block(rdd, bsize=None, dtype=None):
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
        rdd = rdd.map(lambda x: x.values())
        return DictRDD(rdd, entry.keys(), bsize, dtype)
    elif isinstance(entry, tuple):
        return DictRDD(rdd, bsize, dtype)
    else:  # Fallback to array packing
        return ArrayRDD(rdd, bsize, dtype)


class BlockRDD(object):

    def __init__(self, rdd, bsize=-1, dtype=list, noblock=False):
        if isinstance(rdd, BlockRDD):
            self._rdd = rdd._rdd
            dtype = rdd.dtype
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
            lambda x: _block_collection(x, bsize, dtype))

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
            rdd = indexed.filter(lambda (x, i): i in indices)
            if not ascending:
                rdd = rdd.sortBy(lambda (x, i): i, ascending)
        elif hasattr(index, "__iter__"):
            # TODO: check monotoniticity to avoid unnunnecessary sorting
            arg = indices.tolist()
            rdd = indexed.filter(lambda (x, i): i in indices) \
                         .sortBy(lambda (x, i): arg.index(i))
        elif isinstance(index, int):
            rdd = indexed.filter(lambda (x, i): i == indices)
        else:
            raise KeyError("Unexpected type of index: {0}".format(type(index)))

        rdd = rdd.map(lambda (x, i): x)
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

    def transform(self, f, *args, **kwargs):
        """Equivalent to map, compatibility purpose only.
        Column parameter ignored.
        """
        rdd = self._rdd.map(f)
        return self.__class__(rdd, noblock=True, **self.get_params())


class ArrayRDD(BlockRDD):

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

    def __init__(self, rdd, bsize=-1, dtype=None, noblock=False):
        # TODO: accept list of RDDs then zip them
        super(ArrayRDD, self).__init__(rdd, bsize, None, noblock)

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
    def shape(self):
        """Returns the shape of the data."""
        # TODO cache
        first = self.first().shape
        shape = self._rdd.map(lambda x: x.shape[0]).sum()
        return (shape,) + first[1:]

    def toarray(self):
        """Returns the data as numpy.array from each partition."""
        return np.concatenate(self.collect())

    def sum(self, axis=None):
        if axis in (None, 0):
            return self._rdd.map(lambda x: x.sum(axis=axis)).sum()
        else:
            return self._rdd.map(lambda x: x.sum(axis=axis)) \
                            .reduce(lambda a, b: _unpack_blocks([a, b]))

    def dot(self, other):
        # TODO naive dot implementation with another ArrayRDD
        return self.map(lambda x: x.dot(other))

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
                mean_ab = _unpack_blocks([mean_a, mean_b])
            return n_ab, mean_ab

        n, mean = self._rdd.map(mapper).treeReduce(reducer)
        return mean

    # def tosparse(self):
    #     return sp.vstack(self.collect())

    # def cartesian(self, other):
    #     return TupleRDD(self._rdd.cartesian(other._rdd), False)


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
            def zipper((a, b)):
                if not isinstance(a, tuple):
                    a = (a,)
                return a + (b,)

            if columns is None:
                columns = range(len(rdd))
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
            rdd = reduce(lambda t, x: t.zip(x).map(zipper), rdd).persist()
        elif isinstance(rdd, (RDD, BlockRDD)):
            if columns is None:
                columns = range(len(rdd.first()))
        else:
            raise TypeError("Rdd Must be an instance of RDD or BlockRDD!")

        if not len(columns) == len(set(columns)):
            raise ValueError("Column names must be unique!")
        elif not hasattr(columns, "__iter__"):
            raise ValueError("Columns parameter must be iterable!")

        if dtype is None:
            dtype = [None] * len(columns)
        elif len(columns) != len(dtype):
            raise ValueError("Columns and dtype lengths must be equal!")

        self.columns = tuple(columns)
        super(DictRDD, self).__init__(rdd, bsize, dtype, noblock)

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
        return rdd.mapPartitions(lambda x: _block_tuple(x, bsize, dtype))

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
        elif hasattr(key, "__iter__"):
            if tuple(key) == self.columns:
                return self
            indices = [self.columns.index(k) for k in key]
            dtype = [self.dtype[i] for i in indices]
            rdd = self._rdd.map(lambda x: tuple(x[i] for i in indices))
            return DictRDD(rdd, bsize=self.bsize, columns=key, dtype=dtype,
                           noblock=True)
        else:
            index = self.columns.index(key)
            rdd = self._rdd.map(lambda x: x[index])
            if self.dtype[index] is None:
                return ArrayRDD(rdd, bsize=self.bsize, noblock=True)
            else:
                return BlockRDD(rdd, bsize=self.bsize, dtype=self.dtype[index],
                                noblock=True)

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
        return self._rdd.flatMap(lambda cols: zip(*cols))

    @property
    def shape(self):
        """Returns the shape of the data.
        """
        return (super(DictRDD, self).get(0).shape[0], self.columns)

    def transform(self, fn, column=None):
        """Execute a transformation on a column or columns. Returns the modified
        DictRDD.

        Parameters
        ----------
        f : function
            The function to execute on the columns.
        column : {str, list or None}
            The column(s) to transform. If None is specified the method is
            equivalent to map.

        Returns
        -------
        result : DictRDD
            DictRDD with transformed column(s).

        TODO: optimize
        """
        if column is None:
            indices = range(len(self.columns))
        else:
            if not hasattr(column, '__iter__'):
                column = [column]
            indices = [self.columns.index(c) for c in column]

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

        return self.__class__(self._rdd.map(mapper), noblock=True,
                              **self.get_params())
