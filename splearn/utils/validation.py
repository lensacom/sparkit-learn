import numpy as np
import scipy.sparse as sp
from splearn.rdd import ArrayRDD, SparseRDD, DictRDD


def check_rdd(rdd, expected_dtype):
    """Checks if the blocks in the RDD matches the expected types.

    Parameters:
    -----------
    rdd: splearn.BlockRDD
        The RDD to check
    expected_dtype: {type, list of types, tuple of types, dict of types}
        Expected type(s). If the RDD is a DictRDD the parameter type is
        restricted to dict.

    Returns:
    --------
    accept: bool
        Returns if the types are matched.
    """
    if isinstance(rdd, DictRDD):
        if not isinstance(expected_dtype, dict):
            raise TypeError('Expected {0} for parameter '
                             'expected_dtype, got {1}.' \
                             .format(dict, type(expected_dtype)))
        accept = True
        types = dict(zip(rdd.columns, rdd.dtype))
        for key, values in expected_dtype.iteritems():
            if not isinstance(values, (tuple, list)):
                values = [values]
            accept = accept and types[key] in values
        return accept

    if not isinstance(expected_dtype, (tuple, list)):
        expected_dtype = [expected_dtype]

    return rdd.dtype in expected_dtype
