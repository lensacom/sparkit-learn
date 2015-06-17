from splearn.rdd import DictRDD, BlockRDD


def check_rdd_dtype(rdd, expected_dtype):
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
    if not isinstance(rdd, BlockRDD):
        raise TypeError("Expected {0} for parameter rdd, got {1}."
                        .format(BlockRDD, type(rdd)))
    if isinstance(rdd, DictRDD):
        if not isinstance(expected_dtype, dict):
            raise TypeError('Expected {0} for parameter '
                            'expected_dtype, got {1}.'
                            .format(dict, type(expected_dtype)))
        accept = True
        types = dict(list(zip(rdd.columns, rdd.dtype)))
        for key, values in expected_dtype.items():
            if not isinstance(values, (tuple, list)):
                values = [values]
            accept = accept and types[key] in values
        return accept

    if not isinstance(expected_dtype, (tuple, list)):
        expected_dtype = [expected_dtype]

    return rdd.dtype in expected_dtype

def check_rdd(rdd, expected_dtype):
    """Wrapper function to check_rdd_dtype. Raises TypeError in case of dtype
    mismatch.

    Parameters:
    -----------
    rdd: splearn.BlockRDD
        The RDD to check
    expected_dtype: {type, list of types, tuple of types, dict of types}
        Expected type(s). If the RDD is a DictRDD the parameter type is
        restricted to dict.
    """
    if not check_rdd_dtype(rdd, expected_dtype):
        raise TypeError("{0} dtype mismatch.".format(rdd))
