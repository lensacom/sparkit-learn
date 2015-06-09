from sklearn.neighbors import LSHForest


class SparkLSHForest(LSHForest):

    """Performs approximate nearest neighbor search using LSH forest.

    LSH Forest: Locality Sensitive Hashing forest [1] is an alternative
    method for vanilla approximate nearest neighbor search methods.
    LSH forest data structure has been implemented using sorted
    arrays and binary search and 32 bit fixed-length hashes.
    Random projection is used as the hash family which approximates
    cosine distance.

    The cosine distance is defined as ``1 - cosine_similarity``: the lowest
    value is 0 (identical point) but it is bounded above by 2 for the farthest
    points. Its value does not depend on the norm of the vector points but
    only on their relative angles.

    Parameters
    ----------

    n_estimators : int (default = 10)
        Number of trees in the LSH Forest.

    min_hash_match : int (default = 4)
        lowest hash length to be searched when candidate selection is
        performed for nearest neighbors.

    n_candidates : int (default = 10)
        Minimum number of candidates evaluated per estimator, assuming enough
        items meet the `min_hash_match` constraint.

    n_neighbors : int (default = 5)
        Number of neighbors to be returned from query function when
        it is not provided to the :meth:`kneighbors` method.

    radius : float, optinal (default = 1.0)
        Radius from the data point to its neighbors. This is the parameter
        space to use by default for the :meth`radius_neighbors` queries.

    radius_cutoff_ratio : float, optional (default = 0.9)
        A value ranges from 0 to 1. Radius neighbors will be searched until
        the ratio between total neighbors within the radius and the total
        candidates becomes less than this value unless it is terminated by
        hash length reaching `min_hash_match`.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Attributes
    ----------

    hash_functions_ : list of GaussianRandomProjectionHash objects
        Hash function g(p,x) for a tree is an array of 32 randomly generated
        float arrays with the same dimenstion as the data set. This array is
        stored in GaussianRandomProjectionHash object and can be obtained
        from ``components_`` attribute.

    trees_ : array, shape (n_estimators, n_samples)
        Each tree (corresponding to a hash function) contains an array of
        sorted hashed values. The array representation may change in future
        versions.

    original_indices_ : array, shape (n_estimators, n_samples)
        Original indices of sorted hashed values in the fitted index.

    References
    ----------

    .. [1] M. Bawa, T. Condie and P. Ganesan, "LSH Forest: Self-Tuning
           Indexes for Similarity Search", WWW '05 Proceedings of the
           14th international conference on World Wide Web,  651-660,
           2005.

    Examples
    --------
      >>> from sklearn.neighbors import LSHForest

      >>> X_train = [[5, 5, 2], [21, 5, 5], [1, 1, 1], [8, 9, 1], [6, 10, 2]]
      >>> X_test = [[9, 1, 6], [3, 1, 10], [7, 10, 3]]
      >>> lshf = LSHForest()
      >>> lshf.fit(X_train)  # doctest: +NORMALIZE_WHITESPACE
      LSHForest(min_hash_match=4, n_candidates=50, n_estimators=10,
                n_neighbors=5, radius=1.0, radius_cutoff_ratio=0.9,
                random_state=None)
      >>> distances, indices = lshf.kneighbors(X_test, n_neighbors=2)
      >>> distances                                        # doctest: +ELLIPSIS
      array([[ 0.069...,  0.149...],
             [ 0.229...,  0.481...],
             [ 0.004...,  0.014...]])
      >>> indices
      array([[1, 2],
             [2, 0],
             [4, 0]])

    """
    pass
