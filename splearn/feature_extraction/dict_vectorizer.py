# -*- coding: utf-8 -*-

import numpy as np
import scipy.sparse as sp
from pyspark import AccumulatorParam
from sklearn.externals import six
from sklearn.feature_extraction import DictVectorizer

from ..base import SparkBroadcasterMixin
from ..rdd import DictRDD
from ..utils.validation import check_rdd


class SparkDictVectorizer(DictVectorizer, SparkBroadcasterMixin):

    """Transforms DictRDDs containing feature-value mappings to vectors.
    This transformer turns lists of mappings (dict-like objects) of feature
    names to feature values into Numpy arrays or scipy.sparse matrices for use
    with scikit-learn estimators.
    When feature values are strings, this transformer will do a binary one-hot
    (aka one-of-K) coding: one boolean-valued feature is constructed for each
    of the possible string values that the feature can take on. For instance,
    a feature "f" that can take on the values "ham" and "spam" will become two
    features in the output, one signifying "f=ham", the other "f=spam".
    Features that do not occur in a sample (mapping) will have a zero value
    in the resulting array/matrix.

    Parameters
    ----------
    dtype : callable, optional
        The type of feature values. Passed to Numpy array/scipy.sparse matrix
        constructors as the dtype argument.
    separator: string, optional
        Separator string used when constructing new features for one-hot
        coding.
    sparse: boolean, optional.
        Whether transform should produce scipy.sparse matrices.
        True by default.
    sort: boolean, optional.
        Whether feature_names_ and vocabulary_ should be sorted when fitting.
        True by default.

    Attributes
    ----------
    vocabulary_ : dict
        A dictionary mapping feature names to feature indices.
    feature_names_ : list
        A list of length n_features containing the feature names (e.g., "f=ham"
        and "f=spam").

    Examples
    --------
    >>> import numpy as np
    >>> from splearn.feature_extraction import SparkDictVectorizer
    >>> from splearn.rdd import ArrayRDD, DictRDD
    >>> D = np.array([{'foo': 1, 'bar': 2},
    >>>               {'foo': 3, 'baz': 1},
    >>>               {'bar': 2, 'baz': 4},
    >>>               {'foo':5}]).reshape((4,1))
    >>> D_rdd = sc.parallelize(D, 2)
    >>> D_dict = DictRDD(D_rdd, columns=('X',))
    >>> v = SparkDictVectorizer(sparse=False)
    >>> X = v.fit(D_dict)
    >>> v.transform(D_dict).collect()
    [(array([[ 2.,  0.,  1.],
             [ 0.,  1.,  3.]]),),
     (array([[ 2.,  4.,  0.],
             [ 0.,  0.,  5.]]),)]
    >>> test = sc.parallelize([{'foo': 4, 'unseen_feature': 3}])
    >>> print np.concatenate(v.transform(ArrayRDD(test)).collect())
    [[ 0.  0.  4.]]
    """

    __transient__ = ['feature_names_', 'vocabulary_']

    def fit(self, Z):
        """Learn a list of feature name -> indices mappings.

        Parameters
        ----------
        Z : DictRDD with column 'X'
            Dict(s) or Mapping(s) from feature names (arbitrary Python
            objects) to feature values (strings or convertible to dtype).

        Returns
        -------
        self
        """
        X = Z[:, 'X'] if isinstance(Z, DictRDD) else Z

        """Create vocabulary
        """
        class SetAccum(AccumulatorParam):

            def zero(self, initialValue):
                return set(initialValue)

            def addInPlace(self, v1, v2):
                v1 |= v2
                return v1

        accum = X.context.accumulator(set(), SetAccum())

        def mapper(X, separator=self.separator):
            feature_names = []
            for x in X:
                for f, v in six.iteritems(x):
                    if isinstance(v, six.string_types):
                        f = "%s%s%s" % (f, self.separator, v)
                    feature_names.append(f)
            accum.add(set(feature_names))

        X.foreach(mapper)  # init vocabulary
        feature_names = list(accum.value)

        if self.sort:
            feature_names.sort()

        vocab = dict((f, i) for i, f in enumerate(feature_names))

        self.feature_names_ = feature_names
        self.vocabulary_ = vocab

        return self

    def transform(self, Z):
        """Transform ArrayRDD's (or DictRDD's 'X' column's) feature->value dicts
        to array or sparse matrix.
        Named features not encountered during fit or fit_transform will be
        silently ignored.

        Parameters
        ----------
        Z : ArrayRDD or DictRDD with column 'X' containing Mapping or
            iterable over Mappings, length = n_samples
            Dict(s) or Mapping(s) from feature names (arbitrary Python
            objects) to feature values (strings or convertible to dtype).

        Returns
        -------
        Z : transformed, containing {array, sparse matrix}
            Feature vectors; always 2-d.
        """
        mapper = self.broadcast(super(SparkDictVectorizer, self).transform,
                                Z.context)
        dtype = sp.spmatrix if self.sparse else np.ndarray
        return Z.transform(mapper, column='X', dtype=dtype)

    def fit_transform(self, Z):
        """Learn a list of feature name -> indices mappings and transform Z.
        Like fit(Z) followed by transform(Z).

        Parameters
        ----------
        Z : Z : ArrayRDD or DictRDD with column 'X' containing Mapping or
            iterable over Mappings
            Dict(s) or Mapping(s) from feature names (arbitrary Python
            objects) to feature values (strings or convertible to dtype).

        Returns
        -------
        Z : transformed, containing {array, sparse matrix}
            Feature vectors; always 2-d.
        """
        return self.fit(Z).transform(Z)
