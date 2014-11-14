# -*- coding: utf-8 -*-

from ..rdd import DictRDD, ArrayRDD

import numpy as np
import scipy.sparse as sp

from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import _document_frequency


class SparkHashingVectorizer(HashingVectorizer):

    def transform(self, Z):
        mapper = super(SparkHashingVectorizer, self).transform
        if isinstance(Z, DictRDD):
            return Z.map(mapper, column='X')
        elif isinstance(Z, ArrayRDD):
            return Z.map(mapper)
        else:
            raise TypeError(
                "Expected DictRDD or ArrayRDD, given {0}".format(type(Z)))

    fit_transform = transform


class SparkTfidfTransformer(TfidfTransformer):

    def fit(self, Z):
        def mapper(X):
            if not sp.issparse(X):
                X = sp.csc_matrix(X)
            if self.use_idf:
                return _document_frequency(X)

        if self.use_idf:
            if isinstance(Z, DictRDD):
                X = Z['X']
            elif isinstance(Z, ArrayRDD):
                X = Z
            else:
                raise TypeError(
                    "Expected DictRDD or ArrayRDD, given {0}".format(type(Z)))

            n_samples, n_features = X.shape
            df = X.map(mapper).sum()

            # perform idf smoothing if required
            df += int(self.smooth_idf)
            n_samples += int(self.smooth_idf)

            # log1p instead of log makes sure terms with zero idf don't get
            # suppressed entirely
            idf = np.log(float(n_samples) / df) + 1.0
            self._idf_diag = sp.spdiags(idf,
                                        diags=0, m=n_features, n=n_features)
        return self

    def transform(self, Z, copy=True):
        mapper = super(SparkTfidfTransformer, self).transform
        if isinstance(Z, DictRDD):
            return Z.map(mapper, column='X')
        elif isinstance(Z, ArrayRDD):
            return Z.map(mapper)
        else:
            raise TypeError(
                "Expected DictRDD or ArrayRDD, given {0}".format(type(Z)))

