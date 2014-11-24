# -*- coding: utf-8 -*-

from ..rdd import DictRDD, ArrayRDD

import numpy as np
import scipy.sparse as sp

from collections import defaultdict

from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import _document_frequency


class SparkCountVectorizer(CountVectorizer):

    def _count_vocab(self, raw_documents, fixed_vocab):
        if isinstance(raw_documents, DictRDD):
            raw_documents = raw_documents[:, 'X']

        if fixed_vocab:
            vocabulary = self.vocabulary_
        else:
            mapper = lambda x: super(SparkCountVectorizer, self)._count_vocab(x, False)[0].keys()
            reducer = lambda x, y: np.unique(np.concatenate([x, y]))
            keys = raw_documents.map(mapper).reduce(reducer)
            vocabulary = dict((f, i) for i, f in enumerate(keys))
            self.vocabulary_ = vocabulary

        self.fixed_vocabulary_ = True
        mapper = lambda x: \
            super(SparkCountVectorizer, self)._count_vocab(x, True)[1]
        return vocabulary, raw_documents.map(mapper)

    def fit(self, raw_documents):
        self.fit_transform(raw_documents)
        return self

    def fit_transform(self, raw_documents):
        return super(SparkCountVectorizer, self).fit_transform(raw_documents)


class SparkHashingVectorizer(HashingVectorizer):

    def transform(self, Z):
        mapper = super(SparkHashingVectorizer, self).transform
        if isinstance(Z, DictRDD):
            return Z.transform(mapper, column='X')
        elif isinstance(Z, ArrayRDD):
            return Z.transform(mapper)
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
            return Z.transform(mapper, column='X')
        elif isinstance(Z, ArrayRDD):
            return Z.transform(mapper)
        else:
            raise TypeError(
                "Expected DictRDD or ArrayRDD, given {0}".format(type(Z)))
