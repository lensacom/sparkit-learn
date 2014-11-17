# -*- coding: utf-8 -*-

from pyspark.rdd import RDD
from pyspark import SparkContext

from sklearn.feature_extraction import DictVectorizer

from ..rdd import ArrayRDD, DictRDD


class SparkDictVectorizer(DictVectorizer):

    def fit(self, Z):
        X = Z[:, 'X'] if isinstance(Z, DictRDD) else Z

        feature_names = X.map(self._fit).reduce(lambda a, b: a.union(b))
        feature_names = list(feature_names)
        feature_names.sort()

        vocab = dict((f, i) for i, f in enumerate(feature_names))

        self.feature_names_ = feature_names
        self.vocabulary_ = vocab

        return self

    def _fit(self, X, y=None):
        feature_names = []

        for x in X:
            for f, v in x.iteritems():
                if isinstance(v, basestring):
                    f = "%s%s%s" % (f, self.separator, v)
                feature_names.append(f)

        return set(feature_names)

    def transform(self, Z):
        f = super(SparkDictVectorizer, self).transform
        if isinstance(Z, DictRDD):
            return Z.transform(f, column='X')
        else:
            return Z.transform(f)

    def fit_transform(self, Z):
        return self.fit(Z).transform(Z)

