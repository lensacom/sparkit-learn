# -*- coding: utf-8 -*-

from pyspark.mllib._common import _deserialize_double_vector
from pyspark.serializers import FramedSerializer
from pyspark.rdd import RDD


class DoubleVectorSerializer(FramedSerializer):

    def __init__(self):
        pass

    def loads(self, obj):
        return _deserialize_double_vector(bytearray(obj))


class MllibDictVectorizer(object):

    def __init__(self):
        self._vocab = None

    @property
    def vocab(self):
        if not self.api:
            raise ValueError("Vocab wasn't initalized")
        return self.api.getVocab()

    def fit(self, rdd):
        sc = rdd.context
        self.api = sc._jvm.PythonDictVectorizerAPI()
        self.api.fit(rdd._jrdd)

    def transform(self, rdd):
        sc = rdd.context
        if not self.api:
            raise ValueError("Vocab wasn't initialized")
        jrdd = self.api.transform(rdd._jrdd)
        return RDD(jrdd, sc, DoubleVectorSerializer())

    def fit_transform(self, rdd):
        sc = rdd.context
        self.api = sc._jvm.PythonDictVectorizerAPI()
        jrdd = self.api.fitTransform(rdd._jrdd)
        return RDD(jrdd, sc, DoubleVectorSerializer())
