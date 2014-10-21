# -*- coding: utf-8 -*-

from pyspark.mllib._common import _deserialize_double_vector
from pyspark.serializers import FramedSerializer
from pyspark.rdd import RDD
from pyspark import SparkContext


class DoubleVectorSerializer(FramedSerializer):

    def __init__(self):
        pass

    def loads(self, obj):
        return _deserialize_double_vector(bytearray(obj))


class DictVectorizer(object):

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


def test_dictvectorizer(sc):
    test_data = [[('foo', 1), ('bar', 2)], [('foo', 3), ('baz', 1)]]
    rdd = sc.parallelize(test_data)
    dv = DictVectorizer()
    print dv.fit_transform(rdd).collect()
    print dv.vocab


def test_dictvectorizer_2(sc):
    test_data = [[('foo', 1), ('bar', '2')], [('foo', 3), ('baz', '1')]]
    rdd = sc.parallelize(test_data)
    dv = DictVectorizer()
    dv.fit(rdd)
    print dv.transform(rdd).collect()
    print dv.vocab


if __name__ == "__main__":
    sc = SparkContext(appName="lensaNLP test")
    test_dictvectorizer(sc)
    test_dictvectorizer_2(sc)
    sc.stop()
