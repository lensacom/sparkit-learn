# -*- coding: utf-8 -*-

from pyspark import SparkContext
from pyspark.mllib._common import (_get_unmangled_double_vector_rdd,
                                   _serialize_double_vector)
from pyspark.mllib.linalg import Vectors


class DbscanModel(object):

    def __init__(self, model):
        self._model = model

    def predict(self, point):
        serialized = _serialize_double_vector(point)
        return self._model.predict(serialized)


class Dbscan(object):

    @classmethod
    def train(cls, rdd, epsilon, numOfPoints):
        sc = rdd.context
        jrdd = _get_unmangled_double_vector_rdd(rdd)._jrdd
        model = sc._jvm.PythonDbscanAPI().train(jrdd, epsilon, numOfPoints)
        return DbscanModel(model)


if __name__ == '__main__':
    sc = SparkContext(appName="lensaNLP test")
    model = Dbscan.train(sc.parallelize([Vectors.sparse(4, {1: 1.0, 3: 5.5})]), 25, 30)
    print(model.predict(Vectors.sparse(4, {1: 1.0, 3: 5.5})))
    sc.stop()
