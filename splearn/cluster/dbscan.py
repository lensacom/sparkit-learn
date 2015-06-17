# -*- coding: utf-8 -*-

from pyspark.mllib._common import (_get_unmangled_double_vector_rdd,
                                   _serialize_double_vector)


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
