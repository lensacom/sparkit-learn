import unittest
import logging
from pyspark import SparkContext
from pyspark.conf import SparkConf


class SplearnTestCase(unittest.TestCase):

    def setUp(self):
        logger = logging.getLogger("py4j.java_gateway")
        logger.setLevel(logging.ERROR)
        logger.addHandler(logging.StreamHandler())

        class_name = self.__class__.__name__
        conf = SparkConf().setAppName(class_name) \
                          .setMaster('local[2]') \
                          .set('spark.executor.memory', '512m')
        self.sc = SparkContext(conf=conf)


    def tearDown(self):
        self.sc.stop()
        # To avoid Akka rebinding to the same port, since it doesn't unbind
        # immediately on shutdown
        self.sc._jvm.System.clearProperty("spark.driver.port")
