package splearn.feature.dictvectorizer


import org.scalatest._
import sparktest.SparkTestUtils
import org.apache.spark.mllib.linalg.Vectors


class DictVectorizerSpec extends SparkTestUtils with ShouldMatchers {

  val testData = Array(Array("foo" -> 1, "bar" -> 2), Array("foo" -> 3, "baz" -> 1))

  val testData2 = Array(Array("foo" -> "1", "bar" -> "2"), Array("foo" -> "3", "baz" -> "1"))

  sparkTest("DictVectorizer extractFeatureKey") {
    val key0 = DictVectorizer.extractFeatureKey("=")("key" -> "value")
    val key1 = DictVectorizer.extractFeatureKey("=")("key" -> 2)
    key0 should be ("key=value")
    key1 should be ("key")
  }

  sparkTest("DictVectorizer fitTransform") {
    val data = sc.parallelize(testData)
    val model = DictVectorizer.fitTransform(data)
    model.vocab should be (Map("bar" -> 0l, "baz" -> 1l, "foo" -> 2l))    
    model.data.collect should be (Array(Vectors.sparse(3, Array(0, 2), Array(2.0, 1.0)), 
  					Vectors.sparse(3, Array(1, 2), Array(1.0, 3.0))))
  }

  sparkTest("DictVectorizer learnVocab") {
    val data = sc.parallelize(testData)
    val vocab = DictVectorizer.fit(data)
    vocab should be (Map("bar" -> 0l, "baz" -> 1l, "foo" -> 2l))
  }

  sparkTest("DictVectorizer learnVocab string value") {
    val data = sc.parallelize(testData2)
    val vocab = DictVectorizer.fit(data)
    vocab should be (Map("bar=2" -> 0l, "baz=1" -> 1l, "foo=1" -> 2l, "foo=3" -> 3l))
  }
}
