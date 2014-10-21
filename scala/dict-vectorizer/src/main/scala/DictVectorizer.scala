package splearn.feature.dictvectorizer

import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.{Vectors, Vector}


case class Dict2VecModel(vocab: Map[String, Int], data: RDD[Vector])


object DictVectorizer {

  def extractFeatureKey[V](separator: String)(kv: (String, V)): String = kv match {
    case (k, _: Int) => k
    case (k, v) => s"${k}${separator}$v"
  }

  def fit[V](X: RDD[Array[(String, V)]], separator: String = "="): Map[String, Int] = {
    val keys = X.flatMap(_.map(extractFeatureKey(separator))).distinct()
    keys.sortBy(identity).collect().zipWithIndex.toMap
  }

  def transform[V](X: RDD[Array[(String, V)]], vocab: Map[String, Int], separator: String = "=") = {
    val data = X.map { x =>
      val elements = x.foldLeft(List[(Int, Double)]())((acc, kv) => kv match {case (k, v) => {
	val extractedKey = extractFeatureKey(separator)(k -> v)
        val value = v match {
          case vint: Int => vint.toDouble
          case _ => 1.0
        }
	if (vocab.contains(extractedKey)) (vocab(extractedKey) -> value) :: acc else acc
      }})
      Vectors.sparse(vocab.size, elements)
    }
    Dict2VecModel(vocab, data)
  }

  def fitTransform[V](X: RDD[Array[(String, V)]], separator: String = "=") = {
    val vocab = fit[V](X, separator)
    transform(X, vocab, separator)
  }
}
