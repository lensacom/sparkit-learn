package org.apache.spark.mllib.api.python

import splearn.feature.dictvectorizer.{Dict2VecModel, DictVectorizer}
import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.api.java.JavaRDD
import scala.collection.JavaConverters._
import net.razorvine.pickle.Unpickler
import java.util.ArrayList


@DeveloperApi
class PythonDictVectorizerAPI extends Serializable {
  
  var vocab: Option[Map[String, Int]] = None

  def getVocab = vocab.getOrElse(Map[String, Int]())

  def fit(inputData: JavaRDD[Array[Byte]]) {
    val rdd = unpickleInputData(inputData)
    vocab = Some(DictVectorizer.fit(rdd))
  }

  def transform(inputData: JavaRDD[Array[Byte]]) = {
    val rdd = unpickleInputData(inputData)
    val model = DictVectorizer.transform(rdd, getVocab)
    new JavaRDD(model.data.map({v => SerDe.serializeDoubleVector(v)}))
  }

  def fitTransform(inputData: JavaRDD[Array[Byte]]) = {
    val rdd = unpickleInputData(inputData)
    val model = DictVectorizer.fitTransform(rdd)
    vocab = Some(model.vocab)
    new JavaRDD(model.data.map({v => SerDe.serializeDoubleVector(v)}))
  }

  private def unpickleInputData(inputData: JavaRDD[Array[Byte]]) = {
    inputData.rdd.map(d => {
      val l = new Unpickler().loads(d).asInstanceOf[ArrayList[Array[Any]]]
      l.asScala.toArray.map(t => (t(0).asInstanceOf[String], t(1)))
    })
  }
}
