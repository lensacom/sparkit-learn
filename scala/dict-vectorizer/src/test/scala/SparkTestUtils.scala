package sparktest

import org.scalatest._
import org.apache.spark.SparkContext
import org.apache.log4j.{Level, Logger}


object SparkTest extends org.scalatest.Tag("com.qf.test.tags.SparkTest")


trait SparkTestUtils extends FunSuite {
  var sc: SparkContext = _

  def sparkTest(name: String, silenceSpark: Boolean = true)(body: => Unit) {
    test(name, SparkTest) {
      val prevLevels = if (silenceSpark) Some(SparkUtil.silenceSpark()) else None
      sc = new SparkContext("local[4]", name)
      try {
        body
      } finally {
        sc.stop
        sc = null
        System.clearProperty("spark.master.port")
        SparkUtil.restoreLogLevels(prevLevels)
      }
    }
  }
}


object SparkUtil {
  def silenceSpark() = {
    setLogLevels(Level.WARN, Seq("spark", "org.eclipse.jetty", "akka", "org.apache.spark"))
  }

  def restoreLogLevels(prevLevels: Option[Map[String, Level]]) = prevLevels match {
     case Some(levels) => levels.foreach({case (name, level) => Logger.getLogger(name).setLevel(level)})
     case None => null
   }

  def setLogLevels(level: org.apache.log4j.Level, loggers: Seq[String]) = {
    loggers.map {
      loggerName =>
        val logger = Logger.getLogger(loggerName)
        val prevLevel = logger.getLevel()
        logger.setLevel(level)
        loggerName -> prevLevel
    }.toMap
  }
}
