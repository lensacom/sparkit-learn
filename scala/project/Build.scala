import sbt.Keys._
import sbt._
import sbtassembly.Plugin._
import org.scalastyle.sbt.ScalastylePlugin


object RootBuild extends Build {

  val Organization = "splearn"
  val ScalaVersion = "2.10.4"
  val Version = "0.0.1-SNAPSHOT"

  lazy val dictVectorizer = Project(
    id="dict-vectorizer",
    base=file("dict-vectorizer"),
    settings = assemblySettings ++ Seq(
      organization := Organization,
      name := "dict-vectorizer",
      version := Version,
      scalaVersion := ScalaVersion,
      libraryDependencies ++= Seq(
	"org.scalatest" % "scalatest_2.10" % "2.2.0" % "test",
	"org.apache.spark" %% "spark-core" % "1.1.0" % "provided",
	"org.apache.spark" %% "spark-mllib" % "1.1.0" % "provided"
      )
    ) ++ ScalastylePlugin.Settings
  )
}
