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

  lazy val sparkDbscan = RootProject(
    uri("https://github.com/lesbroot/spark_dbscan.git#4fe5c23c7ec06c9af8822e9ad03d70cf6f7bf73b")
  )

  lazy val dbscanWrapper = Project(
    id="dbscan-wrapper",
    base=file("dbscan-wrapper"),
    settings = assemblySettings ++ Seq(
      organization := Organization,
      name := "dbscan-wrapper",
      version := Version,
      scalaVersion := ScalaVersion,
//      resolvers += "Aliaksei Litouka's repository" at "http://alitouka-public.s3-website-us-east-1.amazonaws.com/",
      libraryDependencies ++= Seq(
//	"org.alitouka" % "spark_dbscan_2.10" % "0.0.2",
	"org.apache.spark" %% "spark-core" % "1.1.0" % "provided",
	"org.apache.spark" %% "spark-mllib" % "1.1.0" % "provided"
      )
    ) ++ ScalastylePlugin.Settings
  ).dependsOn(sparkDbscan)

}
