package org.template.textclassification


import io.prediction.controller.PPreparator
import io.prediction.controller.Params
import org.apache.spark.SparkContext
import org.apache.spark.mllib.feature.{IDF, IDFModel, HashingTF}
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

import scala.collection.immutable.HashMap
import scala.collection.JavaConversions._
import scala.math._


// 1. Initialize Preparator parameters. Recall that for our data
// representation we are only required to input the n-gram window
// components.

case class PreparatorParams(
  nGram : Int
) extends Params



// 2. Initialize your Preparator class.

class Preparator(pp: PreparatorParams) extends PPreparator[TrainingData, PreparedData] {

  // Prepare your training data.
  def prepare(sc : SparkContext, td: TrainingData): PreparedData = {
    new PreparedData(td, pp.nGram)
  }
}

//------PreparedData------------------------

class PreparedData (
val td : TrainingData,
val nGram : Int
) extends Serializable {



  // 1. Hashing function: Text -> term frequency vector.

  private val hasher = new HashingTF()

  private def hashTF (folder : String, ret: String, subject: String, from: String) : Vector = {
    val parseString = folder + " " + ret + " " + subject + " " + from
    val newList : Array[String] = parseString.split(" ")
    .sliding(nGram)
    .map(_.mkString)
    .toArray


    hasher.transform(newList)
  }

  // 2. Term frequency vector -> t.f.-i.d.f. vector.

  val idf : IDFModel = new IDF().fit(td.data.map(e => hashTF(e.folder, e.ret, e.subject, e.from)))


  // 3. Document Transformer: text => tf-idf vector.

  def transform(folder : String, ret: String, subject: String, from: String): Vector = {
    // Map(n-gram -> document tf)
    idf.transform(hashTF(folder, ret, subject, from))
  }


  // 4. Data Transformer: RDD[documents] => RDD[LabeledPoints]

  val transformedData: RDD[(LabeledPoint)] = {
    td.data.map(e => LabeledPoint(e.label, transform(e.folder, e.ret, e.subject, e.from)))

  }


  // 5. Finally extract category map, associating label to category.
  val categoryMap = td.data.map(e => (e.label, e.category)).collectAsMap



}




