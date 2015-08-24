package org.template.textclassification

import io.prediction.controller.P2LAlgorithm
import io.prediction.controller.Params
import org.apache.spark.SparkContext
import org.apache.spark.mllib.classification.NaiveBayes
import org.apache.spark.mllib.classification.NaiveBayesModel
import org.apache.spark.mllib.linalg.Vector
import com.github.fommil.netlib.F2jBLAS

import scala.math._

// 1. Define parameters for Supervised Learning Model. We are
// using a Naive Bayes classifier, which gives us only one
// hyperparameter in this stage.

case class  NBAlgorithmParams(
  lambda: Double
) extends Params



// 2. Define SupervisedAlgorithm class.

class NBAlgorithm(
  val sap: NBAlgorithmParams
) extends P2LAlgorithm[PreparedData, NBModel, Query, PredictedResult] {

  // Train your model.
  def train(sc: SparkContext, pd: PreparedData): NBModel = {
    new NBModel(pd, sap.lambda)
  }

  // Prediction method for trained model.
  def predict(model: NBModel, query: Query): PredictedResult = {
    model.predict(query.folder, query.ret, query.subject, query.from)
  }
}

class NBModel(
val pd: PreparedData,
lambda: Double
) extends Serializable {



  // 1. Fit a Naive Bayes model using the prepared data.

  private val nb : NaiveBayesModel = NaiveBayes.train(
    pd.transformedData, lambda)



  // 2. Set up linear algebra framework.

  private def innerProduct (x : Array[Double], y : Array[Double]) : Double = {
    //println(x.zip(y).map(e => e._1 * e._2).sum)
    x.zip(y).map(e => e._1 * e._2).sum
    
  }

  val normalize = (u: Array[Double]) => {
    val uSum = u.sum

    u.map(e => e / uSum)
  }



  // 3. Given a document string, return a vector of corresponding
  // class membership probabilities.

  private def getScores(folder : String, ret: String, subject: String, from: String): Array[Double] = {
    // Helper function used to normalize probability scores.
    // Returns an object of type Array[Double]

    // Vectorize query,

    

    val x: Vector = pd.transform(folder, ret, subject, from)

    normalize(
      nb.pi
      .zip(nb.theta)
      .map(
      e => exp((innerProduct(e._2, x.toArray)) + e._1))
    )
  }

  // 4. Implement predict method for our model using
  // the prediction rule given in tutorial.

  def predict(folder : String, ret: String, subject: String, from: String) : PredictedResult = {

    val x: Array[Double] = getScores(folder, ret, subject, from)
    val y: (Double, Double) = (nb.labels zip x).maxBy(_._2)

    //println(new PredictedResult(pd.categoryMap.getOrElse(y._1, ""), y._2))
    new PredictedResult(pd.categoryMap.getOrElse(y._1, ""), y._2)
  }
}