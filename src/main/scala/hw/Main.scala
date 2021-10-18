package hw

import breeze.linalg.DenseVector
import breeze.stats.regression.leastSquares
import regressor.CustomSGD
import utils.Utils._

object Main {
  def main(args: Array[String]) {
    val argList = args.toList
    val config = parseArgs(argList)
    val (trainData, testData) = readCsv(config.dataset_path)
    val (trainX, trainY) = prepareData(trainData)
    val (testX, testY) = prepareData(testData)
    val modelCustom = new CustomSGD(trainX.cols)
    modelCustom.fit(trainX, trainY)
    val predYCustom: DenseVector[Double] = modelCustom.predict(testX)
    val modelBreeze = leastSquares(trainX, trainY)
    val predYBreeze: DenseVector[Double] = modelBreeze(testX)
    val maeCustom = meanAbsoluteError(predYCustom, testY)
    val maeBreeze = meanAbsoluteError(predYBreeze, testY)
    println(s"Custom SGD MAE: $maeCustom")
    println(s"Breeze leastSquares MAE: $maeBreeze")
    vectorToFile(config.output_path, predYCustom)
  }
}
