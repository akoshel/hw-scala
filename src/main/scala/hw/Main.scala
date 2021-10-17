package hw

import breeze.stats.regression.leastSquares
import utils.Utils.{meanAbsoluteError, prepareData, readCsv, vectorToFile}


object Main {
  def main(args: Array[String]) {
    val (trainData, testData) = readCsv("C:/Homeworks/hw-scala/data/Marketing_Data.csv")
    val (trainX, trainY) = prepareData(trainData)
    val (testX, testY) = prepareData(testData)
    val model = leastSquares(trainX, trainY)
    val predY = model(testX)
    println(meanAbsoluteError(predY, testY))
    val outputPath: String = "C:/Homeworks/hw-scala/data/result.txt"
    vectorToFile(outputPath, predY)
  }
}
