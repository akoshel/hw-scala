package utils

import breeze.linalg.{DenseMatrix, DenseVector, csvread}
import breeze.numerics.abs
import breeze.stats.mean

import java.io.{File, FileWriter}

object Utils {
  def readCsv(path: String): (DenseMatrix[Double], DenseMatrix[Double]) = {
    val inputFile: File = new File(path)
    val data: DenseMatrix[Double] = csvread(inputFile, skipLines = 1)
    val splitBoundary = (data.rows * 0.75).toInt
    val trainData = data(0 to splitBoundary, ::)
    val testData = data(splitBoundary until data.rows, ::)
    (trainData, testData)
  }

  def vectorToFile(path: String, vector: DenseVector[Double]): Unit = {
    val fileWriter = new FileWriter(new File(path))
    for (n <- vector) {
      fileWriter.write(s"$n\n")
    }
    fileWriter.close()
  }

  def prepareData(matrix: DenseMatrix[Double]): (DenseMatrix[Double], DenseVector[Double]) = {
    val featureCol: Seq[Int] = Seq(0, 1, 2)
    val targetCol: Int = 3
    val features = matrix(::, featureCol).toDenseMatrix
    val target: DenseVector[Double] = matrix(::, targetCol)
    (features, target)
  }

  def meanAbsoluteError(pred: DenseVector[Double], hold: DenseVector[Double]): Double = {
    mean(abs(pred - hold))
  }
}
