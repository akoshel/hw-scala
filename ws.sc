import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.stats.regression.leastSquares
import utils.Utils.{meanAbsoluteError, prepareData, readCsv, vectorToFile}
import regressor.CustomSGD
import java.util.Arrays

val (trainData, testData) = readCsv("C:/Homeworks/hw-scala/data/Marketing_Data.csv")

val (trainX, trainY) = prepareData(trainData)
val (testX, testY) = prepareData(testData)


val model = leastSquares(trainX, trainY)
model.coefficients
val predY = model(testX)

meanAbsoluteError(predY, testY)

val outputPath: String = "C:/Homeworks/hw-scala/data/result.txt"

//vectorToFile(outputPath, predY)

//val modelCustom = new CustomSGD(trainX.cols)
//modelCustom.fit(trainX, trainY)
//val predict = testX * modelCustom.w
val w = DenseVector(1.0, 2.0, 3.0)
val pre = trainX(1, ::).t
pre * w
pre.dot(w)