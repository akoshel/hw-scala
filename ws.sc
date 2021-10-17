import breeze.stats.regression.leastSquares
import utils.Utils.{readCsv, prepareData, meanAbsoluteError, vectorToFile}

val (trainData, testData) = readCsv("C:/Homeworks/hw-scala/data/Marketing_Data.csv")

val (trainX, trainY) = prepareData(trainData)
val (testX, testY) = prepareData(testData)


val model = leastSquares(trainX, trainY)

val predY = model(testX)

meanAbsoluteError(predY, testY)

val outputPath: String = "C:/Homeworks/hw-scala/data/result.txt"

vectorToFile(outputPath, predY)
