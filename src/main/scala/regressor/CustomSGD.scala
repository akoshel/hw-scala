package regressor

import breeze.linalg.{DenseMatrix, DenseVector}

class CustomSGD(numFeatures: Int) {
  val lr: Double = 0.00001
  val numIterations: Int = 1000
  var w: DenseVector[Double] = DenseVector.rand(3)
  var b: Double = 1


  def fit(trainX: DenseMatrix[Double], trainY: DenseVector[Double]): Unit = {
    for (_ <- 0 until numIterations) {
      for (r <- 0 until trainX.rows) {
        val prediction = trainX(r, ::).t.dot(w) + b
        val grad_w = trainX(r, ::).t * (prediction - trainY(r))
        val grad_b = prediction - trainY(r)
        w -= lr * grad_w
        b -= lr * grad_b
      }
    }
    println("Fit done")
  }

  def predict(testX: DenseMatrix[Double]): DenseVector[Double] = {
    testX * w
  }

}
