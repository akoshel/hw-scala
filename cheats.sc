import breeze.linalg.DenseVector
import breeze.stats.distributions.Gaussian


val  a = DenseVector(1, 2, 3)
val  b = DenseVector(1, 2, 3)
a * b

val normal01 = Gaussian(0, 1)
DenseVector.rand(3, normal01)
DenseVector.rand(3)