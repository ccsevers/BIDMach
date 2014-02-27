import edu.berkeley.bid.CUMAT
import BIDMach.models.RandomForest
import BIDMach.models.EntropyEval

Mat.useCache = true;

// :load tests/testMultiply.scala
val matName = "/home/derrick/code/NewRandomForest/BIDMach/tests/multiply.mat"
// saveAs("/home/derrick/code/NewRandomForest/BIDMach/tests/multiply.mat", FMat(leftTotsT1), "leftTotsT1", FMat(leftTotsT2), "leftTotsT2")
val lT1 : FMat = load(matName,"leftTotsT1")
val leftTotsT1 : GMat = GMat(lT1)
val lT2 : FMat = load(matName,"leftTotsT2")
val leftTotsT2 : GMat = GMat(lT2)
// val l : FMat = load(matName,"leftAccumPctsts")
// val leftAccumPctsts : GMat = GMat(l)
println("Running Multiply With Mat.useCache = true")
val result = leftTotsT1 * leftTotsT2
println(result)
