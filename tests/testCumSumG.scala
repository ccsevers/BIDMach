import edu.berkeley.bid.CUMAT
import BIDMach.models.RandomForest
import BIDMach.models.EntropyEval

// :load tests/testCumSumG.scala
val matName = "/home/derrick/code/NewRandomForest/BIDMach/tests/leftImpurityCumSumG.mat"
// saveAs(matName, FMat(pctsts), "pctsts", IMat(jc), "jc", FMat(leftAccumPctsts), "leftAccumPctsts")
val pct : FMat = load(matName,"pctsts")
val pctsts : GMat = GMat(pct)
val j : IMat = load(matName,"jc")
val jc : GIMat = GIMat(j)
// val l : FMat = load(matName,"leftAccumPctsts")
// val leftAccumPctsts : GMat = GMat(l)
println("Running CumSumG:")
val result = cumsumg(pctsts, jc, null);
// println("Recorded Output")
// println(leftAccumPctsts)
println("CumSumG Output:")
println(result)
println("If you run this script multiple times you will notice that the Output of CumSumG is different each time")