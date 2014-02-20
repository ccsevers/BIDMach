import edu.berkeley.bid.CUMAT
import BIDMach.models.RandomForest
import BIDMach.models.EntropyEval

// :load tests/testTreeSteps.scala
// saveAs("/home/derrick/code/NewRandomForest/BIDMach/tests/testTreeSteps.mat", IMat(treesArray), "treesArray", FMat(fs), "fs", IMat(newTreePos), "newTreePos", IMat(treeCats), "treeCats")
val treesArray : IMat = load("/home/derrick/code/NewRandomForest/BIDMach/tests/testTreeSteps.mat","treesArray")
val tA : GIMat = GIMat(treesArray)
val fs : FMat = load("/home/derrick/code/NewRandomForest/BIDMach/tests/testTreeSteps.mat","fs")
val f : GMat = GMat(fs)
val newTreePos : IMat = load("/home/derrick/code/NewRandomForest/BIDMach/tests/testTreeSteps.mat","newTreePos")
val nTP : GIMat = GIMat(newTreePos)
val treeCats : IMat = load("/home/derrick/code/NewRandomForest/BIDMach/tests/testTreeSteps.mat","treeCats")
val tC : GIMat = GIMat(treeCats)
println("Running TreeSearch:")
GMat.treeSearch(tA, f, nTP, tC)
println("TreeSearch Output for Categories:")
println(tC)
println("Instead of returning 1,1,1,1 it should return -1186361344,-1186361344,-1186361344,-1186361344")
