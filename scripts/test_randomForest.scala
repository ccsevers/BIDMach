/**
TODO CONCERNS:
1) What happens if certain nodes are empty
-> Looks like we must have all positive features because of the -1 from the maxs
 */

// :load /home/derrick/code/RandomForest/BIDMat/lib/test_randomForest.scala
import edu.berkeley.bid.CUMAT
import BIDMach.models.RandomForest
//import BIDMat.CPURandomForest
import BIDMach.models.EntropyEval

Mat.useCache = false;

// Test Random Forest!
val x : DMat = load("../Data/bidmatSpamData.mat", "Xtrain"); 
val y : DMat = load("../Data/bidmatSpamData.mat", "ytrain");

def testGPURandomForest : RandomForest = {
	val useGini = true
	val d = 7
	val t = 3
	val ns = 2
	// val feats : GMat = GMat(x.t);
	val feats : GMat = GMat(21\4.0\2\3 on 31\7.0\1\15 on 1.0\2.0\9\12) 
	val f : Int = feats.nrows;
	val n : Int = feats.ncols;
	// val cats : GMat = GMat(((iones(n,1) * irow(0->2)) == y).t);
	val cats : GMat = GMat(0\1\0\0 on 1\0\1\1);

	val randomForest : RandomForest = new RandomForest(d, t, ns, feats, cats, useGini);
	randomForest.train;
	println(randomForest.treePos.nrows)
	println(randomForest.treePos.ncols)
	println("Starting Classification")
	println(randomForest.classify(feats))
	randomForest
}

def testCPURandomForest {
	val d = 2;
	val t = 2;
	val ns = 2;
	// val feats : GMat = GMat(x.t);
	val feats : FMat = FMat(21\4.0\2\3 on 31\7.0\1\15 on 1.0\2.0\9\12) 
	val f : Int = feats.nrows;
	val n : Int = feats.ncols;
	// val cats : GMat = GMat(((iones(n,1) * irow(0->2)) == y).t);
	val cats : FMat = FMat(1\0\0\0 on 0\1\1\1);

	val randomForest : CPURandomForest = new CPURandomForest(d, t, ns, feats, cats);
	randomForest.train;
}

// testCPURandomForest
val rF = testGPURandomForest

/**
	Testing TreeProd
**/
// println("testing treeProd")
// val useGPU = feats match {case a:GMat => true; case _ => false };
// val n = feats.ncols;
// val f = feats.nrows;
// val c = cats.nrows;
// val nnodes = (math.pow(2, d) + 0.5).toInt; 
// println("nnodes: " + nnodes)
// /* Class Variable Matrices */
// val treePos = feats.izeros(t,n); //  GIMat.newOrCheckGIMat(t, n, null); 
// // treePos(0, 0) = 0;
// // treePos(0, 1) = 0;
// // treePos(0, 2) = 0;
// // treePos(0, 3) = 0;
// treePos(0, 0) = 1;
// treePos(0, 1) = 1;
// treePos(0, 2) = 1;
// treePos(0, 3) = 2;
// var treesArray = feats.izeros(ns, t * nnodes);
// val treeTemp = IMat(f * rand(ns, t * nnodes));
// min(treeTemp, f-1, treeTemp);
// treesArray <-- treeTemp;
// val oTreePos = feats.izeros(t, n); 
// val oTreeVal = feats.zeros(t, n);

// for (k <- 1 until d) {
// 	println("Running the treeprod #" + k);
// 	GMat.treeProd(treesArray, feats, treePos, oTreeVal);
// 	val e = new EntropyEval(oTreeVal, cats, d, k)
// 	e.getThresholdsAndUpdateTreesArray(treePos, oTreeVal, treesArray)

// 	println("Starting TreeStep #" + k)
// 	GMat.treeProd(treesArray, feats, treePos, treePos);
// 	println("treePos Changed after stepping")
// 	println(treePos)
// }

