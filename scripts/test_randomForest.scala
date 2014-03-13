/**
TODO CONCERNS:
1) What happens if certain nodes are empty
-> Looks like we must have all positive features because of the -1 from the maxs
 */

// :load /home/derrick/code/RandomForest/BIDMat/lib/test_randomForest.scala
import edu.berkeley.bid.CUMAT
import BIDMach.models.RandomForest
import BIDMach.models.CPURandomForest
import BIDMach.models.BothRandomForest
// import BIDMach.models.EntropyEval

// Test Random Forest!
Mat.useCache = false;

val x : DMat = load("../Data/bidmatSpamData.mat", "Xtrain"); 
val xTest : DMat = load("../Data/bidmatSpamData.mat", "Xtest");
val y : DMat = load("../Data/bidmatSpamData.mat", "ytrain");
val yTest : DMat = load("../Data/bidmatSpamData.mat", "ytest");

def calcAccuracy(guess : Mat , actual : Mat) : Mat = {
	println("calcAccuracy")
	val correctness = (guess == actual)
	val summed = sum(correctness)
	println(correctness)
	return summed/ (correctness.length.toFloat)
}

def testCPUBothRandomForest : BothRandomForest = {
	val numCats = 2
	val impurityType = 1
	val d = 4
	val t = 1
	val ns = 2
	val feats : FMat = FMat(x.t);
	val f : Int = feats.nrows;
	val n : Int = feats.ncols;
	val cats : FMat = FMat(((iones(n,1) * irow(0->numCats)) == y).t);

	val randomForest : BothRandomForest = new BothRandomForest(d, t, ns, feats, cats, impurityType, numCats);
	println("BothRandomForest: Train")
	randomForest.train;

	randomForest
}

def testBothRandomForest : BothRandomForest = {
	println("testBothRandomForest")
	val numCats = 2
	val impurityType = 1
	val d = 4
	val t = 1
	val ns = 2
	val feats : GMat = GMat(x.t);
	// val feats : GMat = GMat(21\4.0\2\3 on 31\7.0\1\15 on 1.0\2.0\9\12) 
	val f : Int = feats.nrows;
	val n : Int = feats.ncols;
	val cats : GMat = GMat(((iones(n,1) * irow(0->numCats)) == y).t);
	// val cats : GMat = GMat(0\1\1\0 on 1\0\0\1);
	println("BothRandomForest: Create")
	val randomForest : BothRandomForest = new BothRandomForest(d, t, ns, feats, cats, impurityType, numCats);
	println("BothRandomForest: Train")
	randomForest.train;
	println(randomForest.treePos.nrows)
	println(randomForest.treePos.ncols)


	println("Starting Classification")
	GPUmem
	val testFeats = GMat(xTest.t)
	val testN : Int = testFeats.ncols
	val testCats : GMat = GMat(((iones(testN,1) * irow(0->numCats)) == yTest).t);
	val guessTemp = randomForest.classify(testFeats)
	val guess = GMat(guessTemp(0, 0->testN))
	println("guess")
	println(guess)
	println("Num ones for Guess")
	println(sum(guess))
	println("Num Guess Total")
	println(guess.ncols)
	println("testCats(1, 0->testN)")
	println(testCats(1, 0->testN))
	println("num actual ones")
	println(sum(testCats(1, 0->testN)))
	val accuracy = calcAccuracy(guess, testCats(1, 0->testN))
	println("accuracy")
	println(accuracy)
	randomForest
}

def testGPURandomForest : RandomForest = {
	val numCats = 2
	val impurityType = 1
	val d = 19
	val t = 64
	val ns = 2
	val feats : GMat = GMat(x.t);
	// val feats : GMat = GMat(21\4.0\2\3 on 31\7.0\1\15 on 1.0\2.0\9\12) 
	val f : Int = feats.nrows;
	val n : Int = feats.ncols;
	val cats : GMat = GMat(((iones(n,1) * irow(0->numCats)) == y).t);
	// val cats : GMat = GMat(0\1\1\0 on 1\0\0\1);
	val randomForest : RandomForest = new RandomForest(d, t, ns, feats, cats, impurityType, numCats);
	randomForest.train;
	println(randomForest.treePos.nrows)
	println(randomForest.treePos.ncols)

	println("Starting Classification")
	val testFeats = GMat(xTest.t)
	val testN : Int = testFeats.ncols
	val testCats : GMat = GMat(((iones(testN,1) * irow(0->numCats)) == yTest).t);
	val guessTemp = randomForest.classify(testFeats)
	val guess = GMat(guessTemp(0, 0->testN))
	println("guess")
	println(guess)
	println("Num ones for Guess")
	println(sum(guess))
	println("Num Guess Total")
	println(guess.ncols)
	println("testCats(1, 0->testN)")
	println(testCats(1, 0->testN))
	println("num actual ones")
	println(sum(testCats(1, 0->testN)))
	val accuracy = calcAccuracy(guess, testCats(1, 0->testN))
	println("accuracy")
	println(accuracy)
	randomForest
}

def testCPURandomForest : CPURandomForest = {
	val numCats = 2
	val impurityType = 1
	val gainThreshold = 0.2f
	val d = 15
	val t = 64
	val ns = 2
	val feats : FMat = FMat(x.t);
	// val feats : FMat = FMat(21\4.0\2\3 on 31\7.0\1\15 on 1.0\2.0\9\12) 
	val f : Int = feats.nrows;
	val n : Int = feats.ncols;
	val cats : FMat = FMat(((iones(n,1) * irow(0->numCats)) == y).t);
	// val cats : FMat = FMat(0\1\1\0 on 1\0\0\1);
	val randomForest : CPURandomForest = new CPURandomForest(d, t, ns, feats, cats, impurityType, numCats, gainThreshold);
	println("CPURandomForest: Train")
	randomForest.train;

	println("Starting Classification")
	val testFeats = FMat(xTest.t)
	val testN : Int = testFeats.ncols
	val testCats : FMat = FMat(((iones(testN,1) * irow(0->numCats)) == yTest).t);
	val guessTemp = randomForest.classify(testFeats)
	val guess = FMat(guessTemp(0, 0->testN))
	println("guess")
	println(guess)
	println("Num ones for Guess")
	println(sum(guess))
	println("Num Guess Total")
	println(guess.ncols)
	println("testCats(1, 0->testN)")
	println(testCats(1, 0->testN))
	println("num actual ones")
	println(sum(testCats(1, 0->testN)))
	val accuracy = calcAccuracy(guess, testCats(1, 0->testN))
	println("accuracy")
	println(accuracy)

	randomForest
}

def testDigitsCPURandomForest : CPURandomForest = {
	val x : IMat = load("../Data/digits.mat", "xTrain"); 
	val y : DMat = load("../Data/digits.mat", "yTrain"); 
	val xTest : IMat = load("../Data/digits.mat", "xTest");
	val yTest : DMat = load("../Data/digits.mat", "yTest");

	val numCats = 10
	val impurityType = 1
	val gainThreshold = 0f
	val d = 9
	val t = 12
	val ns = 2
	val feats : FMat = FMat(x);
	// val feats : FMat = FMat(21\4.0\2\3 on 31\7.0\1\15 on 1.0\2.0\9\12) 
	val f : Int = feats.nrows;
	val n : Int = feats.ncols;
	val cats : FMat = FMat(((iones(n,1) * irow(0->numCats)) == FMat(y)  ).t);
	// val cats : FMat = FMat(0\1\1\0 on 1\0\0\1);
	val randomForest : CPURandomForest = new CPURandomForest(d, t, ns, feats, cats, impurityType, numCats, gainThreshold);
	println("CPURandomForest: Train")
	randomForest.train;

	println("Starting Classification")
	val testFeats = FMat(xTest)
	val testN : Int = testFeats.ncols
	val testCats : FMat = FMat(((iones(testN,1) * irow(0->numCats)) == FMat(yTest)   ).t);
	val guessTemp = randomForest.classify(testFeats)
	val guess = FMat(guessTemp(0, 0->testN))
	println("guess")
	println(guess)
	println("Num ones for Guess")
	println(sum(guess))
	println("Num Guess Total")
	println(guess.ncols)
	println("testCats(1, 0->testN)")
	println(testCats(1, 0->testN))
	println("num actual ones")
	println(sum(testCats(1, 0->testN)))
	val accuracy = calcAccuracy(guess, testCats(1, 0->testN))
	println("accuracy")
	println(accuracy)

	randomForest
}


// val rF = testGPURandomForest
// val rF = testCPURandomForest
// val rF = testBothRandomForest
// val rF = testCPUBothRandomForest
val rF = testDigitsCPURandomForest

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

