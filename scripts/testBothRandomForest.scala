import BIDMach.models.BothRandomForest

Mat.useCache = false;

/** Load the Data */
val x : DMat = load("../Data/bidmatSpamData.mat", "Xtrain"); 
val xTest : DMat = load("../Data/bidmatSpamData.mat", "Xtest");
val y : DMat = load("../Data/bidmatSpamData.mat", "ytrain");
val yTest : DMat = load("../Data/bidmatSpamData.mat", "ytest");

/** Define the Forest Params */
val numCats = 2
val impurityType = 1
val d = 3
val t = 1
val ns = 2
val featsF : FMat = FMat(x.t)
val featsG : GMat = GMat(x.t);
val f : Int = featsF.nrows;
val n : Int = featsF.ncols;
val catsF : FMat = FMat(((iones(n,1) * irow(0->numCats)) == y).t)
val catsG : GMat = GMat(((iones(n,1) * irow(0->numCats)) == y).t)

/** Prep the forests */
val gpuRandomForest : BothRandomForest = new BothRandomForest(d, t, ns, featsG, catsG, impurityType, numCats)
val cpuRandomForest : BothRandomForest = new BothRandomForest(d, t, ns, featsF, catsF, impurityType, numCats)

/** Make the treesArray the same */
val treeTemp = IMat(f * rand(ns + 1, t * gpuRandomForest.nnodes))
min(treeTemp, f-1, treeTemp)
gpuRandomForest.treesArray <-- treeTemp 
cpuRandomForest.treesArray <-- treeTemp

/**
 * Generic gpuCPUTester code
 */

def getAcc(comp : Mat) : Float = {
	(comp) match {
		case (c : FMat) => {
			return sum(sum(c , 1) ,2)(0,0).toFloat / c.length
		}
		case (c : IMat) => {
			return sum(sum(c , 1) ,2)(0,0).toFloat / c.length
		}
	}
}

def compare(gpuVer : Mat, cpuVer : Mat) : Float = {
 	(gpuVer, cpuVer) match {
 		case (gVer : GMat, cVer : FMat) => {
 			return getAcc(FMat(gVer) == cVer)
 		}
 		case (gVer : GIMat, cVer : IMat) => {
 			return getAcc(IMat(gVer) == cVer)
 		}
 	}
}

/**
 * testTreeProd
 */
def testTreeProd(gRForest : BothRandomForest, cRForest : BothRandomForest) {
	gRForest.treeProd(gRForest.treesArray, gRForest.treesArrayG, gRForest.feats, gRForest.treePos, gRForest.oTreeVal)
	cRForest.treeProd(cRForest.treesArray, cRForest.treesArrayF, cRForest.feats, cRForest.treePos, cRForest.oTreeVal)
	println(compare(gRForest.treePos, cRForest.treePos))
	println(compare(gRForest.oTreeVal, cRForest.oTreeVal))
}

def test
/*

/**
 * Main
 */
testTreeProd(gpuRandomForest, cpuRandomForest)

