package BIDMach.models

import BIDMat.{SBMat,CMat,CSMat,DMat,Dict,IDict,FMat,GMat,GIMat,GSMat,HMat,IMat,Mat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMat.Solvers._
import BIDMat.Sorting._
import edu.berkeley.bid.CUMAT

/**
 * Random Forest Implementation
 */
class CPURandomForest(d : Int, t: Int, ns: Int, fs : Mat, cs : Mat, impurityType : Int = 1, numCats : Int, gainThreshold : Float = 0) {
	/*
		Class Variables
		n = # of samples
		f = # of total features
		k = pointer to current level of each tree
		d = largest possible level of all the trees
		t = # of trees
		c = # of categories
		ns = # of features considered per node
		nnodes = # of total nodes per tree

		feats = f x n matrix representing the raw feature values
		cats = c x n matrix representing 0/1 which categories
		treePos = t x n matrix representing the which node a specific sample is on for all trees
		treesArray = ns x (t * (2^d - 1)) matrix representing feature indices for each node
		oTreeVal = t * n matrix representing the float inner products received from running the treeProd method
	*/

	/* Class Variable Constants */
	val useGPU = fs match {case a:GMat => true; case _ => false };
	val feats = fs
	val cats = cs
	val n = feats.ncols
	val f = feats.nrows
	val c = cats.nrows
	val nnodes = (math.pow(2, d) + 0.5).toInt

	/* Class Variable Matrices */
	val treePos = feats.izeros(t,n)
	val treesArray = feats.izeros(ns + 1, t * nnodes)
	val treeTemp = IMat(f * rand(ns + 1, t * nnodes))
	min(treeTemp, f-1, treeTemp)
	treesArray <-- treeTemp
	val treesArrayF = feats.zeros(ns + 1, t * nnodes)
	val oTreePos = feats.izeros(t, n);
	val oTreeVal = feats.zeros(t, n)

	/** Made an instance variable so that this could be tested better */
	var e : CPUEntropyEval = null
	
	/******************************************************************************************************************************
	 * PUBLIC: train
	 ******************************************************************************************************************************/
	def train {
		var k = 0;
		while (k < d - 1) { // d of them; each level
   			treeProd(treesArray, treesArrayF, feats, treePos, oTreeVal);
			e = new CPUEntropyEval(oTreeVal, cats, d, k, impurityType, gainThreshold)
			e.getThresholdsAndUpdateTreesArray(treePos, oTreeVal, treesArray, treesArrayF)
			treeProd(treesArray, treesArrayF, feats, treePos, oTreePos)
			treePos <-- oTreePos
			k = k + 1
		}
		markAllCurPositionsAsLeavesAndCategorizeThem(treesArray, treesArrayF, treePos)
	}

	/******************************************************************************************************************************
	 * PUBLIC: classify
	 ******************************************************************************************************************************/
	
	def classify(tFeats : Mat) : Mat = {
		val newTreePos = tFeats.izeros(t, tFeats.ncols)
		val treeCats = tFeats.izeros(t, tFeats.ncols)
		treeSearch(treesArray, treesArrayF, tFeats, newTreePos, treeCats)
		return voteForBestCategoriesAcrossTrees(treeCats)
	}

	private def voteForBestCategoriesAcrossTrees(treeCats : Mat) : Mat = {
		val treeCatsT = treeCats.t
		println("TREECATS NUM GREATER OR EQUAL TO 2")
 		println(sum(sum(treeCatsT >= numCats, 1), 2))
 		println("TREECATS TOT")
 		println(treeCatsT.length)
		val newTreeCatsT = markWithValueIfGreaterThan(treeCatsT, 0, numCats) // sometimes treeCats has values aren't actually categories. so mark them with zeros
		val accumedTreeCats = accumG(newTreeCatsT, 2, numCats)
		var bundle : (Mat, Mat) = null
		(accumedTreeCats) match {
			case (acTC : IMat) => {
				bundle = maxi2(acTC, 2)
			}
		}
		val majorityVal = bundle._1
		val majorityIndicies = bundle._2
		majorityIndicies.t
	}

	private def markWithValueIfGreaterThan(aMat : Mat, valToMark : Int, threshold : Int) : Mat = {
		val mask = (aMat >= threshold) // what to Mark
		val conjMask = 1 - mask
		mask *@ (valToMark * iones(aMat.nrows, aMat.ncols)) + conjMask *@ aMat
	}

	private def accumG(a : Mat, dim : Int, numBuckets : Int)  : Mat = {
		(dim, a) match {
			case (1, aMat : IMat) => {
				// col by col
				null
			}
			case (2, aMat : IMat) => {
				val iData = (icol(0->aMat.nrows) * iones(1, aMat.ncols)).data
				val i = irow(iData)
				val j = irow(aMat.data)
				val ij = i on j
				val out = accum(ij.t, 1, null, a.nrows, scala.math.max(a.ncols, numBuckets))
				out
			}
		}
	} 

	private def markAllCurPositionsAsLeavesAndCategorizeThem(tArray : Mat, tAFG : Mat, tPos : Mat) {
		val c = new CPUEntropyEval(oTreeVal, cats, d, d, impurityType, gainThreshold)
		var curT = 0
		while (curT < t) {
			tAFG(0,  tPos(curT, 0 -> n) + curT * nnodes) = scala.Float.NegativeInfinity * iones(1, n)
 			c.categorize(tPos, oTreeVal, tArray) 
			curT = curT + 1
		}
	}

	def treeSearch(treesArray : Mat, treesArrayF : Mat, testFeats : Mat, newTreePos : Mat, treeCats : Mat) {
		(treesArray, treesArrayF, testFeats, newTreePos, treeCats) match {
			case (tA : IMat, tAF : FMat, tfs : FMat, tP : IMat, tCats : IMat) => {
				treeSearch(tA, tAF, tfs, tP, tCats, 0, d, true, true) 
			} 
		}
	}
	
	def treeSearch(treesArray : IMat, treesArrayF : FMat, tFS : FMat, treePos : IMat, treePos2 : IMat, curIter : Int, totalIterations : Int, isTreeSteps : Boolean, isTreeSearch : Boolean) {
		if (curIter >= totalIterations) {
			return
		}
		val isAtLastIteration = ( curIter == (totalIterations - 1))
		val t = treePos.nrows
		val n = treePos.ncols
		val nnodes = treesArray.ncols / t
		val ns = treesArray.nrows - 1
		var tt = 0
		while (tt < t) {
			var curNodePos = 0
			var treesArrayIndex = 0 // corresponding index of nodePos in treesArray
			var threshold = 0f
			var nn = 0
			while (nn < n) {
				curNodePos = treePos(tt, nn)
				treesArrayIndex = tt * nnodes + curNodePos
				threshold = treesArrayF(0, treesArrayIndex)
				var isAtLeaf = (threshold == scala.Float.NegativeInfinity)
				if (isAtLastIteration && isTreeSearch) {
					var category = treesArray(1, treesArrayIndex)
					treePos2(tt, nn) = category
				} else if (!isAtLeaf) {
					var nsns = 0
					var featIndex = 0
					var curSum = 0f
					while (nsns < ns) {
						featIndex = treesArray(nsns + 1, treesArrayIndex)
						curSum += (tFS(featIndex, nn))
						nsns += 1
					}
					if (!isTreeSteps) {
					} else {
						if (curSum > threshold) {
							treePos2(tt, nn) = 2 * curNodePos + 2
						} else {
							treePos2(tt, nn) = 2 * curNodePos + 1
						}
					}
				}
				nn += 1
			}
			tt += 1
		}
		treeSearch(treesArray, treesArrayF , tFS, treePos2, treePos2, curIter + 1, totalIterations, isTreeSteps, isTreeSearch)
	}

	def treeProd(treesArray : Mat, treesArrayF : Mat, feats : Mat, treePos : Mat, oTreeVal : Mat) {
		(treesArray, treesArrayF, feats, treePos, oTreeVal) match {
			case (tA: IMat, tAF : FMat, fs : FMat, tP : IMat, oTV : FMat) => {
				treeProd(tA, tAF, fs, tP, oTV, null, false)
			}
			case (tA: IMat, tAF : FMat, fs : FMat, tP : IMat, tP2 : IMat) => {
				treeProd(tA, tAF, fs, tP, null, tP2, true)
			}
		}
	}

	def treeProd(treesArray : IMat, treesArrayF : FMat, feats : FMat, treePos : IMat, oTreeVal : FMat, treePos2 : IMat, isTreeSteps : Boolean) {
		val t = treePos.nrows
		val n = treePos.ncols
		val nnodes = treesArray.ncols / t
		val ns = treesArray.nrows - 1
		var tt = 0
		while (tt < t) {
			var curNodePos = 0
			var treesArrayIndex = 0 // corresponding index of nodePos in treesArray
			var threshold = 0f
			var nn = 0
			while (nn < n) {
				curNodePos = treePos(tt, nn)
				treesArrayIndex = tt * nnodes + curNodePos
				threshold = treesArrayF(0, treesArrayIndex)
				var isAtLeaf = (threshold == scala.Float.NegativeInfinity)
				if (!isAtLeaf) {
					var nsns = 0
					var featIndex = 0
					var curSum = 0f
					while (nsns < ns) {
						featIndex = treesArray(nsns + 1, treesArrayIndex)
						curSum += (feats(featIndex, nn))
						nsns += 1
					}
					if (!isTreeSteps) {
						oTreeVal(tt, nn) = curSum 
					} else {
						if (curSum > threshold) {
							treePos2(tt, nn) = 2 * curNodePos + 2
						} else {
							treePos2(tt, nn) = 2 * curNodePos + 1
						}
					}
				}
				nn += 1
			}
			tt += 1
		}
	} 



}

/**
 * EntropyEval:
 * Given the current depth marks the treesArray with the right thresholds
 */
class CPUEntropyEval(oTreeVal : Mat, cats : Mat, d : Int, k : Int, impurityType : Int, gainThreshold : Float) {
	val useGPU = oTreeVal match {case oTV:GMat => true; case _ => false };
	val n = oTreeVal.ncols
	val t = oTreeVal.nrows;
	val newSortedIndices : IMat = iones(t, 1) * irow(0->n) // for new code
	val treeOffsets = oTreeVal.izeros(1,t)
	val nnodes = (math.pow(2, d) + 0.5).toInt
	val tree_nnodes = (math.pow(2, k) + 0.5).toInt;
	treeOffsets <-- (nnodes * icol(0->t))
	val c = cats.nrows;
	val eps = 1E-5.toFloat


	/******************************************************************************************************************************
	 * PUBLIC: categorize
	 ******************************************************************************************************************************/
	def categorize(treePos : Mat, oTreeVal : Mat, treesArray : Mat) {
		val sortedI = oTreeVal.izeros(t, n)
		sortedI <-- (newSortedIndices)
		val sortedIT = sortedI.t
		handleCategorize(treePos, oTreeVal, treeOffsets, sortedIT, cats, treesArray)
	}

	private def handleCategorize(tP: Mat, oTV : Mat, tO : Mat, sIT : Mat, cts : Mat, tA : Mat) {
		val o1 = getNewSortIndicesTTreePosTAndTreeValsT(sIT, tP, oTV, tO)
		val sTreePosT = o1._2
		val soTreeValT = o1._3

		var curT = 0
		while (curT < t) {
			val o2 = getCurTreePosCurTreeValAndAssociatedSortedCats(sIT, sTreePosT, soTreeValT, cts, tO, curT)
			val curTreePosesT = o2._1
			val curTreeValsT = o2._2
			val pctsts = o2._3
			val fullJCForCurTree = getJCSegmentationForFullTree(curTreePosesT)
			markBestCategories(tP, pctsts, fullJCForCurTree, tA, curT)
			curT += 1
		}
	}

	private def markBestCategories(tPos: Mat, pctsts : Mat, fullJCForCurTree : Mat, tA : Mat, curT : Int) {
		val accumPctst = CPUBIDMatHelpers.cumsumg(pctsts, fullJCForCurTree)
		val tempBundle = CPUBIDMatHelpers.maxg(accumPctst, fullJCForCurTree)
		val totCatsPerGroup = tempBundle._1
		val totCatsPerGroupIndicies = tempBundle._2
		var allBestCatsBundle : (Mat, Mat) = null
		(totCatsPerGroup) match {
			case (tCPG : FMat) => {
				allBestCatsBundle = maxi2(tCPG.t)
			}
		}
		val allBestCatsVals = allBestCatsBundle._1
		val allBestCats = allBestCatsBundle._2
		val allBestCatsT = allBestCats.t
	 	val filteredBestCats = allBestCats(0, tPos(curT, 0->n))
		tA(1, tPos(curT, 0 -> n) + nnodes*curT) = filteredBestCats
	}


	/******************************************************************************************************************************
	 * PUBLIC: getThresholdsAndUpdateTreesArray
	 ******************************************************************************************************************************/
	def getThresholdsAndUpdateTreesArray(treePos : Mat, oTreeVal : Mat, treesArray : Mat, treesArrayFG : Mat) {
		val sortedI = oTreeVal.izeros(t, n)
		sortedI <-- (newSortedIndices)
		val sortedIT = sortedI.t
		handleGetThresholdsAndUpdateTreesArray(treePos, oTreeVal, treeOffsets, sortedIT, cats, treesArray, treesArrayFG)
	}

	private def handleGetThresholdsAndUpdateTreesArray(tP: Mat, oTV : Mat, tO : Mat, sIT : Mat, cts : Mat, tA : Mat, tAFG : Mat) {
		val o1 = getNewSortIndicesTTreePosTAndTreeValsT(sIT, tP, oTV, tO)
		val sTreePosT = o1._2
		val soTreeValT = o1._3

		var curT = 0
		while (curT < t) {
			val o2 = getCurTreePosCurTreeValAndAssociatedSortedCats(sIT, sTreePosT, soTreeValT, cts, tO, curT)
			val curTreePosesT = o2._1
			val curTreeValsT = o2._2
			val pctsts = o2._3
			val fullJCForCurTree = getJCSegmentationForFullTree(curTreePosesT)
			val fullImpurityReductions = calcImpurityReduction(pctsts, fullJCForCurTree, curTreePosesT)
			markThresholdsGivenReductions(fullImpurityReductions, curTreeValsT, tA, tAFG, fullJCForCurTree, curT)
			curT += 1
		}
	}

	private def getNewSortIndicesTTreePosTAndTreeValsT(sIT : Mat, tP : Mat, oTV : Mat, tO : Mat) : (Mat, Mat, Mat) = {
		var sTreePos = tP + 0
		var sTreePosT = sTreePos.t + tO
		val soTreeVal  = oTV + 0f // t, n
		val soTreeValT = soTreeVal.t // n x t
		lexsort2i(sTreePosT, soTreeValT, sIT)

		(sIT, sTreePosT, soTreeValT)
	}

	private def getCurTreePosCurTreeValAndAssociatedSortedCats(sIT : Mat, sTreePosT : Mat, soTreeValT : Mat, cts : Mat, tO : Mat, curT : Int) : (Mat, Mat, Mat) = {
		val curOffset = tO(0 -> 1, curT) // hack
		val curTreePosesTTemp = sTreePosT(0->n, curT) 
		var curTreePosesT = curTreePosesTTemp - curOffset
		val curTreeIndicesT = sIT(0->n, curT)
		val curTreeValsT = soTreeValT(0->n, curT)
		val pcats = CPUBIDMatHelpers.icopyT(curTreeIndicesT, cts)
		(curTreePosesT, curTreeValsT, pcats)
	}

	def getJCSegmentationForFullTree(curTreePoses : Mat) : Mat = {
		val jcTemp = accum(curTreePoses, 1, null, nnodes, 1)
		val jc = 0 on CPUBIDMatHelpers.cumsumg(jcTemp, 0 on nnodes)
		jc
	}

	def markThresholdsGivenReductions(impurityReductions : Mat, curTreeValsT : Mat, tA : Mat, tAFG : Mat, fullJC : Mat, curT : Int) {
		val partialJC = fullJC(((tree_nnodes -1) until (2*tree_nnodes)), 0)
		val mxsimp = CPUBIDMatHelpers.maxg(impurityReductions, partialJC)
		val maxes = mxsimp._1
		val maxis = mxsimp._2
		val newMaxis = getNewMaxis(mxsimp) // helprs marks some things as being leaves if their impurtiy is too little
		var tempMaxis =  newMaxis + 1
		val tempcurTreeValsT = impurityReductions.zeros(1 + curTreeValsT.nrows, 1) 
		tempcurTreeValsT <-- (scala.Float.NegativeInfinity on curTreeValsT)
		val maxTreeProdVals = tempcurTreeValsT(tempMaxis, 0)
		markTreeProdVals(tA, tAFG, maxTreeProdVals, tree_nnodes, nnodes, curT)
	}

	// METHOD: based on things that are in maxes which are smaller than a certain value, go look at maxis and then mask out the ones that are not 
	private def getNewMaxis(mxsimp : (Mat, Mat)) : Mat = {
		val maxes = mxsimp._1
		val maxis = mxsimp._2
		// val mask = izeros(maxes.nrows, maxes.ncols) // keeping this line uncommented means that this method is not doing anything
		val mask = maxes <= gainThreshold //what to mark with -1; uncomment if you want this method to activate
		val conjMask = 1 - mask // what to keep as the same for maxis
		val newMaxis = conjMask *@ maxis + mask *@ (-1 * iones(maxis.nrows, maxis.ncols))
		IMat(newMaxis)
	}

	private def markTreeProdVals(tA: Mat, tAFG : Mat, maxTreeProdVals : Mat, tree_nnodes : Int, nnodes : Int, curT : Int) {
		val indiciesToMark = (nnodes * curT + tree_nnodes -1)->(nnodes * curT + 2*tree_nnodes - 1) //GIMat((nnodes * curT + tree_nnodes -1)->(nnodes * curT + 2*tree_nnodes - 1))
		tAFG(0, indiciesToMark) = maxTreeProdVals.t
	}

	private def markNegOnesAsZero(a : Mat) : Mat = {
		val mask = (a >= 0)
		val x = a.izeros(a.nrows, a.ncols)
		(a *@ mask)
	}

	private def calcImpurityReduction(pctsts : Mat, jc : Mat, curTreePoses : Mat) : Mat = {
		/** Left Impurity */
		val leftAccumPctsts = CPUBIDMatHelpers.cumsumg(pctsts, jc)
		val leftTotsT1 = sum(leftAccumPctsts, 2)
		val leftTotsT2 = leftAccumPctsts.zeros(1, leftAccumPctsts.ncols) + 1
		val leftTots = leftTotsT1 * leftTotsT2
		val leftImpurity = getImpurityFor(leftAccumPctsts, leftTots)
		
		/** Total Impurity*/
		val totsTemps = jc(1 -> jc.length, 0)
		var totsTempMinusOne = totsTemps - 1
		totsTempMinusOne = markNegOnesAsZero(totsTempMinusOne)
		val totsAccumPctstsTemps = leftAccumPctsts(totsTempMinusOne, 0->leftAccumPctsts.ncols)
		var totTots = totsTemps(curTreePoses, 0) * (leftAccumPctsts.zeros(1, leftAccumPctsts.ncols) + 1)
		val totsAccumPctsts = totsAccumPctstsTemps(curTreePoses, 0->leftAccumPctsts.ncols)
		val totsImpurity = getImpurityFor(totsAccumPctsts, totTots)

		/** Right Total Impurity */
		val rightTots = totTots - leftTots
		val rightAccumPctsts = totsAccumPctsts - leftAccumPctsts
		val rightImpurity = getImpurityFor(rightAccumPctsts, rightTots)

		val impurityReduction = (leftImpurity + rightImpurity) - totsImpurity 
		val summedImpurityReduction = sum(impurityReduction, 2)
		return summedImpurityReduction
	}

	private def getImpurityFor(accumPctsts : Mat, tots : Mat) : Mat = {
		(impurityType)  match {
			case (1) => {
				getImpurityForInfoGain(accumPctsts, tots)
			}
			case (2) => {
				getImpurityForGiniImpurityReduction(accumPctsts, tots)
			}
		}
	}

	private def getImpurityForInfoGain(accumPctsts : Mat, tots : Mat) : Mat = {
		val ps = (accumPctsts / (tots + eps)) + eps  
		val conjps = (1f - ps) + eps
		val impurity = -1f * ( ps *@ ln(ps) + (conjps *@ ln(conjps)))
		impurity 
	}

	
	private def getImpurityForGiniImpurityReduction(accumPctsts : Mat, tots : Mat) : Mat = {
		val ps = (accumPctsts / (tots + eps)) + eps  
		val conjps = (1f - ps) + eps
		val impurity = ps *@ conjps
		impurity 
	}

  	private def lexsort2i(a : Mat, b: Mat, i : Mat) {
    	(a, b, i) match {
      	case (aa: GIMat, bb: GMat, ii : GIMat) => GMat.lexsort2i(aa, bb, ii);
      	case (aa: IMat, bb: FMat, ii : IMat) => CPUBIDMatHelpers.lexsort2iCPU(aa, bb, ii)
    	}
  	}

}


class CPUBIDMatHelpers {

}

object CPUBIDMatHelpers {

	def icopyT(indices : Mat, in : Mat) : Mat = {
		(indices, in) match {
      		case (ind: GIMat, i: GMat) => icopyT(ind, i, null)
      		case (ind: IMat, i: FMat) => icopyT(ind, i, null)
    	}
	}

	def icopyT(indices : GIMat, in : GMat, omat : GMat) : GMat = {
		val n = in.ncols
		val c = in.nrows
		val out = GMat.newOrCheckGMat(in.ncols, in.nrows, omat, in.GUID, "icopyT".##)
		val err = CUMAT.icopyt(indices.data, in.data, out.data, n, c, n)
		out
	}

	def icopyT(indices : IMat, in : FMat, out : FMat) : FMat =  {
		val o = in(irow(0 until in.nrows), indices)
		o.t
	}

	def cumsumg(in : Mat, jc : Mat) : Mat = {
		(in, jc) match {
			case (i : GMat, j : GIMat) => {
				GMat.cumsumg(i, j, null)
			}
			case (i : FMat, j : IMat) => {
				cumsumg(i, j, null)
			}
			case (i : IMat, j : IMat) => {
				cumsumg(i, j, null)
			}
		}
	}

	def cumsumg(in : IMat,  jc : IMat, omat : IMat) : IMat = {
		if (jc.length < 2) {
			throw new RuntimeException("cumsumg error: invalid arguments")
		}
		val out = IMat.newOrCheckIMat(in.nrows, in.ncols, omat, in.GUID, jc.GUID, "cumsumg".##)
		var nc = 0
		while (nc < in.ncols) {
			var j = 0
			var start = 0
			var end = 0
			while (j < (jc.length - 1)) {
				var sumSoFar = 0
				start = jc(j, 0)
				end = jc(j + 1, 0)
				var gr = start
				while (gr < end) {
					sumSoFar += in(gr, nc)
					out(gr, nc) = sumSoFar
					gr += 1
				}
				j += 1
			}
			nc += 1
		}
		out 
	}

	def cumsumg(in : FMat,  jc : IMat, omat : FMat) : FMat = {
		if (jc.length < 2) {
			throw new RuntimeException("cumsumg error: invalid arguments")
		}
		val out = FMat.newOrCheckFMat(in.nrows, in.ncols, omat, in.GUID, jc.GUID, "cumsumg".##)
		var nc = 0
		while (nc < in.ncols) {
			var j = 0
			var start = 0
			var end = 0
			while (j < (jc.length - 1)) {
				var sumSoFar = 0f
				start = jc(j, 0)
				end = jc(j + 1, 0)
				var gr = start
				while (gr < end) {
					sumSoFar += in(gr, nc)
					out(gr, nc) = sumSoFar
					gr += 1
				}
				j += 1
			}
			nc += 1
		}
		out 
	}

	def maxg(in : Mat, jc : Mat) : (Mat, Mat) = {
		(in, jc) match {
			case (i : GMat, j : GIMat) => {
				GMat.maxg(i, j, null, null)
			}
			case (i : FMat, j : IMat) => {
				maxg(i, j, null, null)
			}
		}
	}

	def maxg(in : FMat, jc : IMat, omat : FMat, omati : IMat) : (FMat, IMat) = {
		if (jc.length < 2) {
			throw new RuntimeException("maxg error: invalid arguments")
		}
		val out = FMat.newOrCheckFMat(jc.length-1, in.ncols, omat, in.GUID, jc.GUID, "maxg".##)
		val outi = IMat.newOrCheckIMat(jc.length-1, in.ncols, omati, in.GUID, jc.GUID, "maxg_i".##)
		var nc = 0
		while (nc < in.ncols) {
			var j = 0
			var start = 0
			var end = 0
			while (j < (jc.length - 1)) {
				var maxSoFar = scala.Float.NegativeInfinity
				var maxiSoFar = -1
				start = jc(j, 0)
				end = jc(j + 1, 0)
				var gr = start
				while (gr < end) {
					if (in(gr, nc) > maxSoFar) {
						maxSoFar = in(gr, nc)
						maxiSoFar = gr
					}
					gr += 1
				}
				out(j, nc) = maxSoFar
				outi(j, nc) = maxiSoFar
				j += 1
			}
			nc += 1
		}
		(out, outi)
	}

	def lexsort2iCPU(a : IMat, b : FMat, i : IMat) = {
		lexsort2iArr(a.data, b.data, i.data)
	}

	private def lexsort2iArr(a:Array[Int], b:Array[Float], i:Array[Int]) = {

		def comp(i1 : Int, i2 : Int) : Int = {
			val a1 = a(i1)
			val a2 = a(i2)
			val b1 = b(i1)
			val b2 = b(i2)
			if (compareInt(a1, a2) == 0) {
				return compareFloat(b1, b2)
			} else {
				return compareInt(a1, a2)
			}
		}
		def swap(i1 : Int, i2 : Int) = {
			val tempA = a(i2)
			a(i2) = a(i1)
			a(i1) = tempA
			val tempB = b(i2)
			b(i2) = b(i1)
			b(i1) = tempB
			val tempI = i(i2)
			i(i2) = i(i1)
			i(i1) = tempI
		}
		quickSort(comp, swap, 0, a.length)
	}

	private def compareInt(i : Int, j : Int) : Int = {
		if (i < j) {
			return -1
		} else if (i == j) {
			return 0
		} else {
			return 1
		}
	}

	private def compareFloat(i : Float, j : Float) : Int = {
		if (i < j) {
			return -1
		} else if (i == j) {
			return 0
		} else {
			return 1
		}
	}

}