package BIDMach.models

import BIDMat.{SBMat,CMat,CSMat,DMat,Dict,IDict,FMat,GMat,GIMat,GSMat,HMat,IMat,Mat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMat.Solvers._
import edu.berkeley.bid.CUMAT

/**
 * CPURandomForest Implementation
 */
class CPURandomForest(d : Int, t: Int, ns: Int, feats : Mat, cats : Mat) {
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

		feats = f x n matrix representing the raw feature values (FMat)
		cats = c x n matrix representing 0/1 which categories (FMat)
		treePos = t x n matrix representing the which node a specific sample is on for all trees (IMat)
		treesArray = ns x (t * (2^d - 1)) matrix representing feature indices for each node (FMat)
		oTreeVal = t * n matrix representing the float inner products received from running the treeProd method (FMat)
	*/

	/* Class Variable Constants */
	val useGPU = feats match {case a:GMat => true; case _ => false };
	val n = feats.ncols;
	val f = feats.nrows;
	val c = cats.nrows;
	val nnodes = (math.pow(2, d) + 0.5).toInt; 
	
	/* Class Variable Matrices */
	val treePos = feats.izeros(t,n);//  GIMat.newOrCheckGIMat(t, n, null); 
	treePos.clear
	var treesArray = feats.zeros(ns + 1, t * nnodes);
	val treeTemp = FMat(IMat(f * rand(ns + 1, t * nnodes)));
	min(treeTemp, f-1, treeTemp);
	treesArray <-- treeTemp;
	val oTreePos = feats.izeros(t, n); 
	val oTreeVal = feats.zeros(t, n)

	def train {
		println("TreesArray:\n" + treesArray)
		println("Feats:\n" + feats)
		var k = 0
		while (k < d) {
			println("TreeProd: Values Stored")
			// CPURandomForest.treeProd(treesArray, feats, treePos, oTreeVal, false)
			// println("TreeProd: Next Pos Values Stored")
			CPURandomForest.treeProd(treesArray, feats, treePos, oTreePos, true)
			treePos <-- oTreePos
			k = k + 1
		}
	}

	object CPURandomForest {


		def treeProd(treesArray : Mat, feats : Mat, treePos : Mat, oTreeVal : Mat, isTreeSteps : Boolean) {
			(treesArray, feats, treePos, oTreeVal) match {
				case (tA: FMat, fs : FMat, tP : IMat, oTV : FMat) => {
					println("Before: treePos:\n" + treePos)
					println("Before: oTV:\n" + oTV)
					CPURandomForest.treeProd(tA, fs, tP, oTV, isTreeSteps)
					println("After: oTV:\n" + oTV)
					println("After: treePos:\n" + treePos)
				}
				case (tA: FMat, fs : FMat, tP : IMat, tP2 : IMat) => {
					println("Before: treePos:\n" + treePos)
					CPURandomForest.treeSteps(tA, fs, tP, tP2, isTreeSteps)
					println("After: NEWtreePos:\n" + oTreeVal)
				}
			}
		}

		def treeProd(treesArray : FMat, feats : FMat, treePos : IMat, oTreeVal : FMat, isTreeSteps : Boolean) {
			val t = oTreeVal.nrows
			val n = oTreeVal.ncols
			val nnodes = treesArray.ncols / t
			val ns = treesArray.nrows - 1
			var tt = 0
			while (tt < t) {
				var curNodePos = 0
				var treesArrayIndex = 0
				var treesArrayVals : FMat = null
				var threshold : Float = 0
				var nn = 0
				while (nn < n) {
					println("tt: " + tt + " nn: " + nn)
					var curTreeProdSum : Float = 0
					curNodePos = treePos(tt, nn)
					treesArrayIndex = tt * nnodes + curNodePos
					threshold = treesArray(0, treesArrayIndex)
					treesArrayVals = treesArray(1->(ns+1), treesArrayIndex -> (treesArrayIndex + 1))
					var ii = 0
					var refFeatVal : Float = 0
					while (ii < treesArrayVals.nrows) {
						refFeatVal = feats(IMat(treesArrayVals(ii, 0)), nn)(0,0)
						curTreeProdSum += refFeatVal
						ii = ii + 1
					}
					var isLeaf = (threshold == scala.Float.NegativeInfinity)
					if (!isTreeSteps) {
						if (!isLeaf) {
							oTreeVal(tt, nn) = curTreeProdSum 
						}
					} else {
						if (!isLeaf) {
							if (curTreeProdSum > threshold) {
								oTreeVal(tt, nn) = 2 * curNodePos + 2
							} else {
								oTreeVal(tt, nn) = 2 * curNodePos + 1
							}
						}
					}
					nn = nn + 1	
				}
				tt = tt + 1
			}
		}

		def treeSteps(treesArray : FMat, feats : FMat, treePos : IMat, oTreePos : IMat, isTreeSteps : Boolean) {
			val t = oTreeVal.nrows
			val n = oTreeVal.ncols
			val nnodes = treesArray.ncols / t
			val ns = treesArray.nrows - 1
			var tt = 0
			while (tt < t) {
				var curNodePos = 0
				var treesArrayIndex = 0
				var treesArrayVals : FMat = null
				var threshold : Float = 0
				var nn = 0
				while (nn < n) {
					println("tt: " + tt + " nn: " + nn)
					var curTreeProdSum : Float = 0
					curNodePos = treePos(tt, nn)
					treesArrayIndex = tt * nnodes + curNodePos
					threshold = treesArray(0, treesArrayIndex)
					treesArrayVals = treesArray(1->(ns+1), treesArrayIndex -> (treesArrayIndex + 1))
					var ii = 0
					var refFeatVal : Float = 0
					while (ii < treesArrayVals.nrows) {
						refFeatVal = feats(IMat(treesArrayVals(ii, 0)), nn)(0,0)
						curTreeProdSum += refFeatVal
						ii = ii + 1
					}
					var isLeaf = (threshold == scala.Float.NegativeInfinity)
					if (isTreeSteps) {
						if (!isLeaf) {
							if (curTreeProdSum > threshold) {
								oTreePos(tt, nn) = 2 * curNodePos + 2
							} else {
								oTreePos(tt, nn) = 2 * curNodePos + 1
							}
						}
					}
					nn = nn + 1	
				}
				tt = tt + 1
			}
		}


	}
}

class CPUHelpers {
}

object CPUHelpers {
		
	// CUMAT.icopyt(curTreeIndicesT.data, cts.data, pctst.data, n, c, n)
	// in = cats (c * n)
	def icopyT(indices : IMat, in : FMat) : FMat =  {
		val o = in(irow(0 until in.nrows), indices)
		o.t
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
			var sumSoFar = 0f
			while (j < (jc.length - 1)) {
				start = jc(j, 0)
				end = jc(j + 1, 0)
				var gr = start
				while (gr < end) {
					sumSoFar += in(gr, nc).toFloat
					out(gr, nc) = sumSoFar
					gr += 1
				}
				j += 1
			}
			nc += 1
		}
		out 
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
			var maxSoFar = scala.Float.NegativeInfinity
			var maxiSoFar = -1
			while (j < (jc.length - 1)) {
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

	// TODO: 
	def lexsort2i(a:IMat, b:FMat, i:IMat) {
		// val ab = GMat.embedmat(a,b)
		// val err = CUMAT.lsortk(ab.data, i.data, i.length, 1);
		// if (err != 0) throw new RuntimeException("lexsort2i error %d: " + cudaGetErrorString(err) format err);
		// GMat.extractmat(a, b, ab);
	}
	
}

// /**
//  * CPUEntropyEval:
//  * Given the current depth marks the treesArray with the right thresholds
//  */
// class CPUEntropyEval(oTreeVal : Mat, cats : Mat, d : Int, k : Int) {
// 	val n = oTreeVal.ncols
// 	val t = oTreeVal.nrows;
// 	val sortedIndices : IMat = iones(t,1) * irow(0->n)
// 	val treeOffsets = oTreeVal.izeros(1,t)
// 	val nnodes = (math.pow(2, d) + 0.5).toInt
// 	println("TreeOffsets")
// 	treeOffsets <-- (nnodes * icol(0->t))
// 	println(treeOffsets)
// 	val c = cats.nrows;
// 	val pcatst = oTreeVal.zeros(cats.ncols, cats.nrows);
// 	println("curdepth: " + k)

// 	/******************************************************************************************************************************
// 	 * PUBLIC: categorize
// 	 ******************************************************************************************************************************/
// 	/**
// 	def categorize(treePos : Mat, oTreeVal : Mat, treesArray : Mat) {
// 		val sortedI = oTreeVal.izeros(t, n)
// 		sortedI <-- (newSortedIndices)
// 		val sortedIT = sortedI.t
// 		(treePos, oTreeVal, treeOffsets, sortedIT, cats, pcatst, treesArray) match {
// 				case (tP: GIMat, oTV : GMat, tO : GIMat, sIT : GIMat, cts : GMat, pctst : GMat, tA : GIMat) => {
// 					handleGPUCategorize(tP, oTV, tO, sIT, cts, pctst, tA)
// 				}
// 		}
// 	}

// 	private def handleGPUCategorize(tP: GIMat, oTV : GMat, tO : GIMat, sIT : GIMat, cts : GMat, pctsts : GMat, tA : GIMat) {
// 		val o1 = getNewSortIndicesTTreePosTAndTreeValsT(sIT, tP, oTV, tO)
// 		val sTreePosT = o1._2
// 		val soTreeValT = o1._3

// 		var curT = 0
// 		while (curT < t) {
// 			val o2 = getCurTreePosCurTreeValAndAssociatedSortedCats(sIT, sTreePosT, soTreeValT, cts, pctsts, tO, curT)
// 			val curTreePosesT = o2._1
// 			val curTreeValsT = o2._2
// 			val fullJCForCurTree = getJCSegmentationForFullTree(curTreePosesT)
// 			markBestCategories(tP, pctsts, fullJCForCurTree, tA, curT)
// 			curT += 1
// 		}
// 	}
// 	*/

// 	private def markBestCategories(tPos: IMat, pctsts : FMat, fullJCForCurTree : IMat, tA : IMat, curT : Int) {
// 		println("pctsts")
// 		println(pctsts)
// 		println("fullJCForCurTree")
// 		println(fullJCForCurTree.t)
// 		val accumPctst = cumsumg(pctsts, fullJCForCurTree, null)
// 		println("accumPctst")
// 		println(accumPctst)
// 		println("accumPctst.nrows")
// 		println(accumPctst.nrows)
// 		println("accumPctst.ncols")
// 		println(accumPctst.ncols)
// 		val tempBundle = maxg(accumPctst, fullJCForCurTree)
// 		val totCatsPerGroup = tempBundle._1
// 		println("totCatsPerGroup.t")
// 		println(totCatsPerGroup.t)
// 		val totCatsPerGroupIndicies = tempBundle._2
// 		println("totCatsPerGroupIndicies.t")
// 		println(totCatsPerGroupIndicies.t)
// 		println("totCatsPerGroup.ncols")
// 		println(totCatsPerGroup.ncols)

// 		// TODO: HACKS
// 		// val allBestCatsBundle = maxg(totCatsPerGroup.t, GIMat(0 on totCatsPerGroup.ncols))
// 		val allBestCatsBundle = maxi2(totCatsPerGroup.t)
// 		val allBestCatsVals = allBestCatsBundle._1
// 		println("Seems like a problem is here when marking the categories")
// 		val allBestCats = allBestCatsBundle._2
// 		val allBestCatsT = allBestCats.t
// 		println("allBestCatsVals")
// 		println(allBestCatsVals)
// 		println("allBestCats")
// 		println(allBestCats)
// 		// println("filter off the -1 and, and get the associated indicies then mark those...")
// 		// val temp = IMat(100 * rand(1, n));
// 	 // 	val bestCats = pctsts.izeros(1, n)
// 	 // 	bestCats <-- temp
// 	 	val filteredBestCats = allBestCats(0, tPos(curT, 0->n))
// 	 	println("tPos(curT, 0->n)")
// 	 	println(tPos(curT, 0->n))
// 	 	println("filteredBestCats")
// 	 	println(filteredBestCats)
// 		tA(1,  tPos(curT, 0 -> n) + nnodes*curT) = filteredBestCats
// 	}

// 	/******************************************************************************************************************************
// 	 * PUBLIC: newGetThresholdsAndUpdateTreesArray
// 	 ******************************************************************************************************************************/
// 	def newGetThresholdsAndUpdateTreesArray(treePos : Mat, oTreeVal : Mat, treesArray : Mat) {
// 		val sortedI = oTreeVal.izeros(t, n)
// 		sortedI <-- (newSortedIndices)
// 		val sortedIT = sortedI.t
// 		(treePos, oTreeVal, treeOffsets, sortedIT, cats, pcatst, treesArray) match {
// 				case (tP: IMat, oTV : FMat, tO : IMat, sIT : IMat, cts : FMat, pctst : FMat, tA : IMat) => {
// 					handleCPUGetThresholdsAndUpdateTreesArray(tP, oTV, tO, sIT, cts, pctst, tA)
// 				}
// 		}
// 	}

// 	private def handleCPUGetThresholdsAndUpdateTreesArray(tP: GIMat, oTV : GMat, tO : GIMat, sIT : GIMat, cts : GMat, pctsts : GMat, tA : GIMat) {
// 		val o1 = getNewSortIndicesTTreePosTAndTreeValsT(sIT, tP, oTV, tO)
// 		val sTreePosT = o1._2
// 		val soTreeValT = o1._3

// 		var curT = 0
// 		while (curT < t) {
// 			println("handleCPUGetThresholdsAndUpdateTreesArray")
// 			println("WE ARE ON DEPTH # " + k + " AND TREE #" + curT)
// 			// val (dmy, freebytes, allbytes) = GPUmem
// 			// println("dmy: " + dmy + " freebytes: " + freebytes + " allbytes: " + allbytes)
// 			val o2 = getCurTreePosCurTreeValAndAssociatedSortedCats(sIT, sTreePosT, soTreeValT, cts, pctsts, tO, curT)
// 			val curTreePosesT = o2._1
// 			val curTreeValsT = o2._2
// 			println("curTreePoses")
// 			println(curTreePosesT.t)
// 			println("curTreeVals")
// 			println(curTreeValsT.t)
// 			val fullJCForCurTree = getJCSegmentationForFullTree(curTreePosesT)
// 			val fullImpurityReductions = calcImpurityReduction(pctsts, fullJCForCurTree, curTreePosesT)
// 			markThresholdsGivenReductions(fullImpurityReductions, curTreeValsT, tA, fullJCForCurTree, curT)
// 			curT += 1
// 		}
// 	}

// 	private def getNewSortIndicesTTreePosTAndTreeValsT(sIT : IMat, tP : IMat, oTV : FMat, tO : IMat) : (IMat, IMat, FMat) = {
// 		/* Make Copies of TreePos and TreeVals*/
// 		val sTreePos = (tP + 0) // t, n
// 		val sTreePosT : IMat = sTreePos.t + tO  // n x t
// 		val soTreeVal : FMat = (oTV + 0f) // t, n
// 		val soTreeValT : FMat = soTreeVal.t // n x t
// 		println("getNewSortIndicesTTreePosTAndTreeValsT")
// 		println("WE ARE ON DEPTH # " + k)
// 		// val (dmy, freebytes, allbytes) = GPUmem
// 		// println("dmy: " + dmy + " freebytes: " + freebytes + " allbytes: " + allbytes)

// 		/* Sort it! */
// 		lexsort2i(sTreePosT, soTreeValT, sIT)

// 		(sIT, sTreePosT, soTreeValT)
// 	}

// 	private def getCurTreePosCurTreeValAndAssociatedSortedCats(sIT : IMat, sTreePosT : IMat, soTreeValT : FMat, cts : FMat, pctst : FMat, tO : IMat, curT : Int) : (IMat, FMat) = {
// 		println("sTreePosT")
// 		println(sTreePosT)
// 		val curOffset = IMat(tO(0, curT))
// 		val curTreePosesTTemp = sTreePosT(0->n, curT) 
// 		val curTreePosesT = curTreePosesTTemp - curOffset
// 		val curTreeIndicesT = sIT(0->n, curT)
// 		val curTreeValsT = soTreeValT(0->n, curT)
// 		println("curTreePosesT before icopyT")
// 		println(curTreePosesT.t)
// 		println("curTreeValsT before icopyT")
// 		println(curTreeValsT.t)
// 		pctst.clear
// 		println("cts")
// 		println(cts)
// 		CUMAT.icopyt(curTreeIndicesT.data, cts.data, pctst.data, n, c, n)
// 		println("pctst.t")	
// 		println(pctst.t)	
// 		(curTreePosesT, curTreeValsT)
// 	}

// 	def getJCSegmentationForFullTree(curTreePoses : IMat) : GIMat = {
// 		val jcTemp : GMat = GMat.accum(curTreePoses, 1, null, nnodes, 1)
// 		println("jcTemp.t")
// 		println(jcTemp.t)
// 		val jc = IMat(IMat(0 on FMat(cumsumg(jcTemp, IMat(0 on nnodes))))) // TODO: HACK
// 		println("calculated jc.t")
// 		println(jc.t)
// 		jc
// 	}

// 	def markThresholdsGivenReductions(impurityReductions : FMat, curTreeValsT : FMat, tA : IMat, fullJC : IMat, curT : Int) {
// 		println("fullJC")
// 		println(fullJC.t)
// 		val partialJC = fullJC(((tree_nnodes -1) until (2*tree_nnodes)), 0)
// 		println("partialJC")
// 		println(partialJC.t)
// 		val mxsimp = maxg(impurityReductions, partialJC)
// 		println("mxsimp")
// 		println(mxsimp)
// 		val maxes = mxsimp._1
// 		val maxis = mxsimp._2

// 		val tempMaxis = maxis + IMat(1)
// 		println("tempMaxis")
// 		println(tempMaxis)
// 		val tempcurTreeValsT = impurityReductions.zeros(1 + curTreeValsT.nrows, 1) 
// 		tempcurTreeValsT <-- (scala.Float.NegativeInfinity on curTreeValsT)
// 		val maxTreeProdVals = tempcurTreeValsT(tempMaxis, 0)
// 		markTreeProdVals(tA, maxTreeProdVals, tree_nnodes, nnodes, curT)
// 	}

// 	private def markTreeProdVals(tA: IMat, maxTreeProdVals : FMat, tree_nnodes : Int, nnodes : Int, curT : Int) {
// 		val indiciesToMark = GIMat((nnodes * curT + tree_nnodes -1)->(nnodes * curT + 2*tree_nnodes - 1))
// 		println("Indicies to Mark: ")
// 		println(indiciesToMark)
// 		println("What to Mark: ")
// 		println(maxTreeProdVals.t)
// 		val tArray : FMat = new FMat(tA.nrows, tA.ncols, tA.data, tA.length)
// 		tArray(0, indiciesToMark) = maxTreeProdVals.t
// 		println("TreesArrays: Marked with TreeProds")
// 		println(tArray)
// 	}

// 	private def calcImpurityReduction(pctsts : GMat, jc : GIMat, curTreePoses : GIMat) : GMat = {
// 		println("calcImpurityReduction")

// 		/** Left Total Impurity */
// 		println("LEFT IMPURITY STUFF")
// 		val leftAccumPctsts = cumsumg(pctsts, jc, null)
// 		// println("Saving Matrix for CumSumG")
// 		// saveAs("/home/derrick/code/NewRandomForest/BIDMach/tests/leftImpurityCumSumG.mat", FMat(pctsts), "pctsts", IMat(jc), "jc", FMat(leftAccumPctsts), "leftAccumPctsts")
// 		// while(true) {

// 		// };
// 		val leftTotsT1 = sum(leftAccumPctsts, 2)
// 		val leftTotsT2 = leftAccumPctsts.zeros(1, leftAccumPctsts.ncols) + GMat(1)
// 		println("TEST MULTIPLICATION")
// 		println(leftTotsT1)
// 		println(leftTotsT2)
// 		// saveAs("/home/derrick/code/NewRandomForest/BIDMach/tests/multiply.mat", FMat(leftTotsT1), "leftTotsT1", FMat(leftTotsT2), "leftTotsT2")
// 		val leftTots = leftTotsT1 * leftTotsT2
// 		// val leftTots = sum(leftAccumPctsts, 2) * (leftAccumPctsts.zeros(1, leftAccumPctsts.ncols) + GMat(1))
// 		val leftImpurity = getImpurityFor(leftAccumPctsts, leftTots)
		
// 		/** Total Impurity*/
// 		println("TOTAL IMPURITY STUFF")
// 		val totsTemps = jc(1 -> jc.length, 0)
// 		println("totsTemps")
// 		println(totsTemps)
// 		val totsAccumPctstsTemps = leftAccumPctsts(totsTemps - GIMat(1), GIMat(0->leftAccumPctsts.ncols))
// 		println("totsAccumPctstsTemps")
// 		println(totsAccumPctstsTemps)
// 		val totTots = GMat(totsTemps(curTreePoses, GIMat(0))) * (leftAccumPctsts.zeros(1, leftAccumPctsts.ncols) + GMat(1))
// 		println("totTots")
// 		println(totTots)
// 		val totsAccumPctsts = totsAccumPctstsTemps(curTreePoses, GIMat(0->leftAccumPctsts.ncols))  //* (leftAccumPctsts.zeros(1, leftAccumPctsts.ncols) + GMat(1))
// 		println("totsAccumPctsts")
// 		println(totsAccumPctsts)
// 		val totsImpurity = getImpurityFor(totsAccumPctsts, totTots)

// 		/** Right Total Impurity */
// 		println("RIGHT IMPURITY STUFF")
// 		val rightTots = totTots - leftTots
// 		val rightAccumPctsts = totsAccumPctsts - leftAccumPctsts
// 		val rightImpurity = getImpurityFor(rightAccumPctsts, rightTots)

// 		val impurityReduction = totsImpurity - leftImpurity - rightImpurity
// 		println("ImpurityReduction")
// 		println(impurityReduction)
// 		println("ImpurityReduction Summed ")
// 		val summedImpurityReduction = sum(impurityReduction, 2)
// 		println(summedImpurityReduction)
// 		return summedImpurityReduction
// 	}

// 	private def getImpurityFor(accumPctsts : GMat, tots : GMat) : GMat = {
// 		(impurityType)  match {
// 			case (1) => {
// 				getImpurityForInfoGain(accumPctsts, tots)
// 			}
// 			case (2) => {
// 				getImpurityForGiniImpurityReduction(accumPctsts, tots)
// 			}
// 		}
// 	}

// 	private def getImpurityForInfoGain(accumPctsts : GMat, tots : GMat) : GMat = {
// 		println("getImpurityForInfoGain")
// 		println("accumPctsts")
// 		println(accumPctsts)
// 		println("tots")
// 		println(tots)
// 		val ps = (accumPctsts / (tots + eps)) + eps  
// 		val conjps = (1f - ps) + eps
// 		println("conjps")
// 		println(conjps)
// 		println("ps")
// 		println(ps)
// 		val impurity = -1f * ( ps *@ ln(ps) + (conjps *@ ln(conjps)))
// 		println("impurity")
// 		println(impurity)
// 		impurity 
// 	}

	
// 	private def getImpurityForGiniImpurityReduction(accumPctsts : GMat, tots : GMat) : GMat = {
// 		// add some e val to 
// 		println("getImpurityForGiniImpurityReduction")
// 		println("accumPctsts")
// 		println(accumPctsts)
// 		println("tots")
// 		println(tots)
// 		val ps = (accumPctsts / (tots + eps)) + eps  
// 		val conjps = (1f - ps) + eps
// 		println("conjps")
// 		println(conjps)
// 		println("ps")
// 		println(ps)
// 		val impurity = ps *@ conjps
// 		println("impurity")
// 		println(impurity)
// 		impurity 
// 	}

//   	private def lexsort2i(a : Mat, b: Mat, i : Mat) {
//     	(a, b, i) match {
//       	case (aa: GIMat, bb: GMat, ii : GIMat) => GMat.lexsort2i(aa, bb, ii)
//     	}
//   	}

// }



