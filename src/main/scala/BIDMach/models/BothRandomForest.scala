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
class BothRandomForest(d : Int, t: Int, ns: Int, fs : Mat, cs : Mat, impurityType : Int = 1, numCats : Int) {
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
	println("UseGPU: " + useGPU)
	val feats = fs
	val cats = cs
	val n = feats.ncols
	val f = feats.nrows
	val c = cats.nrows
	val nnodes = (math.pow(2, d) + 0.5).toInt
	println("nnodes: " + nnodes)
	/* Class Variable Matrices */
	val treePos = feats.izeros(t,n) //  GIMat.newOrCheckGIMat(t, n, null); 
	treePos.clear
	val treesArray = feats.izeros(ns + 1, t * nnodes)
	val treeTemp = IMat(f * rand(ns + 1, t * nnodes))
	min(treeTemp, f-1, treeTemp)
	treesArray <-- treeTemp
	val treesArrayF = feats.zeros(ns + 1, t * nnodes)
	var treesArrayG : GMat = null
	(treesArray) match {
		case (tA : GIMat) => {
			println("TREESARRAYG")
			treesArrayG = new GMat(tA.nrows, tA.ncols, tA.data, tA.length)
		}
		case _ => {}
	}
	println(treesArray)
	println(treesArrayG)
	val oTreePos = feats.izeros(t, n);
	val oTreeVal = feats.zeros(t, n)

	/** Made an instance variable so that this could be tested better */
	var e : BothEntropyEval = null
	
	/******************************************************************************************************************************
	 * PUBLIC: train
	 ******************************************************************************************************************************/
	def train {
		var k = 0;
		while (k < d - 1) { // d of them; each level
			println("At Depth: " + k);
			println("treePos at beginning")
			println(treePos)
			/* 
			calculate all the inner products 
			*/
   			println("Starting treeProd")
   			if (useGPU) {
   				treeProd(treesArray, treesArrayG, feats, treePos, oTreeVal);
   				println(treesArray)
   				println(treesArrayG)
   			} else {
   				treeProd(treesArray, treesArrayF, feats, treePos, oTreeVal);
   			}

			e = new BothEntropyEval(oTreeVal, cats, d, k, impurityType)
			if (useGPU) {
				e.newGetThresholdsAndUpdateTreesArray(treePos, oTreeVal, treesArray, treesArrayG)
				println(treesArray)
   				println(treesArrayG)
			} else {
				e.newGetThresholdsAndUpdateTreesArray(treePos, oTreeVal, treesArray, treesArrayF)
			}

			println("Starting TreeStep")
			if (useGPU) {
				treeProd(treesArray, treesArrayG, feats, treePos, oTreePos)
				println(treesArray)
   				println(treesArrayG)
			} else {
				treeProd(treesArray, treesArrayF, feats, treePos, oTreePos)
			}
			treePos <-- oTreePos
			k = k + 1
		}
		// mark last row all Leaves and Categorize them!
		println("BEGINNING TO MARK")
		if (useGPU) {
			markAllCurPositionsAsLeavesAndCategorizeThem(treesArray, treesArrayG, treePos)
		} else {
			markAllCurPositionsAsLeavesAndCategorizeThem(treesArray, treesArrayF, treePos)
		}
		
		println("treesArray after marking all current positions as leaves and categorizing them")
		println(treesArray)
		println(treesArrayG)
	}

	/******************************************************************************************************************************
	 * PUBLIC: classify
	 ******************************************************************************************************************************/
	def classify(feats : Mat) : Mat = {
		(feats) match {
			case (fs: GMat) => {
				val newTreePos = fs.izeros(t, fs.ncols);//  GIMat.newOrCheckGIMat(t, n, null); 
				newTreePos.clear
				val treeCats = feats.izeros(t, fs.ncols)
				treeCats.clear
				println("Yea: NewTreePos: " + newTreePos)
				GMat.treeSearch(treesArray, fs, newTreePos, treeCats)
				(treeCats) match {
					case (tCats : GIMat) => {
						return voteForBestCategoriesAcrossTrees(tCats)
					}
				}
			}
		}
	}

	/**
	 * returned is n x 1
	 */
	private def voteForBestCategoriesAcrossTrees(treeCats : GIMat) : GIMat = {
		val treeCatsT = treeCats.t
		val accumedTreeCats = accumG(treeCatsT, 2, numCats)
		println("fortTheVoteTreeCats")
		println(treeCats)
		println("accumedTreeCats")
		println(accumedTreeCats)
		val bundle = maxi2(accumedTreeCats, 2)
		val majorityVal = bundle._1
		val majorityIndicies = bundle._2
		println("majorityIndicies")
		println(majorityIndicies)
		majorityIndicies.t
	}

	private def accumG(a : GIMat, dim : Int, numBuckets : Int)  : GMat = {
		(dim) match {
			case (1) => {
				// col by col
				null
			}
			case (2) => {
				// row by row
				val iTemp = GIMat(icol(0->a.nrows) * iones(1, a.ncols))
				val i = reshape(iTemp, iTemp.length, 1)
				val j = reshape(a, a.length, 1)
				val ij = i \ j
				val omat = GMat.accum(ij, 1, null, a.nrows , numBuckets)
				omat
			}
		}
	} 

	private def reshape(a : GIMat, newRows : Int, newCols : Int) : GIMat = {
		val omat =  new GIMat(newRows, newCols, a.data, a.length)
		omat
	}
	/**
	 *
	 * Mark all current positions as Leaves
	 * TODO: Maybe mark and compute the categories too?
	 *
	 */
	private def markAllCurPositionsAsLeavesAndCategorizeThem(tArray : Mat, tAFG : Mat, tPos : Mat) {
	 	// (tArray, tAFG, tPos) match {
			// case (tA : GIMat, tArrFG : GMat, tPos : GIMat) => {
				// TODO: do a sort here?
		val c = new BothEntropyEval(oTreeVal, cats, d, d, impurityType) // TODO: Change name if good...
		var curT = 0
		while (curT < t) {
			if (useGPU) {
				tAFG(0,  GIMat(tPos(curT, 0 -> n)) + curT * nnodes) = scala.Float.NegativeInfinity * GMat(iones(1, n))
			} else {
				tAFG(0,  tPos(curT, 0 -> n) + curT * nnodes) = scala.Float.NegativeInfinity * iones(1, n)
			}
 			c.categorize(tPos, oTreeVal, tArray) 
			curT = curT + 1
		}
	 		// }
	 	// }
	}

	/******************************************************************************************************************************
	 * BOTH Code
	 ******************************************************************************************************************************/
	 def treeProd(treesArray : Mat, treesArrayF : Mat, feats : Mat, treePos : Mat, oTreeVal : Mat) {
		(treesArray, treesArrayF, feats, treePos, oTreeVal) match {
			case (tA: IMat, tAF : FMat, fs : FMat, tP : IMat, oTV : FMat) => {
				println("Before: treePos:\n" + treePos)
				println("Before: oTV:\n" + oTV)
				treeProd(tA, tAF, fs, tP, oTV, null, false)
				println("After: oTV:\n" + oTV)
				println("After: treePos:\n" + treePos)
			}
			case (tA: IMat, tAF : FMat, fs : FMat, tP : IMat, tP2 : IMat) => {
				println("Before: treePos:\n" + treePos)
				treeProd(tA, tAF, fs, tP, null, tP2, true)
				println("After: NewtreePos:\n" + oTreeVal)
			}
			case (tA: GIMat, tAF : GMat, fs : GMat, tP : GIMat, oTV : GMat) => {
				println("Before: treePos:\n" + treePos)
				println("Before: oTV:\n" + oTV)
				GMat.treeProd(tA, fs, tP, oTV)
				println("After: oTV:\n" + oTV)
				println("After: treePos:\n" + treePos)
			}
			case (tA: GIMat, tAF : GMat, fs : GMat, tP : GIMat, tP2 : GIMat) => {
				println("Before: treePos:\n" + treePos)
				GMat.treeProd(tA, fs, tP, tP2)
				println("After: NewtreePos:\n" + oTreeVal)
			}
		}
	}

	/******************************************************************************************************************************
	 * CPU Code
	 ******************************************************************************************************************************/
	def treeProd(treesArray : IMat, treesArrayF : FMat, feats : FMat, treePos : IMat, oTreeVal : FMat, treePos2 : IMat, isTreeSteps : Boolean) {
		val t = oTreeVal.nrows
		val n = oTreeVal.ncols
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
class BothEntropyEval(oTreeVal : Mat, cats : Mat, d : Int, k : Int, impurityType : Int) {
	val useGPU = oTreeVal match {case oTV:GMat => true; case _ => false };
	val n = oTreeVal.ncols
	val t = oTreeVal.nrows;
	val newSortedIndices : IMat = iones(t, 1) * irow(0->n) // for new code
	val sortedIndices : IMat = iones(1,1) * irow(0->n) //iones(t,1) * irow(0->n)
	val treeOffsets = oTreeVal.izeros(1,t)
	val nnodes = (math.pow(2, d) + 0.5).toInt
	val tree_nnodes = (math.pow(2, k) + 0.5).toInt;
	treeOffsets <-- (nnodes * icol(0->t)) 
	println(treeOffsets)
	val c = cats.nrows;
	// val pcatst = oTreeVal.zeros(cats.ncols, cats.nrows);
	var pctst = oTreeVal.zeros(cats.ncols, cats.nrows)
	val eps = 1E-5.toFloat


	/******************************************************************************************************************************
	 * PUBLIC: categorize
	 ******************************************************************************************************************************/
	def categorize(treePos : Mat, oTreeVal : Mat, treesArray : Mat) {
		val sortedI = oTreeVal.izeros(t, n)
		sortedI <-- (newSortedIndices)
		val sortedIT = sortedI.t
		(treePos, oTreeVal, treeOffsets, sortedIT, cats, treesArray) match {
				case (tP: GIMat, oTV : GMat, tO : GIMat, sIT : GIMat, cts : GMat, tA : GIMat) => {
					handleGPUCategorize(tP, oTV, tO, sIT, cts, tA)
				}
		}
	}

	private def handleGPUCategorize(tP: GIMat, oTV : GMat, tO : GIMat, sIT : GIMat, cts : GMat, tA : GIMat) {
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

	// should be ready
	private def markBestCategories(tPos: Mat, pctsts : Mat, fullJCForCurTree : Mat, tA : Mat, curT : Int) {
		println("pctsts")
		println(pctsts)
		println("fullJCForCurTree")
		println(fullJCForCurTree.t)
		val accumPctst = BIDMatHelpers.cumsumg(pctsts, fullJCForCurTree)
		println("accumPctst")
		println(accumPctst)
		println("accumPctst.nrows")
		println(accumPctst.nrows)
		println("accumPctst.ncols")
		println(accumPctst.ncols)
		val tempBundle = BIDMatHelpers.maxg(accumPctst, fullJCForCurTree)
		val totCatsPerGroup = tempBundle._1
		println("totCatsPerGroup.t")
		println(totCatsPerGroup.t)
		val totCatsPerGroupIndicies = tempBundle._2
		println("totCatsPerGroupIndicies.t")
		println(totCatsPerGroupIndicies.t)
		println("totCatsPerGroup.ncols")
		println(totCatsPerGroup.ncols)

		// TODO hack
		var allBestCatsBundle : (Mat, Mat) = null 
		(totCatsPerGroup) match {
			case (tCPG : FMat) => {
				allBestCatsBundle = maxi2(tCPG.t)
			}
			case (tCPG : GMat) => {
				allBestCatsBundle = maxi2(tCPG.t)
			}
		}

		val allBestCatsVals = allBestCatsBundle._1
		println("Seems like a problem is here when marking the categories")
		val allBestCats = allBestCatsBundle._2
		val allBestCatsT = allBestCats.t
		println("allBestCatsVals")
		println(allBestCatsVals)
		println("allBestCats")
		println(allBestCats)

	 	val filteredBestCats = allBestCats(0, tPos(curT, 0->n))
	 	println("tPos(curT, 0->n)")
	 	println(tPos(curT, 0->n))
	 	println("filteredBestCats")
	 	println(filteredBestCats)
	 	if (useGPU) {
			tA(1, GIMat(tPos(curT, 0 -> n)) + nnodes*curT) = filteredBestCats
		} else {
			tA(1, tPos(curT, 0 -> n) + nnodes*curT) = filteredBestCats
		}
	}


	/******************************************************************************************************************************
	 * PUBLIC: newGetThresholdsAndUpdateTreesArray
	 ******************************************************************************************************************************/
	def newGetThresholdsAndUpdateTreesArray(treePos : Mat, oTreeVal : Mat, treesArray : Mat, treesArrayFG : Mat) {
		val sortedI = oTreeVal.izeros(t, n)
		sortedI <-- (newSortedIndices)
		val sortedIT = sortedI.t
		handleGPUGetThresholdsAndUpdateTreesArray(treePos, oTreeVal, treeOffsets, sortedIT, cats, treesArray, treesArrayFG)
		// (treePos, oTreeVal, treeOffsets, sortedIT, cats, treesArray, treesArrayFG) match {
		// 		case (tP: GIMat, oTV : GMat, tO : GIMat, sIT : GIMat, cts : GMat, tA : GIMat, tAFG : GMat) => {
		// 			handleGPUGetThresholdsAndUpdateTreesArray(tP, oTV, tO, sIT, cts, tA, tAFG)
		// 		}
		// }
	}

	private def handleGPUGetThresholdsAndUpdateTreesArray(tP: Mat, oTV : Mat, tO : Mat, sIT : Mat, cts : Mat, tA : Mat, tAFG : Mat) {
		val o1 = getNewSortIndicesTTreePosTAndTreeValsT(sIT, tP, oTV, tO)
		val sTreePosT = o1._2
		val soTreeValT = o1._3

		var curT = 0
		while (curT < t) {
			println("handleGPUGetThresholdsAndUpdateTreesArray")
			println("WE ARE ON DEPTH # " + k + " AND TREE #" + curT)
			// val (dmy, freebytes, allbytes) = GPUmem
			// println("dmy: " + dmy + " freebytes: " + freebytes + " allbytes: " + allbytes)
			val o2 = getCurTreePosCurTreeValAndAssociatedSortedCats(sIT, sTreePosT, soTreeValT, cts, tO, curT)
			val curTreePosesT = o2._1
			val curTreeValsT = o2._2
			val pctsts = o2._3
			println("curTreePoses")
			println(curTreePosesT.t)
			println("curTreeVals")
			println(curTreeValsT.t)
			val fullJCForCurTree = getJCSegmentationForFullTree(curTreePosesT)
			val fullImpurityReductions = calcImpurityReduction(pctsts, fullJCForCurTree, curTreePosesT)
			markThresholdsGivenReductions(fullImpurityReductions, curTreeValsT, tA, tAFG, fullJCForCurTree, curT)
			curT += 1
		}
	}

	private def getNewSortIndicesTTreePosTAndTreeValsT(sIT : Mat, tP : Mat, oTV : Mat, tO : Mat) : (Mat, Mat, Mat) = {
		/* Make Copies of TreePos and TreeVals*/
		var sTreePos : Mat = null
		var sTreePosT : Mat = null
		(tP, tO) match {
			case (t : GIMat, tOff : GIMat) => {
				val tPlusZ : GIMat = t + 0
				val tPlusZT : GIMat = tPlusZ.t
				sTreePos = tPlusZ
				sTreePosT = tPlusZT + tOff		
			}
			case _ => {
				sTreePos = tP + 0
				sTreePosT = sTreePos.t + tO  // n x t
			}
		}
		val soTreeVal  = oTV + 0f // t, n
		val soTreeValT = soTreeVal.t // n x t
		println("getNewSortIndicesTTreePosTAndTreeValsT")
		println("WE ARE ON DEPTH # " + k)
		// val (dmy, freebytes, allbytes) = GPUmem
		// println("dmy: " + dmy + " freebytes: " + freebytes + " allbytes: " + allbytes)

		/* Sort it! */
		lexsort2i(sTreePosT, soTreeValT, sIT)

		(sIT, sTreePosT, soTreeValT)
	}

	private def getCurTreePosCurTreeValAndAssociatedSortedCats(sIT : Mat, sTreePosT : Mat, soTreeValT : Mat, cts : Mat, tO : Mat, curT : Int) : (Mat, Mat, Mat) = {
		println("sTreePosT")
		println(sTreePosT)
		val curOffset = tO(0 -> 1, curT) // hack
		val curTreePosesTTemp = sTreePosT(0->n, curT) 
		var curTreePosesT : Mat = null
		(curTreePosesTTemp, curOffset) match {
			case (cTPTTemp : GIMat, cO : GIMat) => {
				curTreePosesT = cTPTTemp - cO
			}
		}
		val curTreeIndicesT = sIT(0->n, curT)
		val curTreeValsT = soTreeValT(0->n, curT)
		println("curTreePosesT before icopyT")
		println(curTreePosesT.t)
		println("curTreeValsT before icopyT")
		println(curTreeValsT.t)
		println("cts")
		println(cts)
		val (dmy, freebytes, allbytes) = GPUmem
		println("dmy: " + dmy + " freebytes: " + freebytes + " allbytes: " + allbytes)
		val pcats = BIDMatHelpers.icopyT(curTreeIndicesT, cts)
		val (dmy1, freebytes1, allbytes1) = GPUmem
		println("dmy: " + dmy1 + " freebytes: " + freebytes1 + " allbytes: " + allbytes1)
		// CUMAT.icopyt(curTreeIndicesT.data, cts.data, pctst.data, n, c, n)
		println("pcats.t")	
		println(pcats.t)	
		(pcats) match {
			case (p : GMat) => {
				(curTreePosesT, curTreeValsT, p)
			}
		}
	}

	def getJCSegmentationForFullTree(curTreePoses : Mat) : Mat = {
		(curTreePoses) match {
			case (cTP : GIMat) => {
				val jcTemp : GMat = GMat.accum(cTP, 1, null, nnodes, 1)
				println("jcTemp.t")
				println(jcTemp.t)
				val jc = GIMat(IMat(0 on FMat(cumsumg(jcTemp, GIMat(0 on nnodes))))) // TODO: HACK
				println("calculated jc.t")
				println(jc.t)
				jc
			}
		}
	}

	// should be ready
	def markThresholdsGivenReductions(impurityReductions : Mat, curTreeValsT : Mat, tA : Mat, tAFG : Mat, fullJC : Mat, curT : Int) {
		println("fullJC")
		println(fullJC.t)
		val partialJC = fullJC(((tree_nnodes -1) until (2*tree_nnodes)), 0)
		println("partialJC")
		println(partialJC.t)
		val mxsimp = BIDMatHelpers.maxg(impurityReductions, partialJC)
		println("mxsimp")
		println(mxsimp)
		val maxes = mxsimp._1
		val maxis = mxsimp._2

		var tempMaxis : Mat = null
		(maxis) match {
			case (m : GIMat) => {
				tempMaxis =  m + 1
			}
			case _ => {
				tempMaxis =  maxis + 1
			}
		} 
		println("tempMaxis")
		println(tempMaxis)
		val tempcurTreeValsT = impurityReductions.zeros(1 + curTreeValsT.nrows, 1) 
		tempcurTreeValsT <-- (scala.Float.NegativeInfinity on curTreeValsT)
		val maxTreeProdVals = tempcurTreeValsT(tempMaxis, 0)
		markTreeProdVals(tA, tAFG, maxTreeProdVals, tree_nnodes, nnodes, curT)
	}

	private def markTreeProdVals(tA: Mat, tAFG : Mat, maxTreeProdVals : Mat, tree_nnodes : Int, nnodes : Int, curT : Int) {
		val indiciesToMark = GIMat((nnodes * curT + tree_nnodes -1)->(nnodes * curT + 2*tree_nnodes - 1))
		println("Indicies to Mark: ")
		println(indiciesToMark)
		println("What to Mark: ")
		println(maxTreeProdVals.t)
		println(tAFG)
		tAFG(0, indiciesToMark) = maxTreeProdVals.t
		println("TreesArrays: Marked with TreeProds")
		println(tAFG)
		println(tA)
	}

	private def calcImpurityReduction(pctsts : Mat, jc : Mat, curTreePoses : Mat) : Mat = {
		println("calcImpurityReduction")

		/** Left Total Impurity */
		println("LEFT IMPURITY STUFF")
		val leftAccumPctsts = BIDMatHelpers.cumsumg(pctsts, jc)
		// println("Saving Matrix for CumSumG")
		// saveAs("/home/derrick/code/NewRandomForest/BIDMach/tests/leftImpurityCumSumG.mat", FMat(pctsts), "pctsts", IMat(jc), "jc", FMat(leftAccumPctsts), "leftAccumPctsts")
		// while(true) {

		// };
		val leftTotsT1 = sum(leftAccumPctsts, 2)
		val leftTotsT2 = leftAccumPctsts.zeros(1, leftAccumPctsts.ncols) + 1
		println("TEST MULTIPLICATION")
		println(leftTotsT1)
		println(leftTotsT2)
		// saveAs("/home/derrick/code/NewRandomForest/BIDMach/tests/multiply.mat", FMat(leftTotsT1), "leftTotsT1", FMat(leftTotsT2), "leftTotsT2")
		val leftTots = leftTotsT1 * leftTotsT2
		// val leftTots = sum(leftAccumPctsts, 2) * (leftAccumPctsts.zeros(1, leftAccumPctsts.ncols) + GMat(1))
		val leftImpurity = getImpurityFor(leftAccumPctsts, leftTots)
		
		/** Total Impurity*/
		println("TOTAL IMPURITY STUFF")
		val totsTemps = jc(1 -> jc.length, 0)
		println("totsTemps")
		println(totsTemps)
		var totsTempMinusOne : Mat = null
		(totsTemps) match {
			case (tTmp : GIMat) => {
				totsTempMinusOne = tTmp - 1
			}
			case _ => {
				totsTempMinusOne = totsTemps - 1
			}
		}
		val totsAccumPctstsTemps = leftAccumPctsts(totsTempMinusOne, 0->leftAccumPctsts.ncols)
		println("totsAccumPctstsTemps")
		println(totsAccumPctstsTemps)
		var totTots : Mat = null
		if (useGPU) {
			totTots = GMat(totsTemps(curTreePoses, 0)) * (leftAccumPctsts.zeros(1, leftAccumPctsts.ncols) + 1) 
		} else {
			totTots = totsTemps(curTreePoses, 0) * (leftAccumPctsts.zeros(1, leftAccumPctsts.ncols) + 1)  // GMat(totsTemps(curTreePoses, GIMat(0))) * (leftAccumPctsts.zeros(1, leftAccumPctsts.ncols) + 1)

		}
		println("totTots")
		println(totTots)
		val totsAccumPctsts = totsAccumPctstsTemps(curTreePoses, 0->leftAccumPctsts.ncols)  //* (leftAccumPctsts.zeros(1, leftAccumPctsts.ncols) + GMat(1))
		println("totsAccumPctsts")
		println(totsAccumPctsts)
		val totsImpurity = getImpurityFor(totsAccumPctsts, totTots)

		/** Right Total Impurity */
		println("RIGHT IMPURITY STUFF")
		val rightTots = totTots - leftTots
		val rightAccumPctsts = totsAccumPctsts - leftAccumPctsts
		val rightImpurity = getImpurityFor(rightAccumPctsts, rightTots)

		val impurityReduction = totsImpurity - leftImpurity - rightImpurity
		println("ImpurityReduction")
		println(impurityReduction)
		println("ImpurityReduction Summed ")
		val summedImpurityReduction = sum(impurityReduction, 2)
		println(summedImpurityReduction)
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
		println("getImpurityForInfoGain")
		println("accumPctsts")
		println(accumPctsts)
		println("tots")
		println(tots)
		val ps = (accumPctsts / (tots + eps)) + eps  
		val conjps = (1f - ps) + eps
		println("conjps")
		println(conjps)
		println("ps")
		println(ps)
		val impurity = -1f * ( ps *@ ln(ps) + (conjps *@ ln(conjps)))
		println("impurity")
		println(impurity)
		impurity 
	}

	
	private def getImpurityForGiniImpurityReduction(accumPctsts : Mat, tots : Mat) : Mat = {
		println("getImpurityForGiniImpurityReduction")
		println("accumPctsts")
		println(accumPctsts)
		println("tots")
		println(tots)
		val ps = (accumPctsts / (tots + eps)) + eps  
		val conjps = (1f - ps) + eps
		println("conjps")
		println(conjps)
		println("ps")
		println(ps)
		val impurity = ps *@ conjps
		println("impurity")
		println(impurity)
		impurity 
	}

  	private def lexsort2i(a : Mat, b: Mat, i : Mat) {
    	(a, b, i) match {
      	case (aa: GIMat, bb: GMat, ii : GIMat) => GMat.lexsort2i(aa, bb, ii);
      	case (aa: IMat, bb: FMat, ii : IMat) => lexsort2i(aa, bb, ii)
    	}
  	}


	 /******************************************************************************************************************************
	 * GPU Code
	 ******************************************************************************************************************************/


}

class BIDMatHelpers {

}

object BIDMatHelpers {

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
				println("GPU Version")
				GMat.cumsumg(i, j, null)
			}
			case (i : FMat, j : IMat) => {
				println("CPU Version")
				cumsumg(i, j, null)
			}
		}
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

	def lexsort2i(a : IMat, b : FMat, i : IMat) = {
		lexsort2iArr(a.data, b.data, i.data)
	}

	private def lexsort2iArr[@specialized(Int) IT, @specialized(Float) FT](a:Array[IT], b:Array[FT], i:Array[IT])(implicit ordI:Ordering[IT], ordF: Ordering[FT]) = {

		def comp(i1 : Int, i2 : Int) : Int = {
			val a1 = a(i1)
			val a2 = a(i2)
			val b1 = b(i1)
			val b2 = b(i2)
			if (ordI.compare(a1, a2) == 0) {
				return ordF.compare(b1, b2)
			} else {
				return ordI.compare(a1, a2)
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

}
