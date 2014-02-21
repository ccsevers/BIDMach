package BIDMach.models

import BIDMat.{SBMat,CMat,CSMat,DMat,Dict,IDict,FMat,GMat,GIMat,GSMat,HMat,IMat,Mat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMat.Solvers._
import edu.berkeley.bid.CUMAT

/**
 * Random Forest Implementation
 */
class RandomForest(d : Int, t: Int, ns: Int, feats : Mat, cats : Mat, impurityType : Int = 1, numCats : Int) {
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
	val useGPU = feats match {case a:GMat => true; case _ => false };
	println("UseGPU: " + useGPU)
	val n = feats.ncols;
	val f = feats.nrows;
	val c = cats.nrows;
	val nnodes = (math.pow(2, d) + 0.5).toInt; 
	println("nnodes: " + nnodes)
	/* Class Variable Matrices */
	val treePos = feats.izeros(t,n);//  GIMat.newOrCheckGIMat(t, n, null); 
	treePos.clear
	var treesArray = feats.izeros(ns + 1, t * nnodes);
	var gTreesArray : GMat = null;
	// TODO: 
	(treesArray) match {
		case (tA : GIMat) => {
			gTreesArray = new GMat(tA.nrows, tA.ncols, tA.data, tA.length)
		} 
	}
	val treeTemp = IMat(f * rand(ns + 1, t * nnodes));
	min(treeTemp, f-1, treeTemp);
	treesArray <-- treeTemp;
	val oTreePos = feats.izeros(t, n); 
	val oTreeVal = feats.zeros(t, n)
	
	/******************************************************************************************************************************
	 * PUBLIC: train
	 ******************************************************************************************************************************/
	def train {
		var k = 0;
		for (k <- 0 until d - 1) { // d of them; each level
			println("At Depth: " + k);
			val (dmy, freebytes, allbytes) = GPUmem
			println("dmy: " + dmy + " freebytes: " + freebytes + " allbytes: " + allbytes)
			println("treePos at beginning")
			println(treePos)
			/* 
			calculate all the inner products 
			*/
			// treeprod(unsigned int *trees, float *feats, int *tpos, int *otpos, int nrows, int ncols, int ns, int tstride, int ntrees, int doth)			
			// treeprod(treesArray.data, feats.data, treePos.data, oTreeVal.data, t, n, ns, nnodes * (ns + 1), t, 0);
			// treeProd(treesArray, feats, treePos, oTreeVal, t, n, ns, nnodes * (ns + 1), t, 0);
			println("Classes: " + treesArray.getClass + " " + feats.getClass + " " + treePos.getClass + " " + oTreeVal.getClass);
   			println("Starting treeProd")
   			GMat.treeProd(treesArray, feats, treePos, oTreeVal);

			val e = new EntropyEval(oTreeVal, cats, d, k, impurityType)
			// e.getThresholdsAndUpdateTreesArray(treePos, oTreeVal, treesArray)
			e.newGetThresholdsAndUpdateTreesArray(treePos, oTreeVal, treesArray)

			println("Starting TreeStep")
			GMat.treeProd(treesArray, feats, treePos, treePos)
		}
		// mark last row all Leaves and Categorize them!
		markAllCurPositionsAsLeavesAndCategorizeThem(treesArray, treePos)
		println("treesArray after marking all current positions as leaves and categorizing them")
		println(treesArray)
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

				saveAs("/home/derrick/code/NewRandomForest/BIDMach/tests/classify.mat", IMat(treesArray), "treesArray", FMat(fs), "fs", IMat(newTreePos), "newTreePos", IMat(treeCats), "treeCats")
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
		// val treeCatsT = treeCats.t
		// val accumedCats = accumG(treeCatsT, 2, numCats)
		// println("accumedCats")
		// println(accumedCats)
		// val bundle = maxi2(FMat(accumedCats), 2)
		// val majorityVal = bundle._1
		// val majorityIndicies = bundle._2
		// GIMat(majorityIndicies)
		treeCats
	}

	private def accumG(a : GIMat, dim : Int, numBuckets : Int)  : GMat = {
		(dim) match {
			case (1) => {
				// col by col
				null
			}
			case (2) => {
				// row by row
				val aT = a.t
				val iTemp = GIMat(icol(0->a.nrows) * iones(1, a.ncols))
				val i = reshape(iTemp, iTemp.length, 1)
				val j = reshape(aT, aT.length, 1)
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
	private def markAllCurPositionsAsLeavesAndCategorizeThem(tArray : Mat, tPos : Mat) {
	 	(tArray, tPos) match {
			case (tA : GIMat, tPos : GIMat) => {
				// TODO: do a sort here?
				val c = new EntropyEval(oTreeVal, cats, d, d, impurityType) // TODO: Change name if good...
				val tArr : GMat = new GMat(tA.nrows, tA.ncols, tA.data, tA.length)
	 			var curT = 0
	 			while (curT < t) {
	 				tArr(0,  tPos(curT, 0 -> n) + curT * nnodes) = scala.Float.NegativeInfinity * GMat(iones(1, n))
		 			c.categorize(tPos, oTreeVal, tArray) 
	 				curT = curT + 1
	 			}
	 		}
	 	}
	}
}

/**
 * EntropyEval:
 * Given the current depth marks the treesArray with the right thresholds
 */
class EntropyEval(oTreeVal : Mat, cats : Mat, d : Int, k : Int, impurityType : Int) {
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
	val pcatst = oTreeVal.zeros(cats.ncols, cats.nrows);

	val eps = 1E-5.toFloat


	/******************************************************************************************************************************
	 * PUBLIC: categorize
	 ******************************************************************************************************************************/
	def categorize(treePos : Mat, oTreeVal : Mat, treesArray : Mat) {
		val sortedI = oTreeVal.izeros(t, n)
		sortedI <-- (newSortedIndices)
		val sortedIT = sortedI.t
		(treePos, oTreeVal, treeOffsets, sortedIT, cats, pcatst, treesArray) match {
				case (tP: GIMat, oTV : GMat, tO : GIMat, sIT : GIMat, cts : GMat, pctst : GMat, tA : GIMat) => {
					handleGPUCategorize(tP, oTV, tO, sIT, cts, pctst, tA)
				}
		}
	}

	private def handleGPUCategorize(tP: GIMat, oTV : GMat, tO : GIMat, sIT : GIMat, cts : GMat, pctsts : GMat, tA : GIMat) {
		val o1 = getNewSortIndicesTTreePosTAndTreeValsT(sIT, tP, oTV, tO)
		val sTreePosT = o1._2
		val soTreeValT = o1._3

		var curT = 0
		while (curT < t) {
			val o2 = getCurTreePosCurTreeValAndAssociatedSortedCats(sIT, sTreePosT, soTreeValT, cts, pctsts, tO, curT)
			val curTreePosesT = o2._1
			val curTreeValsT = o2._2
			val fullJCForCurTree = getJCSegmentationForFullTree(curTreePosesT)
			markBestCategories(tP, pctsts, fullJCForCurTree, tA, curT)
			curT += 1
		}
	}

	private def markBestCategories(tPos: GIMat, pctsts : GMat, fullJCForCurTree : GIMat, tA : GIMat, curT : Int) {
		println("pctsts")
		println(pctsts)
		println("fullJCForCurTree")
		println(fullJCForCurTree.t)
		val accumPctst = cumsumg(pctsts, fullJCForCurTree, null)
		println("accumPctst")
		println(accumPctst)
		println("accumPctst.nrows")
		println(accumPctst.nrows)
		println("accumPctst.ncols")
		println(accumPctst.ncols)
		val tempBundle = maxg(accumPctst, fullJCForCurTree)
		val totCatsPerGroup = tempBundle._1
		println("totCatsPerGroup.t")
		println(totCatsPerGroup.t)
		val totCatsPerGroupIndicies = tempBundle._2
		println("totCatsPerGroupIndicies.t")
		println(totCatsPerGroupIndicies.t)
		val allBestCatsBundle = maxg(totCatsPerGroup.t, GIMat(0 \ totCatsPerGroup.ncols))
		val allBestCatsVals = allBestCatsBundle._1
		val allBestCats = allBestCatsBundle._2
		val allBestCatsT = allBestCats.t
		println("allBestCatsVals")
		println(allBestCatsVals)
		println("allBestCats")
		println(allBestCats)
		// println("filter off the -1 and, and get the associated indicies then mark those...")
		// val temp = IMat(100 * rand(1, n));
	 // 	val bestCats = pctsts.izeros(1, n)
	 // 	bestCats <-- temp
	 	val filteredBestCats = allBestCats(0, tPos(curT, 0->n))
	 	println("tPos(curT, 0->n)")
	 	println(tPos(curT, 0->n))
	 	println("filteredBestCats")
	 	println(filteredBestCats)
		tA(1,  tPos(curT, 0 -> n) + nnodes*curT) = filteredBestCats

		// println("accumPctst")
		// println(accumPctst)
		// val tempBundle = maxg(accumPctst, fullJCForCurTree)
		// println("tempBundle._1.t")
		// println(tempBundle._1.t)
		// println("tempBundle._2.t")
		// println(tempBundle._2.t)
		// val totCatsPerGroupBundle = markWithBundleValue(tempBundle._1, tempBundle._2)
		// val totCatsPerGroup = tempBundle._1
		// val totCatsPerGroupIndicies = tempBundle._2
		// println("marking best categories")
		// println("totCatsPerGroup")
		// println(totCatsPerGroup.t)
		// println("totCatsPerGroupIndicies")
		// println(totCatsPerGroupIndicies.t)
	}


	/**
	 *
	 */
	// private def markWithBundleValue(v : GMat, vI : GIMat) : (GMat, GIMat) = {
	// 	val newVI = vI + GIMat(1)
	// 	val tempV = v.zeros(1 + v.nrows, v.ncols) 
	// 	tempV <-- (scala.Float.NegativeInfinity * ones(1, tempV.ncols) on v)
	// 	val newV = tempV(newVI, 0)
	// 	(newV, newVI)
	// }

	/******************************************************************************************************************************
	 * PUBLIC: newGetThresholdsAndUpdateTreesArray
	 ******************************************************************************************************************************/
	def newGetThresholdsAndUpdateTreesArray(treePos : Mat, oTreeVal : Mat, treesArray : Mat) {
		val sortedI = oTreeVal.izeros(t, n)
		sortedI <-- (newSortedIndices)
		val sortedIT = sortedI.t
		(treePos, oTreeVal, treeOffsets, sortedIT, cats, pcatst, treesArray) match {
				case (tP: GIMat, oTV : GMat, tO : GIMat, sIT : GIMat, cts : GMat, pctst : GMat, tA : GIMat) => {
					handleGPUGetThresholdsAndUpdateTreesArray(tP, oTV, tO, sIT, cts, pctst, tA)
				}
		}
	}

	private def handleGPUGetThresholdsAndUpdateTreesArray(tP: GIMat, oTV : GMat, tO : GIMat, sIT : GIMat, cts : GMat, pctsts : GMat, tA : GIMat) {
		val o1 = getNewSortIndicesTTreePosTAndTreeValsT(sIT, tP, oTV, tO)
		val sTreePosT = o1._2
		val soTreeValT = o1._3

		var curT = 0
		while (curT < t) {
			val o2 = getCurTreePosCurTreeValAndAssociatedSortedCats(sIT, sTreePosT, soTreeValT, cts, pctsts, tO, curT)
			val curTreePosesT = o2._1
			val curTreeValsT = o2._2
			println("curTreePoses")
			println(curTreePosesT.t)
			println("curTreeVals")
			println(curTreeValsT.t)
			val fullJCForCurTree = getJCSegmentationForFullTree(curTreePosesT)
			val fullImpurityReductions = calcImpurityReduction(pctsts, fullJCForCurTree, curTreePosesT)
			markThresholdsGivenReductions(fullImpurityReductions, curTreeValsT, tA, fullJCForCurTree, curT)
			curT += 1
		}
	}

	private def getNewSortIndicesTTreePosTAndTreeValsT(sIT : GIMat, tP : GIMat, oTV : GMat, tO : GIMat) : (GIMat, GIMat, GMat) = {
		/* Make Copies of TreePos and TreeVals*/
		val sTreePos : GIMat = (tP + GIMat(0)) // t, n
		val sTreePosT : GIMat = sTreePos.t + tO  // n x t
		val soTreeVal : GMat = (oTV + 0f) // t, n
		val soTreeValT : GMat = soTreeVal.t // n x t

		/* Sort it! */
		lexsort2i(sTreePosT, soTreeValT, sIT)

		(sIT, sTreePosT, soTreeValT)
	}

	private def getCurTreePosCurTreeValAndAssociatedSortedCats(sIT : GIMat, sTreePosT : GIMat, soTreeValT : GMat, cts : GMat, pctst : GMat, tO : GIMat, curT : Int) : (GIMat, GMat) = {
		println("sTreePosT")
		println(sTreePosT)
		val curOffset : GIMat = GIMat(tO(0, curT))
		val curTreePosesTTemp = sTreePosT(0->n, curT) 
		val curTreePosesT = curTreePosesTTemp - curOffset
		val curTreeIndicesT = sIT(0->n, curT)
		val curTreeValsT = soTreeValT(0->n, curT)
		println("curTreePosesT before icopyT")
		println(curTreePosesT.t)
		println("curTreeValsT before icopyT")
		println(curTreeValsT.t)
		pctst.clear
		println("cts")
		println(cts)
		CUMAT.icopyt(curTreeIndicesT.data, cts.data, pctst.data, n, c, n)
		println("pctst.t")	
		println(pctst.t)	
		(curTreePosesT, curTreeValsT)
	}

	def getJCSegmentationForFullTree(curTreePoses : GIMat) : GIMat = {
		val jcTemp : GMat = GMat.accum(curTreePoses, 1, null, nnodes, 1)
		println("jcTemp.t")
		println(jcTemp.t)
		val jc = GIMat(IMat(0 on FMat(cumsumg(jcTemp, GIMat(0 on nnodes))))) // TODO: HACK
		println("calculated jc.t")
		println(jc.t)
		jc
	}

	def markThresholdsGivenReductions(impurityReductions : GMat, curTreeValsT : GMat, tA : GIMat, fullJC : GIMat, curT : Int) {
		println("fullJC")
		println(fullJC.t)
		val partialJC = fullJC(((tree_nnodes -1) until (2*tree_nnodes)), 0)
		println("partialJC")
		println(partialJC.t)
		val mxsimp = maxg(impurityReductions, partialJC)
		println("mxsimp")
		println(mxsimp)
		val maxes = mxsimp._1
		val maxis = mxsimp._2

		val tempMaxis = maxis + GIMat(1)
		println("tempMaxis")
		println(tempMaxis)
		val tempcurTreeValsT = impurityReductions.zeros(1 + curTreeValsT.nrows, 1) 
		tempcurTreeValsT <-- (scala.Float.NegativeInfinity on curTreeValsT)
		val maxTreeProdVals = tempcurTreeValsT(tempMaxis, 0)
		markTreeProdVals(tA, maxTreeProdVals, tree_nnodes, nnodes, curT)
	}

	private def markTreeProdVals(tA: GIMat, maxTreeProdVals : GMat, tree_nnodes : Int, nnodes : Int, curT : Int) {
		val indiciesToMark = GIMat((nnodes * curT + tree_nnodes -1)->(nnodes * curT + 2*tree_nnodes - 1))
		println("Indicies to Mark: ")
		println(indiciesToMark)
		println("What to Mark: ")
		println(maxTreeProdVals.t)
		val tArray : GMat = new GMat(tA.nrows, tA.ncols, tA.data, tA.length)
		tArray(0, indiciesToMark) = maxTreeProdVals.t
		println("TreesArrays: Marked with TreeProds")
		println(tArray)
	}

	private def calcImpurityReduction(pctsts : GMat, jc : GIMat, curTreePoses : GIMat) : GMat = {
		println("calcImpurityReduction")

		/** Left Total Impurity */
		println("LEFT IMPURITY STUFF")
		val leftAccumPctsts = cumsumg(pctsts, jc, null)
		// println("Saving Matrix for CumSumG")
		// saveAs("/home/derrick/code/NewRandomForest/BIDMach/tests/leftImpurityCumSumG.mat", FMat(pctsts), "pctsts", IMat(jc), "jc", FMat(leftAccumPctsts), "leftAccumPctsts")
		// while(true) {

		// };
		val leftTotsT1 = sum(leftAccumPctsts, 2)
		val leftTotsT2 = leftAccumPctsts.zeros(1, leftAccumPctsts.ncols) + GMat(1)
		val leftTots = leftTotsT1 * leftTotsT2
		// val leftTots = sum(leftAccumPctsts, 2) * (leftAccumPctsts.zeros(1, leftAccumPctsts.ncols) + GMat(1))
		val leftImpurity = getImpurityFor(leftAccumPctsts, leftTots)
		
		/** Total Impurity*/
		println("TOTAL IMPURITY STUFF")
		val totsTemps = jc(1 -> jc.length, 0)
		println("totsTemps")
		println(totsTemps)
		val totsAccumPctstsTemps = leftAccumPctsts(totsTemps - GIMat(1), GIMat(0->leftAccumPctsts.ncols))
		println("totsAccumPctstsTemps")
		println(totsAccumPctstsTemps)
		val totTots = GMat(totsTemps(curTreePoses, GIMat(0))) * (leftAccumPctsts.zeros(1, leftAccumPctsts.ncols) + GMat(1))
		println("totTots")
		println(totTots)
		val totsAccumPctsts = totsAccumPctstsTemps(curTreePoses, GIMat(0->leftAccumPctsts.ncols))  //* (leftAccumPctsts.zeros(1, leftAccumPctsts.ncols) + GMat(1))
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

	private def getImpurityFor(accumPctsts : GMat, tots : GMat) : GMat = {
		(impurityType)  match {
			case (1) => {
				getImpurityForInfoGain(accumPctsts, tots)
			}
			case (2) => {
				getImpurityForGiniImpurityReduction(accumPctsts, tots)
			}
		}
	}

	private def getImpurityForInfoGain(accumPctsts : GMat, tots : GMat) : GMat = {
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

	
	private def getImpurityForGiniImpurityReduction(accumPctsts : GMat, tots : GMat) : GMat = {
		// add some e val to 
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
      	case (aa: GIMat, bb: GMat, ii : GIMat) => GMat.lexsort2i(aa, bb, ii)
    	}
  	}

  	/******************************************************************************************************************************
	 * DEPRECATED CODE:
	 ******************************************************************************************************************************/
	def getThresholdsAndUpdateTreesArray(treePos : Mat, oTreeVal : Mat, treesArray : Mat) {
		val t = oTreeVal.nrows

		/**
		 * TODO: PASS ONLY A COLUMN IN AT A TIME OF soTreeVal and soTreeValT
		 */
		for (curT <- 0 until t) {
			println("WE ARE ON TREE #" + curT)
			val sortedI = oTreeVal.izeros(1, n); // oTreeVal.izeros(t, 1);
			sortedI <-- (sortedIndices)
			val sortedIT = sortedI.t
			(treePos, oTreeVal, treeOffsets, sortedIT, cats, pcatst, treesArray) match {
				case (tP: GIMat, oTV : GMat, tO : GIMat, sIT : GIMat, cts : GMat, pctst : GMat, tA : GIMat) => {
					/* Sort everything */
					val sTreePosTemp : GIMat = (tP + GIMat(0)) // t, n
					val sTreePos : GIMat = sTreePosTemp(curT, 0->n)
					val sTreePosT : GIMat = sTreePos.t // n x t
					val soTreeValTemp : GMat = (oTV + 0f)
					val soTreeVal : GMat = soTreeValTemp(curT, 0->n)
					val soTreeValT : GMat = soTreeVal.t // n x t
					println("sTreePosT (n x t)")
					println(sTreePosT)
					println("soTreeValT (n x t)")
					println(soTreeValT)
					println("indices unsorted sIT")
					println(sIT)
					lexsort2i(sTreePosT, soTreeValT, sIT);
					println("sTreePosT (n x t) sorted")
					println(sTreePosT)
					println("soTreeValT (n x t) sorted")
					println(soTreeValT)
					println("indices sorted sIT")
					println(sIT)
					println(sIT.getClass)

					// On Tree #curT
					val tree_nnodes = (math.pow(2, k) + 0.5).toInt;
					println("Tree_nnodes: " + tree_nnodes);
					/* Take part of sorted Indices correspoding the tree number curT */
					val curTreeIndices = sIT//sIT(GIMat(0->n), 0)
					println(curTreeIndices)

					/* Make a jc corresponding to the current tree */
					val curOffset : GIMat = GIMat(tO(0, curT))
					println(curOffset.getClass)
					println("Current Offset")
					println(curOffset)
					val curTreePoses = sTreePosT(GIMat(0->n), 0) // - curOffset
					println(curTreePoses.getClass)
					println(curTreePoses)

					// val jcTemp = GMat(2*tree_nnodes, 1);
					// jcTemp.clear
					// TODO WRAP ACCUM!
					// CUMAT.accumJV(curTreePoses.data, 0, 1, jcTemp.data, jcTemp.length, jcTemp.length); //TODO!!!
					// TODO: changes to nnodes instead of 2*tree_nnodes?
					val jcTemp : GMat = GMat.accum(curTreePoses, 1, null, nnodes, 1)
					println("JCTemp")
					println(jcTemp)
					for (i  <- 0 until jcTemp.length) {
						println(jcTemp(i,0) + " ");
					}
					println("")
					// return
					// println("JCTemp2")
					// val jcTemp2  = jcTemp //jcTemp((tree_nnodes -1) to (2*tree_nnodes -2) , curT)
					// println(jcTemp2)
					// val jcTemp3 = cumsumi(jcTemp2, GIMat(0 on tree_nnodes))
					// val jc = GIMat(IMat(0 on FMat(jcTemp3))) // TODO: HACK
					val jc = GIMat(IMat(0 on FMat(cumsumg(jcTemp, GIMat(0 on nnodes))))) // TODO: HACK
					// val jc = jcT((tree_nnodes -1) to (2*tree_nnodes -2), 0)
					println("JC")
					println(jc)

					/* Make PCats the same order as the curTreeIndices which are sorted*/
					// int icopy_transpose(int *iptrs, float *in, float *out, int stride, int nrows, int ncols)
					println("Cats before sort")
					println(cts)
					CUMAT.icopyt(curTreeIndices.data, cts.data, pctst.data, n, c, n)	
					println("Cats after sort")
					println(pctst)

					/* Use the Sorted Categories to figure out the impurity */
					val accumPctst = cumsumg(pctst, jc, null)
					println("Accum PCats")
					println(accumPctst)

					println("calculating the impurityReduction")
					val impurityReductions = calcImpurityReduction(pctst, jc, curTreePoses)

					println(impurityReductions)
					println("JC2 for MAX")
					val jc2 = jc(((tree_nnodes -1) until (2*tree_nnodes)), 0) // DO THE REMOVAL HERE!!! jc removing the stuff that doesnt matter...
					println(jc2)
					val mxsimp = maxg(impurityReductions, jc2)
					// getBestCategoriesToMark
					val bCats = maxg(accumPctst, jc2)
					val bCats2 = bCats._1
					val bCats3 = maxg(bCats2.t, GIMat(0\bCats2.nrows))
					val bestCats = bCats3._2
					println("bestCats")
					println(bestCats)


					val maxes = mxsimp._1
					val maxis = mxsimp._2//(0->(jc.nrows - 1), 0) // TODO
					println("Max Impurity Reduction and Indicies")
					println(mxsimp)
					println("Maxes")
					println(maxes)
					println("Maxis")
					println(maxis)
		
					// TODO: take care of -1
					// add one to all the maxis indices a
					val tempMaxis = maxis + GIMat(1)
					println("tempMaxis")
					println(tempMaxis)
					val tempSoTreeValT = GMat(FMat(scala.Float.NegativeInfinity) on FMat(soTreeValT))
					println("tempSoTreeValT")
					println(tempSoTreeValT)
					val maxTreeProdVals = tempSoTreeValT(tempMaxis, 0)
					println("maxTreeProdVals")
					println(maxTreeProdVals)

					println("MARKING THE TREE PROD VALS IN TREESARRAY")
					markTreeProdVals(tA, maxTreeProdVals, tree_nnodes, nnodes, curT)
					// println("MARKING THE MAX CATEGORIES IN TREESARRAY")
					// markMaxCategories(tA, tree_nnodes, nnodes, curT);
					println(tA)
				}
			}
		}
	}

	// def markMaxCategories(tA : GIMat, tree_nnodes : Int, nnodes : Int, curT : Int) {
	// 	val tArray : GMat = new GMat(tA.nrows, tA.ncols, tA.data, tA.length)
	// 	// tArray(1, GIMat((nnodes * curT + tree_nnodes -1)->(nnodes * curT + 2*tree_nnodes - 1))) = GMat(1000)
	// 	println("TreesArrays: Marked with Max Categories")
	// 	println(tArray)
	// }

	/**
	 *
	 * nodes that don't have any samples currently there already have indices values on maxis of 0
	 * now we must further mark 0 for leaves that will have infogain deltas that are very little
	 *
	 */
	private def markLeaves(infoGain : GMat, maxis : GIMat) {
		// 0.01 * infoGain.ones(infoGain.nrows, infoGain.ncols)
		// inf
	}

	/**
	 * Updates treesArray threshold with node categorization 
	 */
	private def categorizeNodes(tP : Mat, cts : Mat, tA : Mat) {
		(tP, cts, tA) match {
			case (tPos : GIMat, ccats : GIMat, tArray : GIMat) => {

			}
		}
	}



}
