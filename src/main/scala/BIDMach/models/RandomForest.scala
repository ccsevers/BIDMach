package BIDMach.models

import BIDMat.{BMat,CMat,CSMat,DMat,Dict,IDict,FMat,GMat,GIMat,GSMat,HMat,IMat,Mat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMat.Solvers._
import edu.berkeley.bid.CUMAT

/**
 * Random Forest Implementation
 */
 // val (dmy, freebytes, allbytes) = SciFunctions.GPUmem
class RandomForest(d : Int, t: Int, ns: Int, feats : Mat, cats : Mat, useGini : Boolean) {
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
	val treeTemp = IMat(f * rand(ns + 1, t * nnodes));
	min(treeTemp, f-1, treeTemp);
	treesArray <-- treeTemp;
	// val treesArray : GIMat = GIMat.newOrCheckGIMat(ns + 1, t * nnodes, null, f.GUID, c.GUID, "randomForest_treesArray".##); // Fix? maybe don't need newOrCheck
	val oTreePos = feats.izeros(t, n); 
	val oTreeVal = feats.zeros(t, n)
	//GIMat.newOrCheckGIMat(t, n, null, f.GUID, c.GUID, "randomForest_oTreeVal".##);

	/* Variables needed for Train method */
	// val treeOffsets : GIMat = GIMat(nnodes * irow(0->t));
	// val treeOffsets = feats.izeros(1,t);
	// treeOffsets <-- (nnodes * irow(0->t));
	// val embeddedTreePosWithVals : GMat = GMat.newOrCheckGMat(2 * t, n, null); // 2 times because its long long
	// val embeddedTreePosWithVals = feats.zeros(2 * t, n);
	// val embeddedSortedIndices : GIMat = GIMat.newOrCheckGIMat(2 * t, n, null); // 2 times because its long long
	// val embeddedSortedIndices = feats.izeros(2 * t, n);
	// val zeroToN : GIMat = GIMat(icol(0->n));
	// val zeroToN = feats.izeros(n, 1);
	// zeroToN <-- icol(0->n);
	// val pcats = GMat.newOrCheckGMat(cats.nrows, cats.ncols, null);
	// val zeroMat : GIMat = GIMat(izeros(n,t));
	// val zeroMat = feats.izeros(n,t); 

	def train {
		for (k <- 0 until d - 1) { // d of them; each level
			println("At Depth: " + k);
			val (dmy, freebytes, allbytes) = GPUmem
			println("dmy: " + dmy + " freebytes: " + freebytes + " allbytes: " + allbytes)
			/* 
			calculate all the inner products 
			*/
			// treeprod(unsigned int *trees, float *feats, int *tpos, int *otpos, int nrows, int ncols, int ns, int tstride, int ntrees, int doth)			
			// treeprod(treesArray.data, feats.data, treePos.data, oTreeVal.data, t, n, ns, nnodes * (ns + 1), t, 0);
			// treeProd(treesArray, feats, treePos, oTreeVal, t, n, ns, nnodes * (ns + 1), t, 0);
			println("Classes: " + treesArray.getClass + " " + feats.getClass + " " + treePos.getClass + " " + oTreeVal.getClass);
   			println("Starting treeProd")
   			GMat.treeProd(treesArray, feats, treePos, oTreeVal);

			val e = new EntropyEval(oTreeVal, cats, d, k)
			e.getThresholdsAndUpdateTreesArray(treePos, oTreeVal, treesArray)

			println("Starting TreeStep")
			GMat.treeProd(treesArray, feats, treePos, treePos)
			println("treePos Changed after stepping")
			println(treePos)
		}
		// mark last row all Leaves!
		markAllCurPositionsAsLeaves(treesArray, treePos)
		println("treesArray after marking all current positions as leaves")
		println(treesArray)
	}

	def classify(feats : Mat) : Mat = {
		(feats) match {
			case (fs: GMat) => {
				val newTreePos = fs.izeros(t, fs.ncols);//  GIMat.newOrCheckGIMat(t, n, null); 
				newTreePos.clear
				val treeCats = feats.izeros(t, fs.ncols)
				treeCats.clear
				println("Yea: NewTreePos: " + newTreePos)
				GMat.treeSearch(treesArray, fs, newTreePos, treeCats)
				treeCats
			}
		}
	}

	/**
	 * Updates treesArray threshold 
	 */
	def categorizeNodes(tP : Mat, cts : Mat, tA : Mat) {

	}

	/**
	 *
	 * Mark all current positions as Leaves
	 * TODO: Maybe mark and compute the categories too?
	 */
	def markAllCurPositionsAsLeaves(tArray : Mat, tPos : Mat ) {
	 	(tArray, tPos) match {
			case (tA : GIMat, tPos : GIMat) => {
				val tArr : GMat = new GMat(tA.nrows, tA.ncols, tA.data, tA.length)
	 			var curT = 0
	 			while (curT < t) {
	 				tArr(0,  tPos(curT, 0 -> n)) = scala.Float.NegativeInfinity * GMat(iones(1, n))
	 				// tArr(1,  tPos(curT, 0 -> n)) =  0 * GMat(iones(1, n))
	 				curT = curT + 1
	 			}
	 		}
	 	}
	}

	// // DERRICK
 //  	// treeprod(unsigned int *trees, float *feats, int *tpos, int *otpos, int nrows, int ncols, int ns, int tstride, int ntrees, int doth)     
 //  	def treeProd(treesArray : Mat, feats : Mat, treePos : Mat, oTreeVal : Mat) {
 //   		val nrows = feats.nrows;
 //    	val ncols = feats.ncols;
 //    	val ns = treesArray.nrows;
 //    	val ntrees = treePos.nrows;
 //    	val tstride = ns * (treesArray.ncols / ntrees);
 //    	(treesArray, feats, treePos, oTreeVal) match {
 //      		case (tA : GIMat, fs : GMat, tP : GIMat, oTV : GMat) => GMat.treeProd(tA, fs, tP, oTV, nrows, ncols, ns, tstride, ntrees)
 //      		case (tA : GIMat, fs : GMat, tP : GIMat, oTI : GIMat) => GMat.treeSteps(tA, fs, tP, oTI, nrows, ncols, ns, tstride, ntrees, 1)
 //    	}
 //  	}

}

// extra classes
class EntropyEval(oTreeVal : Mat, cats : Mat, d : Int, k : Int) {
	val n = oTreeVal.ncols
	val t = oTreeVal.nrows;
	val sortedIndices : IMat = iones(t,1) * irow(0->n)
	val treeOffsets = oTreeVal.izeros(1,t)
	val nnodes = (math.pow(2, d) + 0.5).toInt
	println("TreeOffsets")
	treeOffsets <-- (nnodes * icol(0->t))
	println(treeOffsets)
	val c = cats.nrows;
	val pcatst = oTreeVal.zeros(cats.ncols, cats.nrows);
	println("curdepth: " + k)

	val eps = 1E-5.toFloat

	def getThresholdsAndUpdateTreesArray(treePos : Mat, oTreeVal : Mat, treesArray : Mat) {
		val t = oTreeVal.nrows
		for (curT <- 0 until t) {
			println("WE ARE ON TREE #" + curT)
			val sortedI = oTreeVal.izeros(t, n);
			sortedI <-- (sortedIndices)
			val sortedIT = sortedI.t
			(treePos, oTreeVal, treeOffsets, sortedIT, cats, pcatst, treesArray) match {
				case (tP: GIMat, oTV : GMat, tO : GIMat, sIT : GIMat, cts : GMat, pctst : GMat, tA : GIMat) => {
					/* Sort everything */
					val sTreePos : GIMat = tP // t, n
					val sTreePosT : GIMat = sTreePos.t + tO // n x t
					val soTreeVal : GMat = oTV + 0f
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
					println("We are on curT #" + curT)
					val tree_nnodes = (math.pow(2, k) + 0.5).toInt;
					println("Tree_nnodes: " + tree_nnodes);
					/* Take part of sorted Indices correspoding the tree number curT */
					val curTreeIndices = sIT(GIMat(0->n), curT)
					println(curTreeIndices)

					/* Make a jc corresponding to the current tree */
					val curOffset : GIMat = GIMat(tO(0, curT))
					println(curOffset.getClass)
					println("Current Offset")
					println(curOffset)
					val curTreePoses = sTreePosT(GIMat(0->n), curT) - curOffset
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
						print(jcTemp(i,0) + " ");
					}
					println("")
					// return
					// println("JCTemp2")
					// val jcTemp2  = jcTemp //jcTemp((tree_nnodes -1) to (2*tree_nnodes -2) , curT)
					// println(jcTemp2)
					// val jcTemp3 = cumsumi(jcTemp2, GIMat(0 on tree_nnodes))
					// val jc = GIMat(IMat(0 on FMat(jcTemp3))) // TODO: HACK
					val jc = GIMat(IMat(0 on FMat(cumsumi(jcTemp, GIMat(0 on nnodes))))) // TODO: HACK
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
					val accumPctst = GMat.cumsumi(pctst, jc, null)
					println("Accum PCats")
					println(accumPctst)

					println("calculating the information gain delta")
					// val impurityReductions = GMat.calcImpurities(accumPctst , null, jc) //TODO:
					// TODO: smoosh all the impurityReducts into a sum... using maxi?
					val impurityReductions = calcInformationGainDelta(accumPctst, jc, curTreePoses)

					println(impurityReductions)
					var maxes : GMat = null
					var maxis : GIMat = null
					println("JC2 for MAX")
					val jc2 = jc(((tree_nnodes -1) until (2*tree_nnodes)), 0) // DO THE REMOVAL HERE!!! jc removing the stuff that doesnt matter...
					println(jc2)
					val mxsimp = maxs(impurityReductions, jc2)
					// getBestCategoriesToMark
					val bCats = maxs(accumPctst, jc2)
					val bCats2 = bCats._1
					val bCats3 = maxs(bCats2.t, GIMat(0\bCats2.nrows))
					val bestCats = bCats3._2
					println("bestCats")
					println(bestCats)


					maxes = mxsimp._1
					maxis = mxsimp._2//(0->(jc.nrows - 1), 0) // TODO
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
					val tempSoTreeValT = GMat(FMat(scala.Float.NegativeInfinity) on FMat(soTreeValT(0 -> soTreeValT.nrows, curT)))
					println("tempSoTreeValT")
					println(tempSoTreeValT)
					val maxTreeProdVals = tempSoTreeValT(tempMaxis, curT)
					println("maxTreeProdVals")
					println(maxTreeProdVals)

					// val maxTreeProdValsTreesIndices = GIMat(curT*nnodes) + maxis
					// println("maxTreeProdValsTreesIndices")
					// println(maxTreeProdValsTreesIndices)

					println("MARKING THE TREE PROD VALS IN TREESARRAY")
					val tArray : GMat = new GMat(tA.nrows, tA.ncols, tA.data, tA.length)
					tArray(0, GIMat((nnodes * curT + tree_nnodes -1)->(nnodes * curT + 2*tree_nnodes))) = maxTreeProdVals.t
					println("MARKING THE MAX CATEGORIES IN TREESARRAY")
					markMaxCategories(tArray, tree_nnodes, nnodes, curT)
					println("New TreesArray")
					println(tArray)
					println(treesArray)
				}
			}
		}
	}

	def getBestCategories(tPos: GIMat, accumPctst : GMat, jc2 : GIMat) {

	}

	def markMaxCategories(tArray : GMat, tree_nnodes : Int, nnodes : Int, curT : Int) {
		// tArray(1, GIMat((nnodes * curT + tree_nnodes -1)->(nnodes * curT + 2*tree_nnodes))) = null
	}

	/**
	 *
	 * nodes that don't have any samples currently there already have indices values on maxis of 0
	 * now we must further mark 0 for leaves that will have infogain deltas that are very little
	 *
	 */
	def markLeaves(infoGain : GMat, maxis : GIMat) {
		// 0.01 * infoGain.ones(infoGain.nrows, infoGain.ncols)
		// inf
	}

	def calcInformationGainDelta(accumPctst : GMat, jc : GIMat, curTreePoses : GIMat) : GMat = {
		// add some e val to
		println("accumPctst")
		println(accumPctst)
		println("jc")
		println(jc)
		println("curTreePoses")
		println(curTreePoses)

		/** for Total Impurity */
		// val indices = jc(1 until jc.length, 0) - GIMat(1)
		// println("indices")
		// println(indices)
		// val totVals = accumPctst(indices, GIMat(0 -> accumPctst.ncols))
		// println("totVals")
		// println(totVals)
		// val tots = totVals(curTreePoses, GIMat(0 -> totVals.ncols)) + GMat(eps); // imp
		
		val totsTemps = jc(1 -> jc.length, 0)
		println("totsTemps")
		println(totsTemps)
		val tots = GMat(totsTemps(curTreePoses, GIMat(0))) * (accumPctst.zeros(1, accumPctst.ncols) + GMat(1))
		println("tots")
		println(tots)
		val rightAccumPctst = tots - accumPctst
		println("rightAccumPctst")
		println(rightAccumPctst)
		println("asdfasdfa")
		println(GMat(0->accumPctst.nrows).t)
		println(accumPctst.zeros(1, accumPctst.ncols) + GMat(1))
		println("aasdafasdf")
		val tempIndices = GMat(1->(accumPctst.nrows + 1)).t * (accumPctst.zeros(1, accumPctst.ncols) + GMat(1))
		println("tempIndices")
		println(tempIndices)
		val totEntropy = null // imp
		val leftEntropy = null // imp
		val rightEntropy = null // imp

		val pses = (accumPctst / (tots + GMat(eps))); // imp
		println("pses")
		println(pses)
		val conjpses = (GMat(1) - pses) + GMat(eps); // imp
		println("conjpses")
		println(conjpses)
		println("ln(pses)")
		println(ln(pses))
		println("ln(conjpses)")
		println(ln(conjpses))
		val infoGain = GMat(-1 * sum((pses *@ ln(pses)) + (conjpses *@ ln(conjpses)),2))
		println("infoGain")
		println(infoGain)
		return infoGain
	}
	
	def calcGiniImpurityReduction(accumPctst : GMat, jc : GIMat) : GMat = {
		// add some e val to 
		return null
	}

	// DERRICK
  	def lexsort2i(a : Mat, b: Mat, i : Mat) {
    	(a, b, i) match {
      	case (aa: GIMat, bb: GMat, ii : GIMat) => GMat.lexsort2i(aa, bb, ii)
    	}
  	}

}
