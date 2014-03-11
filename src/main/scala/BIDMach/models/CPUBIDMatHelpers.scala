package BIDMach

import BIDMat.{SBMat,CMat,CSMat,DMat,Dict,IDict,FMat,GMat,GIMat,GSMat,HMat,IMat,Mat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMat.Solvers._
import BIDMat.Sorting._
import edu.berkeley.bid.CUMAT

class CPUBIDMatHelpers {
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
			var sumSoFar = 0
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
		println("maxg")
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
		println("maxg")
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

object CPUBIDMatHelpers {

}