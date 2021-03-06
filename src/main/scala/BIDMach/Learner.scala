package BIDMach
import BIDMat.{Mat,SBMat,CMat,DMat,FMat,IMat,HMat,GMat,GIMat,GSMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMat.Plotting._
import BIDMat.about
import BIDMach.models._
import BIDMach.updaters._
import BIDMach.datasources._
import scala.collection.immutable.List
import scala.collection.mutable.ListBuffer
import scala.concurrent.future
import scala.concurrent.ExecutionContext.Implicits.global

case class Learner(
    val datasource:DataSource, 
    val model:Model, 
    val regularizer:Regularizer, 
    val updater:Updater, 
		val opts:Learner.Options = new Learner.Options) {
  
  var results:FMat = null
  val dopts:DataSource.Opts = datasource.opts
	val mopts:Model.Opts	= model.opts
	val ropts:Regularizer.Opts = if (regularizer != null) regularizer.opts else null
	val uopts:Updater.Opts = updater.opts
	var useGPU = false
	
	def setup = {
	  Learner.setupPB(datasource, mopts.putBack, mopts.dim)   
  }
  
  def init = {
    datasource.init
    model.init(datasource)
    if (regularizer != null) regularizer.init(model)
    updater.init(model)
    useGPU = model.useGPU
  }
    
  def run = {
    setup
    init
    rerun
  }
   
  def rerun() = {
    flip 
    var cacheState = Mat.useCache
    Mat.useCache = true
    var done = false
    var ipass = 0
    var here = 0L
    var lasti = 0
    var bytes = 0L
    updater.clear
    val reslist = new ListBuffer[FMat]
    val samplist = new ListBuffer[Float]
    while (ipass < opts.npasses && ! done) {
    	var lastp = 0f
      datasource.reset
      var istep = 0
      println("pass=%2d" format ipass)
      while (datasource.hasNext) {
        val mats = datasource.next    
        here += datasource.opts.blockSize
        bytes += 12L*mats(0).nnz
        if ((istep - 1) % opts.evalStep == 0 || ! datasource.hasNext) {
        	val scores = model.evalblockg(mats, ipass)
        	reslist.append(scores.newcopy)
        	samplist.append(here)
        } else {
        	model.doblockg(mats, ipass, here)
        	if (regularizer != null) regularizer.compute(here)
        	updater.update(ipass, here)
        }   
        if (model.opts.putBack >= 0) datasource.putBack(mats, model.opts.putBack)
        istep += 1
        val dsp = datasource.progress
        if (dsp > lastp + opts.pstep && reslist.length > lasti) {
        	val gf = gflop
        	lastp = dsp - (dsp % opts.pstep)
        	print("%5.2f%%, %s, gf=%5.3f, secs=%3.1f, GB=%4.2f, MB/s=%5.2f" format (
        			100f*lastp, 
        			Learner.scoreSummary(reslist, lasti, reslist.length),
        			gf._1,
        			gf._2, 
        			bytes*1e-9,
        			bytes/gf._2*1e-6))  
        			if (useGPU) {
        				print(", GPUmem=%3.2f" format GPUmem._1) 
        			}
        	println
        	lasti = reslist.length
        }
      }
      updater.updateM(ipass)
      ipass += 1
    }
    val gf = gflop
    Mat.useCache = cacheState
    println("Time=%5.4f secs, gflops=%4.2f" format (gf._2, gf._1))
    results = Learner.scores2FMat(reslist) on row(samplist.toList)
  }
  
  def datamats = datasource.asInstanceOf[MatDS].mats
  def modelmats = model.modelmats
  def datamat = datasource.asInstanceOf[MatDS].mats(0)
  def modelmat = model.modelmats(0)
}

case class ParLearnerx(
    val datasources:Array[DataSource], 
    val models:Array[Model], 
    val regularizers:Array[Regularizer], 
    val updaters:Array[Updater], 
		val opts:ParLearner.Options = new ParLearner.Options) {
  
  var um:Array[Mat] = null
  var mm:Array[Mat] = null
  var results:FMat = null
  var useGPU = false
  
  def setup = {
	  for (i <- 0 until opts.nthreads) {
	  	val mopts	= models(i).opts
	  	Learner.setupPB(datasources(i), mopts.putBack, mopts.dim)
	  }   
  }
  
  def init = {
    val thisGPU = if (Mat.hasCUDA > 0) getGPU else 0
  	for (i <- 0 until opts.nthreads) {
  		if (i < Mat.hasCUDA) setGPU(i)
  		datasources(i).init
  		models(i).init(datasources(i))
  		if (regularizers != null) regularizers(i).init(models(i))
  		updaters(i).init(models(i))
  	}
  	useGPU = models(0).useGPU
  	if (Mat.hasCUDA > 0) setGPU(thisGPU)
  	val mml = models(0).modelmats.length
    um = new Array[Mat](mml)
    mm = new Array[Mat](mml)
    for (i <- 0 until mml) {
    	val mm0 = models(0).modelmats(i)
    	mm(i) = mm0.zeros(mm0.nrows, mm0.ncols)
    	um(i) = mm0.zeros(mm0.nrows, mm0.ncols)
    }
  }
  
  def run = {
    setup
    init
    rerun
  }
  
  def rerun() = {
	  flip 
	  var cacheState = Mat.useCache
    Mat.useCache = true
	  val thisGPU = if (useGPU) getGPU else 0
	  if (useGPU) {
	    for (i <- 0 until opts.nthreads) {
	      if (i != thisGPU) connect(i)
	    }
	  }

	  @volatile var done = izeros(opts.nthreads, 1)
	  var ipass = 0
	  var istep0 = 0L
	  var ilast0 = 0L	
	  var bytes = 0L
	  val reslist = new ListBuffer[FMat]
	  val samplist = new ListBuffer[Float]    	  	
	  var lastp = 0f
	  var lasti = 0
	  done.clear
	  for (ithread <- 0 until opts.nthreads) {
	  	future {
	  		if (useGPU && ithread < Mat.hasCUDA) setGPU(ithread)
	  		var here = 0L
	  		updaters(ithread).clear
	  		while (done(ithread) < opts.npasses) {
	  			var istep = 0
	  			while (datasources(ithread).hasNext) {
	  				val mats = datasources(ithread).next
	  				here += datasources(ithread).opts.blockSize
	  				for (j <- 0 until mats.length) bytes += 12L * mats(j).nnz
	  				models(0).synchronized {
	  					istep += 1
	  					istep0 += 1
	  				}
	  				try {
	  					if (istep % opts.evalStep == 0) {
	  						val scores = models(ithread).synchronized {models(ithread).evalblockg(mats, ipass)}
	  						reslist.synchronized { reslist.append(scores) }
	  						samplist.synchronized { samplist.append(here) }
	  					} else {
	  						models(ithread).synchronized {
	  							models(ithread).doblockg(mats, ipass, here)
	  							if (regularizers != null && regularizers(ithread) != null) regularizers(ithread).compute(here)
	  							updaters(ithread).update(ipass, here)
	  						}
	  					}
	  				} catch {
	  				case e:Exception => {
	  					print("Caught exception in thread %d %s\nTrying restart..." format (ithread, e.toString))
	  					restart(ithread)
	  					println("Keep on truckin...")
	  				}
	  				} 
	  				if (useGPU) Thread.sleep(opts.coolit)
	  				if (models(ithread).opts.putBack >= 0) datasources(ithread).putBack(mats, models(ithread).opts.putBack)
//	  				if (istep % (opts.syncStep/opts.nthreads) == 0) syncmodel(models, ithread)
	  			}
	  			models(ithread).synchronized { updaters(ithread).updateM(ipass) }
	  			done(ithread) += 1
	  			while (done(ithread) > ipass) Thread.sleep(1)
	  		}
	  	}
	  }
	  println("pass=%2d" format ipass) 
	  while (ipass < opts.npasses) {
	  	while (mini(done).v == ipass) {
	  		if (istep0 >= ilast0 + opts.syncStep) {
	  			ParLearner.syncmodels(models, mm, um, useGPU)
	  			ilast0 += opts.syncStep
	  		}
	  		if (dsProgress > lastp + opts.pstep) {
	  			while (dsProgress > lastp + opts.pstep) lastp += opts.pstep
	  			val gf = gflop
	  			if (reslist.length > lasti) {
	  				print("%5.2f%%, %s, gf=%5.3f, secs=%3.1f, GB=%4.2f, MB/s=%5.2f" format (
	  						100f*lastp, 
	  						reslist.synchronized {
	  							Learner.scoreSummary(reslist, lasti, reslist.length)
	  						},
	  						gf._1,
	  						gf._2, 
	  						bytes*1e-9,
	  						bytes/gf._2*1e-6))  
	  						if (useGPU) {
	  							for (i <- 0 until math.min(opts.nthreads, Mat.hasCUDA)) {
	  								setGPU(i)
	  								if (i==0) print(", GPUmem=%3.2f" format GPUmem._1) else print(", %3.2f" format GPUmem._1)
	  							}
	  							setGPU(thisGPU)
	  						}
	  				println
	  			}
	  			lasti = reslist.length
	  		} else {
	  		  Thread.sleep(1)
	  		}
	  	}
	  	lastp = 0f
	  	if (ipass < opts.npasses) {
	  	  for (i <- 0 until opts.nthreads) datasources(i).reset
	  	  println("pass=%2d" format ipass+1) 
	  	}
	  	if (opts.resFile != null) {
      	saveAs(opts.resFile, Learner.scores2FMat(reslist) on row(samplist.toList), "results")
      }
	  	ipass += 1
	  }
	  val gf = gflop
	  Mat.useCache = cacheState
	  println("Time=%5.4f secs, gflops=%4.2f, MB/s=%5.2f, GB=%5.2f" format (gf._2, gf._1, bytes/gf._2*1e-6, bytes*1e-9))
	  results = Learner.scores2FMat(reslist) on row(samplist.toList)
  }
  
  def syncmodel(models:Array[Model], ithread:Int) = {
	  mm.synchronized {
	    for (i <- 0 until models(ithread).modelmats.length) {
	    	um(i) <-- models(ithread).modelmats(i)
	    	um(i) ~ um(i) *@ (1f/opts.nthreads)
	    	mm(i) ~ mm(i) *@ (1 - 1f/opts.nthreads)
	    	mm(i) ~ mm(i) + um(i)
	    	models(ithread).modelmats(i) <-- mm(i)
	    }
	  }
  }
  
  def restart(ithread:Int) = {
    if (useGPU) {
      resetGPU
      Mat.trimCache2(ithread)
    }
    models(ithread).init(datasources(ithread))
    for (i <- 0 until models(ithread).modelmats.length) {
    	models(ithread).modelmats(i) <-- mm(i)
    }
    updaters(ithread).init(models(ithread))      
  }
  
  def dsProgress:Float = {
    var sum = 0f
    for (i <- 0 until datasources.length) {
      sum += datasources(i).progress
    }
    sum / datasources.length
  }
  
  def modelmats = models(0).modelmats
  def modelmat = models(0).modelmats(0)

}

class ParLearnerxF(
    dopts:DataSource.Opts,
		ddfun:(DataSource.Opts, Int)=>DataSource,
		mopts:Model.Opts,
		mkmodel:(Model.Opts)=>Model,
		ropts:Regularizer.Opts,
		mkreg:(Regularizer.Opts)=>Regularizer,
		uopts:Updater.Opts,
		mkupdater:(Updater.Opts)=>Updater,
		val lopts:ParLearner.Options = new ParLearner.Options) {

  var dds:Array[DataSource] = null
  var models:Array[Model] = null
  var regularizers:Array[Regularizer] = null
  var updaters:Array[Updater] = null
  var learner:ParLearnerx = null
  
  def setup = {
    dds = new Array[DataSource](lopts.nthreads)
    models = new Array[Model](lopts.nthreads)
    if (mkreg != null) regularizers = new Array[Regularizer](lopts.nthreads)
    updaters = new Array[Updater](lopts.nthreads)
    val thisGPU = if (Mat.hasCUDA > 0) getGPU else 0
    for (i <- 0 until lopts.nthreads) {
      if (mopts.useGPU && i < Mat.hasCUDA) setGPU(i)
    	dds(i) = ddfun(dopts, i)
    	models(i) = mkmodel(mopts)
    	if (mkreg != null) regularizers(i) = mkreg(ropts)
    	updaters(i) = mkupdater(uopts)
    }
    if (0 < Mat.hasCUDA) setGPU(thisGPU)
    learner = new ParLearnerx(dds, models, regularizers, updaters, lopts)
    learner.setup
  }
  
  def init = learner.init
  
  def run = {
    setup
    init
    learner.rerun
  }
}

case class ParLearner(
    val datasource:DataSource, 
    val models:Array[Model], 
    val regularizers:Array[Regularizer], 
    val updaters:Array[Updater], 
		val opts:ParLearner.Options = new ParLearner.Options) {
  
  var um:Array[Mat] = null
  var mm:Array[Mat] = null
  var results:FMat = null
  var cmats:Array[Array[Mat]] = null
  var useGPU = false
  
  def setup = {
	  val mopts	= models(0).opts
	  Learner.setupPB(datasource, mopts.putBack, mopts.dim)  
  }
  
  def init = {
    datasource.init
    useGPU = models(0).opts.useGPU
    val thisGPU = if (useGPU) getGPU else 0
    for (i <- 0 until opts.nthreads) {
      if (useGPU && i < Mat.hasCUDA) setGPU(i)
    	models(i).init(datasource)
    	if (regularizers != null) regularizers(i).init(models(i))
    	updaters(i).init(models(i))
    }
    if (useGPU) setGPU(thisGPU) 
    val mml = models(0).modelmats.length
    um = new Array[Mat](mml)
    mm = new Array[Mat](mml)
    for (i <- 0 until mml) {
    	val mm0 = models(0).modelmats(i)
    	mm(i) = mm0.zeros(mm0.nrows, mm0.ncols)
    	um(i) = mm0.zeros(mm0.nrows, mm0.ncols)
    }
  }
  
  def run = {
    setup 
    init
    rerun
  }
  
  def rerun = {
    flip
    val mm0 = models(0).modelmats(0)
    var cacheState = Mat.useCache
    Mat.useCache = true
    cmats = new Array[Array[Mat]](opts.nthreads)
    for (i <- 0 until opts.nthreads) cmats(i) = new Array[Mat](datasource.omats.length)
    val thisGPU = if (useGPU) getGPU else 0
	  if (useGPU) {
	    for (i <- 0 until opts.nthreads) {
	      if (i != thisGPU) connect(i)
	    }
	  }    
    @volatile var done = iones(opts.nthreads, 1)
    var ipass = 0
    var here = 0L
    var feats = 0L
    var lasti = 0
    var bytes = 0L
    val reslist = new ListBuffer[FMat]
    val samplist = new ListBuffer[Float]
    for (i <- 0 until opts.nthreads) {
    	if (useGPU && i < Mat.hasCUDA) setGPU(i)
    	updaters(i).clear
    }
    setGPU(thisGPU)
    var istep = 0
    var lastp = 0f
    var running = true

    for (ithread <- 0 until opts.nthreads) {
    	future {
    		if (useGPU && ithread < Mat.hasCUDA) setGPU(ithread)
    		while (running) {
    			while (done(ithread) == 1) Thread.sleep(1)
    			try {
    				if ((istep + ithread + 1) % opts.evalStep == 0 || !datasource.hasNext ) {
    					val scores = models(ithread).evalblockg(cmats(ithread), ipass)
    					reslist.synchronized { reslist.append(scores(0)) }
    					samplist.synchronized { samplist.append(here) }
    				} else {
    					models(ithread).doblockg(cmats(ithread), ipass, here)
    					if (regularizers != null && regularizers(ithread) != null) regularizers(ithread).compute(here)
    					updaters(ithread).update(ipass, here)
    				}
    			} catch {
    			case e:Exception => {
    				print("Caught exception in thread %d %s\nTrying restart..." format (ithread, e.toString))
    				restart(ithread)
    				println("Keep on truckin...")
    			}
    			} 
    			done(ithread) = 1 
    		}  
    	}
    }
    while (ipass < opts.npasses) {
    	datasource.reset
      istep = 0
      lastp = 0f
      println("pass=%2d" format ipass)
    	while (datasource.hasNext) {
    		for (ithread <- 0 until opts.nthreads) {
    			if (datasource.hasNext) {
    				val mats = datasource.next
    				for (j <- 0 until mats.length) {
    				  cmats(ithread)(j) = safeCopy(mats(j), ithread) 
    				}
    				here += datasource.opts.blockSize
    				feats += mats(0).nnz
    				done(ithread) = 0;
    				bytes += 12L*mats(0).nnz
    			} 
    		}
      	while (mini(done).v == 0) Thread.sleep(1)
      	Thread.sleep(opts.coolit)
      	istep += opts.nthreads
      	if (istep % opts.syncStep == 0) ParLearner.syncmodels(models, mm, um, useGPU)
      	if (datasource.progress > lastp + opts.pstep) {
      		while (datasource.progress > lastp + opts.pstep) lastp += opts.pstep
      		val gf = gflop
      		if (reslist.length > lasti) {
      			print("%5.2f%%, %s, gf=%5.3f, secs=%3.1f, GB=%4.2f, MB/s=%5.2f" format (
      					100f*lastp, 
      					Learner.scoreSummary(reslist, lasti, reslist.length),
      					gf._1,
      					gf._2, 
      					bytes*1e-9,
      					bytes/gf._2*1e-6))  
      		  if (useGPU) {
      		    for (i <- 0 until math.min(opts.nthreads, Mat.hasCUDA)) {
      		      setGPU(i)
      		      if (i==0) print(", GPUmem=%3.2f" format GPUmem._1) else print(", %3.2f" format GPUmem._1)
      		    }
      		    setGPU(thisGPU)
      		  }
      			println
      		}
      		lasti = reslist.length
      	}
      }
      for (i <- 0 until opts.nthreads) {
        if (useGPU && i < Mat.hasCUDA) setGPU(i); 
        updaters(i).updateM(ipass)
      }
      setGPU(thisGPU)
      ipass += 1
      if (opts.resFile != null) {
      	saveAs(opts.resFile, Learner.scores2FMat(reslist) on row(samplist.toList), "results")
      }
    }
    running = false
    val gf = gflop
    Mat.useCache = cacheState
    println("Time=%5.4f secs, gflops=%4.2f, samples=%4.2g, MB/sec=%4.2g" format (gf._2, gf._1, 1.0*here, bytes/gf._2/1e6))
    results = Learner.scores2FMat(reslist) on row(samplist.toList)
  }
  
  def safeCopy(m:Mat, ithread:Int):Mat = {
    m match {
      case ss:SMat => {
        val out = SMat.newOrCheckSMat(ss.nrows, ss.ncols, ss.nnz, null, m.GUID, ithread, "safeCopy".##)
        ss.copyTo(out)
      }
      case ss:FMat => {
        val out = FMat.newOrCheckFMat(ss.nrows, ss.ncols, null, m.GUID, ithread, "safeCopy".##)
        ss.copyTo(out)
      }
    }
  }
  
  def restart(ithread:Int) = {
    if (useGPU) {
      resetGPU
      Mat.trimCaches(ithread)
    }
    models(ithread).init(datasource)
    models(ithread).modelmats(0) <-- mm(0)
    updaters(ithread).init(models(ithread))      
  }
  
  def datamats = datasource.asInstanceOf[MatDS].mats
  def modelmats = models(0).modelmats
  def datamat = datasource.asInstanceOf[MatDS].mats(0)
  def modelmat = models(0).modelmats(0)
}

class ParLearnerF(
		val ds:DataSource,
		val mopts:Model.Opts,
		mkmodel:(Model.Opts)=>Model,
		ropts:Regularizer.Opts,
		mkreg:(Regularizer.Opts)=>Regularizer,
		val uopts:Updater.Opts,
		mkupdater:(Updater.Opts)=>Updater,
		val lopts:ParLearner.Options = new ParLearner.Options) {
  var models:Array[Model] = null
  var regularizers:Array[Regularizer] = null
  var updaters:Array[Updater] = null
  var learner:ParLearner = null
  
  def setup = {
    models = new Array[Model](lopts.nthreads)
    if (mkreg != null) regularizers = new Array[Regularizer](lopts.nthreads)
    updaters = new Array[Updater](lopts.nthreads) 
    val thisGPU = if (Mat.hasCUDA > 0) getGPU else 0
    for (i <- 0 until lopts.nthreads) {
      if (mopts.useGPU && i < Mat.hasCUDA) setGPU(i)
    	models(i) = mkmodel(mopts)
    	if (mkreg != null) regularizers(i) = mkreg(ropts)
    	updaters(i) = mkupdater(uopts)
    }
    if (0 < Mat.hasCUDA) setGPU(thisGPU)
    learner = new ParLearner(ds, models, regularizers, updaters, lopts)   
    learner.setup
  }
  
  def init =	learner.init
  
  def run = {
    setup
    init
    rerun
  }
  
  def rerun = learner.rerun
}

object Learner {
  
  class Options extends BIDMat.Options {
  	var npasses = 10 
  	var evalStep = 11
  	var pstep = 0.01f
  	var resFile:String = null
  }
  
  def setupPB(ds:DataSource, npb:Int, dim:Int) = {
    ds match {
    case ddm:MatDS => {
    	if (npb >= 0) {
    		ddm.setupPutBack(npb, dim)
    	}
    }
    case _ => {}
    }
  }
  
  def scoreSummary(reslist:ListBuffer[FMat], lasti:Int, length:Int):String = {
    var i = lasti
    var sum = 0.0
    while (i < length) {
      val scoremat = reslist(i)
      sum += mean(scoremat(?,0)).v
      i += 1
    }
    ("ll=%5.3f" format sum/(length-lasti))    
  }
  
  def scores2FMat(reslist:ListBuffer[FMat]):FMat = {
    val out = FMat(reslist(0).nrows, reslist.length)
    var i = 0
    while (i < reslist.length) {
      val scoremat = reslist(i)
      out(?, i) = scoremat(?,0)
      i += 1
    }
    out
  }
}

object ParLearner {
  
  class Options extends 
  Learner.Options {
  	var nthreads = math.max(0, Mat.hasCUDA)
  	var syncStep = 32
  	var coolit = 60
  }
  
  def syncmodels(models:Array[Model], mm:Array[Mat], um:Array[Mat], useGPU:Boolean) = {
	  for (j <- 0 until models(0).modelmats.length) {
	  	mm(j).clear
	  	for (i <- 0 until models.length) {
	  		um(j) <-- models(i).modelmats(j)
	  		mm(j) ~ mm(j) + um(j)
	  	}
	  	mm(j) ~ mm(j) * (1f/models.length)
	  	for (i <- 0 until models.length) {
	  		models(i).modelmats(j) <-- mm(j)
	  	}
	  }
  }
}

