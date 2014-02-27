//  :load tests/testMaxG.scala

val a = GMat(1 \2 \3 on 0 \ 1 \ 4)
val jc = GIMat(0 on 2)

val mg = maxg(a, jc)
val mg_vals = mg._1
val mg_indicies = mg._2
println("mg")
println(mg)
println("mg_vals")
println(mg_vals.getClass)
println(mg_vals)
println("mg_indicies")
println(mg_indicies.getClass)
println(mg_indicies)

val expected_vals = GMat(1 \ 2 \ 4)
val expected_indices = GIMat(0 \ 0 \ 1)
println("expected_vals")
println(expected_vals.getClass)
println(expected_vals)
println("expected_indices")
println(expected_indices)
println(expected_indices.getClass)
println("mg_vals correct?: " + GMat(mg_vals) == expected_vals)
println("mg_indicies correct?: " + GIMat(mg_indicies) == expected_indices)