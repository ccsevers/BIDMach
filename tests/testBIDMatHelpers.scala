import BIDMach.models.BIDMatHelpers._

val a = IMat(1\1\3\2\1)
val b = FMat(5\1\3\2\2)
val i = IMat(0\1\2\3\4)
lexsort2i(a, b, i)
a
b
i