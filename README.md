# cs267-hw
# dgemm-tuned-a: Optimization - 1, change the order of the loops (i,j,k); changes which matrix stays in memory and the type of product. No much improvement.
# dgemm-tuned-b: Optimization - 2, change the order of the loops, Blocking for L1 cache (64), unrolling the loops (4), copy optimization. Speed goes to ~1500, Percentage goes to ~13%
