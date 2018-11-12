import numpy as np
'''
Given numbers a1, a2, a3,.. and their probabilities p1, p2, p3, and a uniform RNG
generate a sample.
sum(pi)= 1
#note when the number is generated, searching the probability intervals
that the prob falls in takes O(n)
but since the commulative probability is sorted we can do binary search and takes O(logn)
'''
import itertools, bisect, random
def non_uniform_generator(A, p):
    random.seed(1)
    cummulative_probs = list(itertools.accumulate(p))
    #returns the first index in commulative probs >= random.random()
    rs = bisect.bisect(cummulative_probs, random.random())
    return A[rs]

print(non_uniform_generator([1,2,3], [.3, .4, .3]))
def nonUniformGen1(a, p):
    random.seed(2)
    r = random.random()
    print(r)
    for i in range(len(p)):
        if r<= np.sum(p[:i+1]):
            return a[i]

print(nonUniformGen1([1,2,3], [.3, .4, .3]))