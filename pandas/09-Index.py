import pandas as pd
#  Index object is an interesting structure in itself,
#  and it can be thought of either as an immutable array or as an ordered set
ind = pd.Index([2, 3, 5, 7, 11])
print(ind)
# The Index in many ways operates like an array.
print(ind[1], ind[::2])
print(ind.size, ind.shape, ind.ndim, ind.dtype, '\n')

#  One difference between Index objects and NumPy arrays is
#  that indices are immutable. that is, they cannot be modified via the normal means
try:
    ind[1] = 110
except:
    print('ERRRRRRRRRROR!!!')
    print('Indicies can not be modified.\n')

'''
The Index object follows many of the conventions used by Python's built-in set data structure, so that unions,
intersections, differences, and other combinations can be computed in a familiar way.
'''
indA = pd.Index([1, 3, 5, 7, 9])
indB = pd.Index([2, 3, 5, 7, 11])
print('Intersection: ', indB & indA)
print('Union: ', indA | indB)
print('Symmetric Difference- present in one not the other! ', indA ^ indB)