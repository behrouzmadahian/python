__author__ = 'bmdahian'
# When standard mathematical operations are used with arrays, they are applied on an element-by-element
# basis, arrays should have the same dimensions!
import numpy as np
a = np.array([1, 2, 3], float)
b = np.array([2, 4, 6], float)
print(a+b, a-b, a/b, a**b)
a = np.array([[1, 2], [3, 4]])
b = np.array([[2, 0], [1, 3]])
# elementwise! multiplication and not matrix multiplication!
print(a*b)
# note: if one array has smaller dimension it will be repeated as necessary to perform the operation
print('Expanding Arr dimension to prform operation')
a = np.array([[1, 2], [3, 4], [5, 6]])
b = np.array([-1, 3])
print(a+b)
print(abs(b))
print(np.sqrt(a))
print(np.pi, np.e)

# if we iterate over a multi dimensional array, iteration will take place along the first axis (rows!)
#s uch that each loop returns a subsection of the array!
for x in a:
    print(x)
#########################
print('basic array operations:')
print(a.sum(), a.prod(), a.mean(), a.var(), a.std(), a.min(), a.max())
# returning the index of max and min:
print(a.argmin(), a.argmax(), a.shape)
print('we can perform this operations along one axis!')
print(a, np.shape(a))
print('Column sums:', a.sum(axis=0))  # col means
print('row sums:', a.sum(axis=1))  # row means
###############
print('sorting an array:')
a = np.array([2, 4, 3, 6, 9, 67, 8, 98])
print(a, sorted(a))
print('returning unique elements from an array:')
a = np.array([11, 1, 2, 2, 3, 4, 5, 5, 5])
print(np.unique(a))
print('returning diagonal elements:')
a = np.identity(4)*3
print(a, np.diagonal(a))
###############################################
print('comparison operators and value testing:')
print('elementwise comparisons:')
a = np.array([1, 3, 0])
b = np.array([0, 3, 2])
print(a > b)
c = a > b  # boolean array!
print(c)
print('comparing to single values:')
print(a > 2)
# are any elements of boolean array true?
print(any(c))
# are all elements of our boolean array true?
print(all(c))

# where(): makes a new array from two arrays of equivalent size, using a boolean filter:
print('Using Where:')
a = np.array([1, 3, 0], float)
a1 = np.array([4, 5, 6])
b = np.where(a != 0, a1, a)
c = np.where(a > 0, 3, 2)
print(a, b, 'and  ', c)
print(np.where(a==0)[0])
#############
print('indicies of non-zero values:')
a = np.array([[0, 1], [3, 0]])

print(a, '\n', a.nonzero())
# checking NaN values and Inf values!
a=np.array([1, np.NaN, np.Inf])
print(a)
print(np.isnan(a))
print(np.isinf(a), np.isfinite(a))


