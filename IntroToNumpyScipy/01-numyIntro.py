import numpy as np
'''
numpy provides functionality to manipulate arrays and matrices!!
arrays: are similar to lists except that every element of an array must be
of the same type, typically a numeric type.
are more efficient than lists and make operations with large amount of data very fast.
'''

a = np.array([1, 2, 3, 4], float)
print(a, a[2:])
# multi-dimensional arrays:
a = np.array([[1, 2, 3], [4, 5, 6]])
print(a, a[0, 0])
print(a[1:, :])  # second row
print(a[:, 1])  # second column
print(a[:, 1:3])  # columns 2 and 3
# array dimensions:
print(a.shape)
# type of values in an array:
print(a.dtype)
# len returns length of the first axis: number of rows!
print(len(a))
print(2 in a)
# reshaping array to have new dimensions:
a = np.array(range(10), float)
print(a)
a = a.reshape((5, 2))
print(a)
a = np.array(range(3))
print(a)
# making all elements zero:
a.fill(0)
print(a)
#######Transposing an array:
a = np.array(range(12))
a = a.reshape((3, 4))
print(a)
a = a.transpose()
print(a)
##############
# one dimensional versions of multi-dimensional array:
a= a.flatten()
print(a)
#########
# concatenating to arrays:
a = np.array([1, 1, 1])
b = np.array([2, 2, 2])
c = np.concatenate((a, b))
print(c)
# we can specify along which axis to concatenate if multidimensional
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
c = np.concatenate((a, b), 0) #rbind
print(c)
c = np.concatenate((a, b), 1)#cbind
print(c)
######################
# the dimensionality of an array can be increased using the newaxis constant in bracket notation:
print('Increasing Dimensionality of an array:')
a = np.array([1, 2, 3])
print(a, a.shape)
a=a[:, np.newaxis]
print(a, a.shape)
a=np.array([1,2,3])
a=a[np.newaxis, :]
print(a, a.shape)
#######################################
# other ways to create arrays:
# arange() is similar to range function but returns an array:
a = np.arange(10)
print(a)
# making one/multi dimension arrays of ones and zeros:
a = np.ones(10)
b = np.zeros(10)
print(a, b)
a = np.ones((2, 5))
b = np.zeros((2, 5))
print(a)
print(b)
#######
# identity matrix:
a = np.identity(4)
print(a)
# eye function returns matrices with ones along kth diagonal:
a = np.eye(4, k=1)
print(a)
