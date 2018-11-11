# 1- dot product:
import numpy as np
a = np.array([1, 2, 3])
b = np.array([2, 2, 2])
print('Inner product of vectors:')
print(np.dot(a, b))
print('this function generalizes to matrix multiplication')
a = np.array([[0, 1], [2, 3]])
b = np.array([2, 3])
print(a, '\n', b)
print('product=', np.dot(b, a))
print(np.allclose(np.dot(b, a), np.matmul(b, a)))
# 2-determinat of a matrix:
print('determinant:', a, a.shape, np.linalg.det(a))
# 3- eigen values and eigen vectors
print( 'Eigen values and Eigen vectors:')
vals, vecs = np.linalg.eig(a)
print(vals, '\n', vecs)
# 3- inverse of a matrix:
print(np.linalg.inv(a))
print(np.dot(a, np.linalg.inv(a)))
print(' 4- Singular value decomposition:')
U, s, Vh = np.linalg.svd(a)
print('eigen Vals:   ', s)
print('Eigen vectors=   ', U, '\n')
print(Vh)
####################################
print('polynomial mathematics:')
# given a set of roots, it is possible to show the polynomial coefficients.
print(np.poly([1, -1]))  # the polynomial is: x^2+0x-1
# opposite operations: if we know the coefficient of polynomial, what are the roots?
print('PolyNomial roots:')
print(np.roots([1, 5, 2, -1]))
# evaluating a polynomial at a particular value:
print('fixed point value of polynomial')
print(np.polyval([1, 2, 3, 1], 1))
################
# fitting a polynomial of specified order to data:
x = [1, 2, 3, 4, 5, 6, 7, 8]
y = [0, 2, 1, 3, 7, 10, 11, 19]
print(np.polyfit(x, y, 2))  # the polynomial is: 0.375x^2-0.8869x+1.053
