'''
Given a square matrix, return its elements in spiral order
'''
import numpy as np
def spiral_order(a):

    if a.shape[0] == 0:
        return []
    if len(a) == 1:
        return list(a[0])
    return list(a[0, :-1]) + list(a[:, -1]) + list(reversed(a[-1, :-1])) + list(reversed(a[1:-1, 0])) + \
           spiral_order(a[1:-1, 1:-1])

a= np.array(np.arange(1, 26))
a= a.reshape((5,5))
print(a)
print(spiral_order(a))

a = np.array([1,2])
a = a[:, np.newaxis]
print(spiral_order(a))