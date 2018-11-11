import numpy as np
# boolean selection of an array
a = np.array([[6, 4], [5, 9]])
print(a >= 6)
print(a[a >= 6])
# selection using indicesL
a = np.array([2, 4, 6, 8])
b = np.array([0, 0, 1, 3, 2, 1])
c = a[b]
print(a, '\n', c)
# using lists for selection
print(a[[1, 1, 1, 1, 0, 3]])
