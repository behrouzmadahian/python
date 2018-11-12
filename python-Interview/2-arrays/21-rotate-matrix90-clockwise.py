#transpose it and then switch the columns: first goes last, ..
# space complexity O(n2)
#time complexity O(n2)
import numpy as np
def rotate(a):
    b = np.empty((a.shape[1], a.shape[0]))
    #transpose
    for i in range(a.shape[1]):
        b[:, i] = a[-i-1,:]
    return b
a = np.arange(1,17).reshape(4,4)
print(a)
print(rotate(a))



