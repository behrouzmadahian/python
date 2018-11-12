import math
'''
Like Binary Search, Jump Search is a searching algorithm for sorted arrays. The basic idea is to check fewer
elements (than linear search) by jumping ahead by fixed steps or skipping some elements
in place of searching all elements.
assume jump size of m, so we look into indicies a[m], a[2m], a[3m],...
once we find an interval that x falls in , we do a linear search in that interval.
Array is sorted.
time complexity: Q(n/m +m-1)  
best step size =sqrt(n) ( it minimizes <n/m +m-1>
Thus time Complexity is O(sqrt(n))
'''
def jumpSearch(arr, value, n):
    step = int(math.sqrt(n))
    i = 1
    while arr[i*step-1] <value:
        i += 1
    for j in range((i-1) * step, i* step,1):
        if arr[j] ==value:
            return True
    return False

a= [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610]
print(jumpSearch(a, 131, len(a)))
def jumpSearch1(arr, value):
    step = int(math.sqrt(len(a)))
    for i in range(0, len(a), step):
        if a[i]> value:
            break
    for k in range(i -step, i, 1):
        if a[k] == value:
            return k
    return False
print(jumpSearch1(a, 55))
