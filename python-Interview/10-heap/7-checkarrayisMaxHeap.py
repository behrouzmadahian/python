'''
Given an array, how to check if the given array represents a Binary Max-Heap.
'''
def isHeap(a, i, size):
    if i >= size:
        return True
    l = 2 * i + 1
    r = 2 * i + 2
    if  l < size and a[l] > a[i]:
        return False
    if r < size and a[r] > a[i]:
        return False
    return isHeap(a, l, size) and isHeap(a, r, size)
a = [90 ,15,10,7,12,11]
print(isHeap(a, 0, len(a)))

#version2:
# last internal node is at index (n-2)/2
def isHeap1(a, i, size):
    if i > (size-2)/2:
        return True
    l = 2 * i + 1
    r = 2 * i + 2
    if  l < size and a[l] > a[i]:
        return False
    if r < size and a[r] > a[i]:
        return False
    return isHeap(a, l, size) and isHeap(a, r, size)

print(isHeap1(a, 0, len(a)))