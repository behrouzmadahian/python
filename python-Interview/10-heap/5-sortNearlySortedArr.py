'''
Given an array of n elements, where each element is at most k away from its target position,
 devise an algorithm that sorts in O(n log k) time.
 1) Create a Min Heap of size k+1 with first k+1 elements. This will take O(k) time (See this GFact)
2) One by one remove min element from heap, put it in result array, and add a
 new element to heap from remaining elements.
'''
import numpy as np
def minheapify(a, size, i):
    smallest_ind = i
    l = 2 * i + 1
    r = 2 * i + 2
    if l < size and a[l] < a[smallest_ind]:
        smallest_ind = l
    if r < size and a[r] < a[smallest_ind]:
        smallest_ind = r
    if smallest_ind != i:
        a[smallest_ind], a[i] = a[i], a[smallest_ind]
        minheapify(a, size, smallest_ind)

def sortAlmostSorted(a, k):
    ak = a[:  k + 1]
    for i in range(len(ak), -1, -1):
        minheapify(ak, len(ak), i)
    kk = 0
    a[kk] = ak[0]
    for i in range(k+1, len(a)):
        ak[0] = a[i]
        minheapify(ak, len(ak), 0)
        kk = kk+1
        a[kk] = ak[0]
    ak = ak[1:]
    print(ak,'::')
    for i in range(len(ak)):
        minheapify(ak, len(ak), 0)
        kk += 1
        a[kk] = ak[0]
        ak = ak[1:]
    return a
a = [0,2, 6, 3, 12, 56, 8]
print(sortAlmostSorted(a, 3))