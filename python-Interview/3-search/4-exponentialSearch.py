'''
Exponential search involves two steps:
1. Find range where element is present
2. Do Binary Search in above found range.
runs in O(logn)

'''
import math
def binary_search_sorted_arr(a, low, high, value):
    if high < low:
        return -1
    mid = int((high+low)/2)
    if value == a[mid]:
        return mid
    elif value < a[mid]:
        return binary_search_sorted_arr(a,low, mid-1, value)
    else:
        return binary_search_sorted_arr(a,mid+1, high,  value)

def exponentialSearch(a, n, value):
    if value ==a[0]:
        return 0
    i = 1
    while i<n and value >= a[i]:
        i *= 2
    # do binary search
    return binary_search_sorted_arr(a, i//2, min(n,i), value)

a= [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610]
print(exponentialSearch(a, len(a)-1, 610 ))

def expSearch(a, value):
    step = int(math.sqrt(len(a)))
    for i in range(0,len(a), step):
        if a[i] > value:
            break
    return binary_search_sorted_arr(a, i-step, max(i, len(a)-1), value)
print(expSearch(a, 610))
