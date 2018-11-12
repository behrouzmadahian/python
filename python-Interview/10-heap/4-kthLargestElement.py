'''
Find kth largest element in array
'''
def maxHeapify(a, size, i):
    # heapifies the subtree rooted at index i
    largest_ind = i
    l = 2 * i + 1
    r = 2 * i + 2
    if l < size and a[l] >a[largest_ind]:
        largest_ind = l
    if r <size and a[r] > a[largest_ind]:
        largest_ind = r
    if largest_ind != i:
        a[largest_ind], a[i] = a[i], a[largest_ind]
        maxHeapify(a, size, largest_ind)

def kthLargest(a, kl):
    n = len(a)
    for i in range(n, -1, -1):
        maxHeapify(a, n, i)
    k = 1
    for i in range(n-1, 0, -1):
        a[0], a[i] = a[i], a[0]
        if k == kl:
            break
        maxHeapify(a, i, 0)
        k +=1
    return a[-k]

arr = [ 12, 11, 13, 5, 6, 7,-1,27,122]
print(kthLargest(arr, 4))


