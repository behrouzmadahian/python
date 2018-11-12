'''
In this version, we use min heap.
Minimum is at the root, we call heapify, starting from element 1 and decreasing the
size to heapify by 1 to the end ot the array!

'''
def minHeapify(arr, size, i):
    smallest_ind = i
    l = 2 * i + 1
    r = 2 * i + 2
    if l < size and arr[l] < arr[smallest_ind]:
        smallest_ind = l
    if r < size and arr[r] < arr[smallest_ind]:
        smallest_ind = r
    if i != smallest_ind:
        arr[i], arr[smallest_ind] = arr[smallest_ind], arr[i]
        minHeapify(arr, size, smallest_ind)

def heapsort2(a):
    n = len(a)
    for i in range(n, -1, -1):
        minHeapify(a, n, i)
    for i in range(n-1, 0, -1):
        a[0], a[i] = a[i], a[0]
        minHeapify(a, i, 0)

def heapsort(a):
    n = len(a)
    for i in range(n, -1, -1):
        minHeapify(a, n, i)
    for i in range(n-1, 0, -1):
        a[0], a[i] = a[i], a[0]
        minHeapify(a, i, 0)


def heapsort1(a):
    n = len(a)
    for i in range(n, -1, -1):
        minHeapify(a, n, i)
    for i in range(1, n, 1):
        a1 = a[i:]
        minHeapify(a1, n-i, 0)
        a[i:] = a1
        print(a[:i+1],a1[0],i)
    return  a


arr = [ 12, 11, 13, 5, 6, 7,-1,27,122]
heapsort1(arr)
print(arr)