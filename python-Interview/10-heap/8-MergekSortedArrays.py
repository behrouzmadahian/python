'''
Given k sorted arrays of size n each, merge them and print the sorted output.
1. Create an output array of size n*k.
2. Create a min heap of size k and insert 1st element in all the arrays into the heap
3. Repeat following steps n*k times.
     a) Get minimum element from heap (minimum is always at root) and store it in output array.
     b) Replace heap root with next element from the array from which the element is extracted.
      If the array doesn’t have any more elements, then replace root with infinite.
       After replacing the root, heapify the tree.
       O(nk*logk)
'''
class MinHeapNode:
    def __init__(self, val, i, j):
        self.val = val
        self.arrInd = i # index of the array element is coming from
        self.next = j   # index of the next element to be picked from array

def minHeapify(arr, size, i):
    smallest_ind = i
    l = 2*i+1
    r = 2*i+2
    if l <size and arr[l].val <arr[smallest_ind].val:
        smallest_ind = l
    if r <size and arr[r].val < arr[smallest_ind].val:
        smallest_ind = r
    if i  != smallest_ind:
        arr[i], arr[smallest_ind] = arr[smallest_ind], arr[i]
        minHeapify(arr, size, smallest_ind)

def mergeKsortedarrays(arrs, k, size):
    # arrs: array of arrays each have size n!
    results = []
    h = [arrs[i][0] for i in range(k)]
    for i in range(k, -1, -1):
        minHeapify(h, k, i)
    for i in range(size*k):
        results.append( h[0].val)
        if h[0].next < len(arrs[h[0].arrInd]) :
            h[0] = arrs[h[0].arrInd][h[0].next]
        else:
            h[0] = MinHeapNode(float('inf'), None, None)
        minHeapify(h, k, 0)
    return results

a1 = []
k = 4
for i in range(k):
    a1.append(MinHeapNode(2*i+1, 0, i+1))

a2 = []
for i in range(k):
    a2.append(MinHeapNode(2*i,1, i+1))
a3=[]
for i in range(k):
    a3.append(MinHeapNode(2*i-1,2, i+1))

print(mergeKsortedarrays([a1,a2,a3], 3, k))
