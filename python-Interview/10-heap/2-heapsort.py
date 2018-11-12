'''
Heap sort is a comparison based sorting technique based on Binary Heap data structure. It is similar to selection
sort where we first find the maximum element and place the maximum element at the end.
We repeat the same process for remaining elements.
Heap Sort Algorithm for sorting in increasing order:
1. Build a max heap from the input data.
2. At this point, the largest item is stored at the root of the heap.
Replace it with the last item of the heap followed by reducing the size of heap by 1. Finally, heapify the root of tree.
3. Repeat above steps while size of heap is greater than 1.
heapify(arr, size, i):
given an array, its size and index i,  makes the subtree rooted at index i, a correct heap ordering.
note that the we have an array, and we want to do the reordering such that it represents a  heap.
I assume a max heap here.
'''
# when we heapify to make a heap from a random array, we start from the subtree rooted at the end of the array and
# move up the tree!
def maxHeapify(arr, size, i):
    largest_ind = i # initialize largest to root index (i)
    l = 2 * i + 1
    r = 2 * i + 2
    if l < size and arr[l] > arr[i]:
        largest_ind = l
    if r < size and arr[r] > arr[largest_ind]:
        largest_ind = r
    if i != largest_ind:
        arr[i], arr[largest_ind] = arr[largest_ind], arr[i]
        maxHeapify(arr, size, largest_ind)

def maxHeapify1(a, size, i):
    largest_ind = i
    l = 2 * i + 1
    r = 2 * i + 2
    if l <size and a[l] > a[largest_ind]:
        largest_ind = l
    if r < size and a[r] > a[largest_ind]:
        largest_ind = r
    if largest_ind !=i:
        a[largest_ind], a[i] = a[i], a[largest_ind]
        maxHeapify1(a, size, largest_ind)

def heapsort1(a):
    n = len(a)
    for i in range(n, -1,-1):
        maxHeapify1(a, n, i)
    for i in range(n-1, 0, -1):
        a[0], a[i] = a[i], a[0]
        maxHeapify1(a, i, 0)


arr = [1, 3, 2, 5, 6]
#maxHeapify(arr, 5, 0)
print(arr)
# #arr = [4, 10, 3, 5, 1]
for i in range(5, -1, -1):
    maxHeapify(arr, 5, i)
    print(i, arr)
print(arr)
#
def heapsort(arr):
    n = len(arr)
    # build a maxheap:
    for i in range(n, -1, -1):
        maxHeapify(arr, n, i)
    #one by one extract elements
    for i in range(n-1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]
        maxHeapify(arr, i, 0)

# Driver code to test above
arr = [ 12, 11, 13, 5, 6, 7,-1,27,122]
heapsort1(arr)
n = len(arr)
print ("Sorted array is")
print(arr)