'''
Like Merge Sort, QuickSort is a Divide and Conquer algorithm. It picks an element as pivot
and partitions the given array around the picked pivot.
There are many different versions of quickSort that pick pivot in different ways.

1.Always pick first element as pivot.
2.Always pick last element as pivot (implemented below)
3.Pick a random element as pivot.
4.Pick median as pivot.
The key process in quickSort is partition(). Target of partitions is, given an array and an element x of array as pivot,
put x at its correct position in sorted array and put all smaller elements (smaller than x) before x, and put all
greater elements (greater than x) after x. All this should be done in linear time.
'''
def partition_pH(a, l, h):
    # l index of first element, h: index of last element
    pivot = a[h]
    start_ind = l
    for i in range(l, h):
        if a[i] < pivot:
            a[i], a[start_ind] = a[start_ind], a[i]
            start_ind += 1
    a[start_ind], a[h] = a[h], a[start_ind]
    return start_ind

aa = [1,5,3, 0, 4]
partition_pH(aa, 0, 4)

def quickSort1(a, l, h ):
    if l<h:
        sorted_ind = partition_pH(a, l, h)
        quickSort1(a, l, sorted_ind-1)
        quickSort1(a, sorted_ind+1, h)

a = [-5, 1, 2, 5, 4, 9, 7, 10, 200, 12, 15, 100, 0]
quickSort1(a, 0, len(a) - 1)
print(a)


def partition_pivot_h(a, l, h):
    # takes last element as pivot
    # keep track of index of smaller values put at beginning
    pivot = a[h]
    smaller_ind = l
    for i in range(l, h):
        if a[i] <= pivot:
            a[smaller_ind], a[i] = a[i], a[smaller_ind]
            smaller_ind += 1
    a[smaller_ind], a[h] = pivot, a[smaller_ind]
    print(smaller_ind)
    return smaller_ind


def partition_pivot_l(a, l, h):
    # put bigger elements at the end!
    pivot = a[l]
    bigger_ind = h
    for i in range(l + 1, h + 1):
        if a[i] >= pivot:
            a[bigger_ind], a[i] = a[i], a[bigger_ind]
            bigger_ind -= 1
    a[bigger_ind], a[l] = pivot, a[bigger_ind]
    return bigger_ind


#
def quickSort(a, l, h):
    if h > l:
        sorted_ind = partition_pivot_l(a, l, h)
        quickSort(a, l, sorted_ind - 1)
        quickSort(a, sorted_ind + 1, h)


a = [-5, 1, 2, 5, 4, 9, 7, 10, 200, 12, 15, 100, 0]
quickSort(a, 0, len(a) - 1)
print(a)






