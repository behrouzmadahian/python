'''
1- keep 3 indies: smaller, equal, bigger
initially, all elements are in equal, iterate over equal  and move elements into smaller or larger groups
if A[equal]== pivot do nothing and advance the index!
'''
def arrange_around_pivot(A, index):
    pivot = A[index]
    smaller_ind, equal_ind, larger_ind = 0, 0, len(A) - 1
    while equal_ind < larger_ind:
        if A[equal_ind] < pivot:
            A[smaller_ind], A[equal_ind] = A[equal_ind], A[smaller_ind]
            smaller_ind += 1
            equal_ind += 1
        if A[equal_ind] == pivot:
            equal_ind += 1
        if A[equal_ind] > pivot:
            A[equal_ind], A[larger_ind] = A[larger_ind], A[equal_ind]
            larger_ind -= 1
        #print(A)
    return A

print(arrange_around_pivot([1,4,3,3,7,4,2,10, 5,3], 2))