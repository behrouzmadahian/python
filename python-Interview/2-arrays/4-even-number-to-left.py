'''
reorder the array such that even numbers appear to the left
'''
def reorder(A):
    even_ind, odd_ind = 0, len(A)-1
    while even_ind < odd_ind:
        if A[even_ind]%2 == 0:
            even_ind += 1
        else:
            A[even_ind], A[odd_ind] = A[odd_ind], A[even_ind]
            odd_ind -= 1
    return A

print(reorder([1, 2, 3, 4, 5, 6]))