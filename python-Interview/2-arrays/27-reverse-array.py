'''
[1,2,3,4,5] -> [5,4,3,2,1]
'''
def reverse_arr(a):
    l, r = 0, len(a)-1
    while l < r:
        a[l], a[r] = a[r], a[l]
        r -=1
        l +=1
    return a
print(reverse_arr([1,2,3,4,5]))
def r_arr(A):
    l, r = 0, len(A)-1
    while l< r:
        A[l], A[r] = A[r], A[l]
        l += 1
        r -= 1
    return A
print(r_arr([1,2,3,4,5]))
