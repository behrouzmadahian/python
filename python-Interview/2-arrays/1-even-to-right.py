'''
write a python program that given an array, puts all even numbers to the right side
'''
def even_to_right(A):
    start, end = 0, len(A) - 1
    while start < end:
        if A[start] %2 != 0:
            start += 1
        else:
            A[start], A[end] = A[end], A[start]
            end -= 1
    return A

print(even_to_right([4,6,5,1,2,3]))

def evRight1(a):
    od_ind = 0
    for i in range(len(a)):
        if a[i] % 2 == 1:
            a[od_ind], a[i] = a[i], a[od_ind]
            od_ind += 1
    return a
print(evRight1([4,6,5,1,2,3]))



