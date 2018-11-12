'''
Given an array and index of pivot, arrange the elements such that elements to the left are smaller and
elements to the right are bigger that pivot
'''
def arrange_around_pivot(A, index):
    start = 0
    pivot = A[index]
    for i in range(len(A)):
        if A[i] < pivot:
            A[start], A[i] = A[i], A[start]
            start += 1
    end = len(A) - 1
    for i in reversed(range(len(A))):
        if A[i] < pivot:
            break
        elif A[i] > pivot:
            A[end], A[i] = A[i], A[end]
            end -= 1

    return(A)
a = [1,4,3,3,7,4,2,10, 5,3]
print(arrange_around_pivot(a, 5))

def arr_around_pivot(a, index):
    a[index], a[-1] = a[-1], a[index]
    smaller_ind = 0
    pivot = a[-1]
    for i in range(len(a)-1):
        if a[i]< pivot:
            a[smaller_ind], a[i] = a[i], a[smaller_ind]
            smaller_ind +=1
    a[smaller_ind],a[-1] = pivot, a[smaller_ind]
    return a
print(arr_around_pivot(a, 5))


