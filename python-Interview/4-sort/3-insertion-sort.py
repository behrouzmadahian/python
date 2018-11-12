'''
for index i:
looks into the array A[:i]
and shifts all elements in A[:i]
that are greater than A[i] one position forward, and insert a[i] into the new position.
takes maximum time if elements are sorted in reverse order.

'''
def insertion_sort(a):
    for i in range(1, len(a)):
        key = a[i]
        j = i-1
        # Move elements of arr[0..i-1], that are
        # greater than key, to one position ahead
        # of their current position
        while j >=0 and key < a[j]:
            a[j + 1] = a[j]
            j -= 1
        a[j+1] = key
    return a
a = [1,2,5,4,9,7,10,200,12,15,100,0]

print(insertion_sort(a))

def insSort(a):
    for i in range(1, len(a)):
        curr = a[i]
        j = i-1
        while j >=0 and a[j]> curr:
            a[j+1] =a[j]
            j -= 1
        a[j+1]= curr
    return a
print(insSort(a))

def insSort2(a):
    for i in range(1, len(a)):
        j = i-1
        curr = a[i]
        while j >=0 and a[j] > curr:
            a[j+1] = a[j]
            j -= 1
        a[j+1] = a[i]
    return a
print(insSort2(a))



