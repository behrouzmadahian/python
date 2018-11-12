import numpy as np
'''
O(n2)
Sorts the array element by element finding minimum
'''
def selectionSort(a):
    for i in range(len(a)):
        min_ind = i
        for j in range(i+1, len(a)):
            if a[j] < a[min_ind]:
                min_ind = j
        a[i], a[min_ind] = a[min_ind], a[i]
    return a

a = [1,2,5,4,9,7,10,200,12,15,100]
print(selectionSort(a))

def selSort(a):
    for i in range(len(a)):
        min = a[i]
        for j in range(i, len(a)):
            if a[j] < min:
                a[j], a[i] = a[i], a[j]
    return a
print(selSort(a))