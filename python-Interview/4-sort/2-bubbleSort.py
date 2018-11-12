'''
Bubble Sort is the simplest sorting algorithm that works by repeatedly
swapping the adjacent elements if they are in wrong order.
on the first pass, the last element is sorted,
On the second pass the last two elements will be sorted,..
O(n2)
'''
def bubble_sort(a):
    for i in range(len(a)):
        for j in range(len(a)-i-1):
            if a[j] > a[j+1]:
                a[j], a[j+1] = a[j+1], a[j]
    return a
a = [1,2,5,4,9,7,10,200,12,15,100,0]

print(bubble_sort(a))

def bubSort(a):
    for i in range(len(a)):
        for j in range(len(a)-i-1):
            if a[j] > a[j+1]:
                a[j], a[j+1] = a[j+1], a[j]
    return a
print(bubSort(a))