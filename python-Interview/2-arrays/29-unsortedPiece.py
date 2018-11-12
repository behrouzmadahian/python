'''
Given an unsorted array arr[0..n-1] of size n, find the minimum length subarray arr[s..e]
such that sorting this subarray makes the whole array sorted.
Example:
    [10, 12, 20, 30, 25, 40, 32, 31, 35, 50, 60],
    your program should be able to find that the sub array lies between the indexes 3 and 8.
'''
def f(a):
    for i in range(len(a)-1):
        if a[i] >a[i+1]:
            start_ind = i
            break
    for i in range(len(a)-1, -1, -1):
        print(i)
        if a[i]<a[i-1]:
            end_ind = i
            break
    return start_ind, end_ind
a = [10, 12, 20,28, 30, 25, 40, 32, 31, 35, 50, 60]
inds =f(a)
print(a[inds[0]:inds[1]+1])