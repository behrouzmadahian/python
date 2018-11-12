'''
Given an unsorted array that may contain duplicates. Also given a number k which is smaller than size of array.
Write a function that returns true if array contains duplicates within k distance.
we can do it in 2 for loops one runs to lengh of the array and the inner to   from i to i +k
O(nk)
second method Hashing:
We can solve this problem in Θ(n) time using Hashing. The idea is to one by one add elements to hash.
 We also remove elements which are at more than k distance from current element. Following is detailed algorithm.

1) Create an empty hashtable.
2) Traverse all elements from left from right. Let the current element be ‘arr[i]’
….a) If current element ‘arr[i]’ is present in hashtable, then return true.
….b) Else add arr[i] to hash and remove arr[i-k] from hash if i is greater than or equal to k

'''
def duplicateKApart(a, k):
    dct = {}
    for i in range(len(a)):
        if a[i] in dct:
            return True
        else:
            dct[a[i]] = a[i]
            if i >= k:
                del dct[a[i-k]]
    return False

a = [1, 2, 3, 4, 1, 2, 3, 4]
a1=[1, 2, 3, 4,1, 5]
print(duplicateKApart(a1, 4))