'''
given an array A[] of n numbers and another number x, determines whether or
not there exist two elements in S whose sum is exactly x.
'''
# array version:
def pairSumN(a, n):
    a = sorted(a)
    l = 0; h = len(a) -1
    while l<h:
        if a[l] +a[h] == n:
            return a[l], a[h]
        if a[l] + a[h] < n:
            l += 1
        else:
            h -= 1
    return False
a = [1,2,3,4,5]
print(pairSumN(a, 10))


'''
1) Initialize Binary Hash Map M[] = {0, 0, ...}
2) Do following for each element A[i] in A[]
   (a)	If M[x - A[i]] is set then print the pair (A[i], x - A[i])
   (b)	Set M[A[i]]
Assume we know the maximum element in the array!
O(n)
'''
def paisSumNHash(a, n, maxVal):
    binMap = [False] * max(n, (maxVal+1))
    for item in a:
        if item <n and binMap[n-item]:
            print(item, n-item)
        else:
            binMap[item] = True


def parisSumNHash1(a,n):
    h = {}
    for i in range(len(a)):
         if n > a[i]:
             try:
                h[n-a[i]]
                print(a[i], n-a[i])
                return
             except:
               h[a[i]] = a[i]
    print(False)
a = [1,2,3,4,5]

paisSumNHash(a, 10, 5)
print('===')
parisSumNHash1(a, 9)
