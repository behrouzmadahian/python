'''
Given an array and its permutation indices, return the permuted array
example: a= [1,2,3,4], p=[3,1,2,0] -> [4, 2, 3, 1]
we make a permutation index negative after its applied
this solution is NOT intuitive but uses O(1) space.
'''

def permute1(a, p):
    for i in range(len(a)):
        if p[i] >=i :
            a[i], a[p[i]] = a[p[i]], a[i]
            #print(i, p, a)
            #print(a)
    return a
a = [20,1, 2, 3, 4,5,6,10]
p = [3, 4, 2, 7,5,1, 6, 0]
#res = [4, 2, 3, 1]
print(permute1(a, p))
