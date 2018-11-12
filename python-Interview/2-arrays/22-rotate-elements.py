
'''
given an array, put the first d elements to the end   1,2,3,4,5 , d = 2 -> 3,4,5,1,2
'''
def r_arr(a, d):
    for i in range(d-1):
        a[i], a[len(a)-i-1] = a[len(a)-i-1], a[i]
    return a


a = [1, 2, 3, 4, 5, 6, 7]
print(r_arr(a, 3))

def rotate_arr(a, d, n):
    tmp = a[:d]
    for i in range(d, n, 1):
        a[i-d] = a[i]
    a[n-d:] = tmp
    return a
# O(n) ans space coplexity O(d)
# now we do O(nd) and O(1) space complexity
def rotate_byOne(a, n):
    tmp = a[0]
    for i in range(1, n, 1):
        a[i-1] = a[i]
    a[-1]= tmp
    return a
def rotate_arr2(a,d,n):
    for i in range(d):
        a= rotate_byOne(a,n)
    return a

a= [1,2,3,4,5,6,7]
print(rotate_arr(a, 3, len(a)))
a= [1,2,3,4,5,6,7]

print(rotate_arr2(a, 3, len(a)))
