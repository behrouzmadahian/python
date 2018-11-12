'''
3,4,5,6,1,2  Min is 1
if array is not rotated min is at position zero
if rotated the previous element to min is bigger than min in the array
we check this for mid array!
if a[mid] > a[high]: search the upper half
else search the lower half
'''
def findMin(a, low, high):
       if high <low:
           return a[0]
       if high ==low:
           return a[low]
       mid = int((high + low)/2)
       if a[mid +1] < a[mid]:
           return a[mid+1]
       elif  a[mid] < a[mid - 1]:
            return a[mid]
       if a[high]> a[mid]:
           return findMin(a, low, mid-1)
       return findMin(a, mid+1, high)
a = [3,4,5,6,7,-1,0,1,2]
print(findMin(a, 0, len(a)-1))
def findMin1(a, l, h):
    if l > h:
        return a[0]
    if l == h:
        return a[l]
    mid = int((l + h)/2)
    if a[mid] > a[mid +1]:
        return a[mid+1]
    if a[mid] < a[mid-1]:
        return a[mid-1]
    if a[mid]> a[h]:
        return findMin1(a, mid+1, h)
    else:
        return findMin1(a, l, mid-1)

print(findMin1(a, 0, len(a)-1))


