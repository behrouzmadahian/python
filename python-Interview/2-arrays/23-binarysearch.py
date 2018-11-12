# returns index of the value, and -1 if not found!
# assumes sorted array!
def binary_search_sorted_arr(a, low, high, value):
    if high < low:
        return -1
    mid = int((high+low)/2)
    if value == a[mid]:
        return mid
    elif value < a[mid]:
        return binary_search_sorted_arr(a,low, mid-1, value)
    else:
        return binary_search_sorted_arr(a,mid+1, high,  value)

a = [1,2,3,4,5,6,7,8,9,10]
print(binary_search_sorted_arr(a,0, len(a) ,6))
# O(logn)

def bSearch(a, l, h, val):
    if l>h:
        return -1
    m = int((l+h)/2)
    if a[m] == val:
        return m
    if a[m]> val:
        bSearch(a, l, m-1, val)
    else:
        bSearch(a, m+1, h, l)
print(bSearch(a,0, len(a), 6))
