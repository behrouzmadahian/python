# returns index of the value, and -1 if not found!
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
#print(binary_search_sorted_arr(a,0, len(a) ,10))
# O(logn)
# only returning True or false :
def binary_search(a, value):
    if len(a)==0:
        return False
    elif value ==a[len(a)//2]:
        return True
    elif value > a[len(a)//2] :
        return binary_search(a[len(a)//2 + 1 :], value)
    else:
        return binary_search(a[:len(a)//2 ], value)

print(binary_search(a, 5))

