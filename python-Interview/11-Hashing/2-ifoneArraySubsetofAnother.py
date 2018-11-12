'''
I may be assumed that elements in both arrays are distinct
1) Create a Hash Table for all the elements of arr1[].
2) Traverse arr2[] and search for each element of arr2[] in the Hash Table. If element is not found then return 0.
3) If all elements are found then return 1.
since assumed that elements are distinct we dont worry about key collision!
'''
def is_subset(a1, a2):
    m = {}
    for item in a1:
        m[item] = item
    for item in a2:
        try:
            m[item]
        except: return False
    return True
a1 = [1,2,3,4,5]
a2 = [1,2,7]
print(is_subset(a1, a2))
