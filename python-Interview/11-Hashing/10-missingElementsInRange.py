'''
Given an array arr[0..n-1] of distinct elements and a range [low, high], find all numbers that are in range,
 but not in array. The missing elements should be printed in sorted order.
 Input: arr[] = {10, 12, 11, 15},
       low = 10, hight = 15
Output: 13, 14
'''
def missingInRange(a, l, h):
    ht = {}
    for item in a:
        ht[item] = item
    for i in range(l, h):
        if not i in ht:
            print(i)

a = [1,2,5,7]
missingInRange(a, 1, 10)


