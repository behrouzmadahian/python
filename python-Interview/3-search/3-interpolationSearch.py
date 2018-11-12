
'''
The Interpolation Search is an improvement over Binary Search for instances,
where the values in a sorted array are uniformly distributed. Binary Search always
goes to middle element to check. On the other hand interpolation search may go to different
locations according the value of key being searched. For example if the value of key is
closer to the last element, interpolation search is likely to start search toward the end side.

// The idea of formula is to return higher value of pos
// when element to be searched is closer to arr[hi]. And
// smaller value when closer to arr[lo]
pos = lo + [ (x-arr[lo])*(hi-lo) / (arr[hi]-arr[Lo]) ]
'''

def interpolationSearch(arr, l, h, value):
    if l>h:
        return -1
    if value < arr[l]:
        return -1
    pos = int(l + (value-arr[l]) * float(h-l) / (arr[h]-arr[l]))
    if  value == arr[pos]:
        return pos
    elif value > arr[pos]:
        return interpolationSearch(arr, pos + 1, h, value)
    else:
        return interpolationSearch(arr, l, pos - 1, value)

a= [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610]
print(interpolationSearch(a, 0, len(a)-1, 55))
