'''
There are given n ropes of different lengths, we need to connect these ropes into one rope.
The cost to connect two ropes is equal to sum of their lengths. We need to connect the ropes with minimum cost.
For example if we are given 4 ropes of lengths 4, 3, 2 and 6. We can connect the ropes in following ways.
1) First connect ropes of lengths 2 and 3. Now we have three ropes of lengths 4, 6 and 5.
2) Now connect ropes of lengths 4 and 5. Now we have two ropes of lengths 6 and 9.
3) Finally connect the two ropes and all ropes have connected.
you can sort the array and continue
or make a minheap and continue
1. Make a min heap
add the root to the next root and add it to the heap
continue till heap only has 1 value
return that value!
'''
# makes the correct ordering for subtree rooted at i!
def minheapify(a,i, size):
    smallest_ind = i
    l = 2 * i + 1
    r = 2 * i + 2
    if l <size and a[l] < a[smallest_ind]:
        smallest_ind = l
    if r < size and a[r] < a[smallest_ind]:
        smallest_ind =r
    if smallest_ind != i:
        a[smallest_ind], a[i] = a[i], a[smallest_ind]
        minheapify(a, smallest_ind, size)

def minCostConnect_ropes(a):
    size = len(a)
    for i in range(size, -1, -1):
        minheapify(a, i, size)
    smallestCost = 0
    for i in range(size-1):
        minheapify(a, 0, len(a))
        firstMin = a[0]
        a = a[1:]
        minheapify(a, 0, len(a))
        secondMin  = a[0]
        smallestCost += firstMin +secondMin
        a[0] = firstMin +secondMin
    print(a,'===')

    return smallestCost

a = [4,2,3,6]
print(minCostConnect_ropes(a))




