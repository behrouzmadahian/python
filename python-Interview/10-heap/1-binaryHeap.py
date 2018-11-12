'''
A Binary Heap is a Binary Tree with following properties.
1) It’s a complete tree (All levels are completely filled except possibly the last level and the last level has all
 keys as left as possible). This property of Binary Heap makes them suitable to be stored in an array.

2) A Binary Heap is either Min Heap or Max Heap. In a Min Binary Heap, the key at root must be minimum
among all keys present in Binary Heap. The same property must be recursively true for all nodes in Binary Tree.
 Max Binary Heap is similar to Min Heap.

A Binary Heap is a Complete Binary Tree. A binary heap is typically represented as array.

The root element will be at Arr[0].
Below table shows indexes of other nodes for the ith node,
-------------------
#Arr[(i-1)/2]	Returns the parent node
Arr[(2*i)+1]	Returns the left child node
Arr[(2*i)+2]	Returns the right child node
----------------------------
The traversal method used to achieve Array representation is Level Order.
Applications of Heaps:
1) Heap Sort: Heap Sort uses Binary Heap to sort an array in O(nLogn) time.
2) Priority Queue: Priority queues can be efficiently implemented using Binary Heap because it supports insert(),
delete() and extractmax(), decreaseKey() operations in O(logn) time.
Graph Algorithms: The priority queues are especially used in Graph Algorithms
like Dijkstra’s Shortest Path and Prim’s Minimum Spanning Tree.
Many problems can be efficiently solved using Heaps. See following for example.
a) K’th Largest Element in an array.
b) Sort an almost sorted array/
c) Merge K Sorted Arrays.

Operations on Min Heap:
1) getMini(): It returns the root element of Min Heap. Time Complexity of this operation is O(1).

2) extractMin(): Removes the minimum element from Min Heap. Time Complexity of this Operation is O(Logn)
as this operation needs to maintain the heap property (by calling heapify()) after removing root.

3) decreaseKey(): Decreases value of key. Time complexity of this operation is O(Logn).
If the new value of a node is greater than parent of the node, then we don’t need to do anything.
Otherwise, we need to traverse up to fix the violated heap property.

4) insert(): Inserting a new key takes O(Logn) time. We add a new key at the end of the tree.
IF new key is greater than its parent, then we don’t need to do anything. Otherwise, we need
to traverse up to fix the violated heap property.

5) delete(): Deleting a key also takes O(Logn) time. We replace the key to be deleted with -inf
by calling decreaseKey(). After decreaseKey(), the minus infinite value must reach root,
 so we call extractMin() to remove key.
'''
from heapq import heappush, heappop, heapify
# heappop - pop and return the smallest element from heap
# heappush - push the value item onto the heap, maintaining heap invariant
# heapify - transform list into heap, in place, in linear time
#time complexity of building a heap is O(n)
class MinHeap:
    def __init__(self):
        self.heap = []
    def parent(self, i):
        return (i-1)//2
    def leftChild(self, i):
        l_ind = 2 * i + 1
        if len(self.heap) > l_ind:
            return l_ind
        return None
    def rChid(self, i):
        r_ind = 2 * i + 2
        if len(self.heap) > r_ind:
            return r_ind
        return None
    def insertKey(self, k):
        heappush(self.heap, k)
    def decreaseKey(self,i, new_val):
        # decrease value of key at index i to new_val. it is assumed that new_val is smaller than heap[i]
        self.heap[i] = new_val
        while i != 0 and self.heap[self.parent(i)] > self.heap[i]:
            self.heap[i], self.heap[self.parent(i)] = self.heap[self.parent(i)], self.heap[i]
            i = self.parent(i)

    def extractMin(self):
        return heappop(self.heap)
    def deleteKey(self, i):
        # This functon deletes key at index i. It first reduces
        # value to minus infinite and then calls extractMin()
        self.decreaseKey(i, float('-inf'))
        self.extractMin()
    def getMin(self):
        return self.heap[0]


heapObj = MinHeap()
heapObj.insertKey(3)
heapObj.insertKey(2)
heapObj.insertKey(1)
heapObj.insertKey(15)
heapObj.insertKey(5)
heapObj.insertKey(4)
heapObj.insertKey(45)
print(heapObj.heap)
heapObj.deleteKey(4)
print(heapObj.heap)
heapObj.decreaseKey(3, -2)
print(heapObj.heap)