'''
It should be a complete tree (i.e. all levels except last should be full).
Every node’s value should be greater than or equal to its child node (considering max-heap).
'''
class Node:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None

def countNodes(root):
    if root is None:
        return 0
    return  1 + countNodes(root.left) + countNodes(root.right)

def isComplete(root, index, size):
    if root is None: return True
    if index >= size: return False
    return isComplete(root.left, 2 * index + 1, size)  and isComplete(root.right, 2 * index + 2, size)

def isComplete1(root, index, size):
    if root is None:
        return True
    if index >= size:
        return False
    return isComplete1(root.left, 2*index+1, size) and isComplete1(root.right, 2*index + 2, size)

def isHeapUtil(root):
    # after checking completeness, checks the ordering of nodes!
    if root is None:
        return True
    if root.right is None and root.left is None:
        return True
    if root.left is None:
        if root.right:
            return False
    if root.left and root.left.val > root.val:
        return False
    if root.right and root.right.val > root.val:
        return False
    else:
        return isHeapUtil(root.left) and isHeapUtil(root.right)

def isHeap(root):
    size = countNodes(root)
    return isComplete(root, 0, size) and isHeapUtil(root)

root =  Node(10)
root.left =  Node(9)
root.right =  Node(8)
root.left.left =  Node(7)
root.left.right =  Node(6)
root.right.left =  Node(5)
root.right.right =  Node(4)
root.left.left.left =  Node(3)
root.left.left.right =  Node(2)
root.left.right.left =  Node(1)
print(isHeap(root))