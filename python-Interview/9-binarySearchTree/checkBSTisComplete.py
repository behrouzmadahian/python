'''
A complete binary tree is a binary tree whose all levels except the last level are completely filled
 and all the leaves in the last level are all to the left side.
 In the array representation of a binary tree, if the parent node is assigned an index of ‘i’ and left
 child gets assigned an index of ‘2*i + 1’ while the right child is assigned an index of ‘2*i + 2’.
  If we represent the above binary tree as an array with the respective indices assigned to
  the different nodes of the tree above from top to down and left to right.
  Calculate the number of nodes (count) in the binary tree.
Start recursion of the binary tree from the root node of the binary tree with index (i) being set as
 0 and the number of nodes in the binary (count).
If the current node under examination is NULL, then the tree is a complete binary tree. Return true.
If index (i) of the current node is greater than or equal to the number of nodes in the binary tree (count) i.e.
 (i>= count), then the tree is not a complete binary. Return false.
Recursively check the left and right sub-trees of the binary tree for same condition.
For the left sub-tree use the index as (2*i + 1) while for the right sub-tree use the index as (2*i + 2).
Time complexity O(N)
'''
class Node:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None

def countNodes(root):
    if root is None: return 0
    return 1 + countNodes(root.left) +countNodes(root.right)

def isComplete(root, index, numNodes):
    if root is None:
        return True
    if index >= numNodes:
        return False
    return  isComplete(root.left, index * 2 + 1, numNodes)  and isComplete(root.right, index * 2 + 2, numNodes)


root = Node(1)
root.left = Node(2)
root.right = Node(3)
root.left.left = Node(4)
root.left.right = Node(5)
root.right.left = Node(6)

node_count = countNodes(root)
index = 0
print(isComplete(root, index, node_count))
