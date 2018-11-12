'''
Binary Search Tree, is a node-based binary tree data structure which has the following properties:

The left subtree of a node contains only nodes with keys less than the node’s key.
The right subtree of a node contains only nodes with keys greater than the node’s key.
The left and right subtree each must also be a binary search tree.
There must be no duplicate nodes.
The above properties of Binary Search Tree provide an ordering among keys so that the operations like search,
 minimum and maximum can be done fast.
'''
class Node:
    def __init__(self, key):
        self.val = key
        self.left = None
        self.right = None

def bSearch (root, key):
    if root is None:
        return -1
    if root.val == key:
        return root
    if root.val > key:
        return bSearch(root.left, key)
    return bSearch(root.right, key)
'''
A new key is always inserted at leaf. We start searching a key from root till we hit a leaf node. 
Once a leaf node is found, the new node is added as a child of the leaf node.
The worst case time complexity of search and insert operations is O(h) where h is height of Binary Search Tree. 
Inorder traversal of BST always produces sorted output.
We can construct a BST with only Preorder or Postorder or Level Order traversal. 
'''
def bInsertion(root, key):
    if root is None:
        root = Node(key)
    if root.val < key:
        if root.right is None:
            root.right = Node(key)
        else:
            bInsertion(root.right, key)
    else:
        if root.left is None:
            root.left = Node(key)
        else:
            bInsertion(root.left, key)

root = Node(5)
root.left = Node(3)
root.left.left = Node(2)
root.left.right = Node(4)
root.right = Node(10)
root.right.right =Node(11)
root.right.left = Node(8)
print(bSearch(root, 8))
bInsertion(root, 15)
print(bSearch(root, 15))