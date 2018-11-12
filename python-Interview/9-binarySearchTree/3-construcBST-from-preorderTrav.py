'''
Given preorder traversal of a binary search tree, construct the BST.
Method1:
the first element is the root.
find the index of first element that is greater than root <i>
a[i:] is part of the right subtree
a[1:i] part of the left subtree
use recurrence and the same logic to construct the BST
O(n2)
'''
class Node:
    def __init__(self, val):
        self.right = None
        self.left = None
        self.val = val
def inOrder(root):
    if root:
        inOrder(root.left)
        print(root.val)
        inOrder(root.right)
# Preorder ={10, 5, 1, 7, 40, 50} (root left right)
def BSTfromPreorder(preOrder,l,h):
    if l>h:
        return None
    root = Node(preOrder[l])
    if l == h:
        return root
    k = l+1
    while  root.val > preOrder[k] and k <= h:
        k += 1
    root.right = BSTfromPreorder(preOrder, k, h)
    root.left = BSTfromPreorder(preOrder, l + 1, k - 1)
    return root

def BSTfromPreorder1(a, l, h):
    if l > h:
        return None
    if l == h:
        root  = Node(a[l])
        return root
    root = Node(a[l])
    k = l + 1
    while a[k] < a[k] and k < h:
        k += 1
    root.right = BSTfromPreorder1(a, k, h)
    root.left = BSTfromPreorder1(a, l +1, k-1)
    return root

preorder = [10, 5, 1, 7, 40, 50]
root = BSTfromPreorder(preorder, 0, 5)
print('==========')
inOrder(root)


