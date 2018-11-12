'''
Given a sorted array. Write a function that creates a Balanced Binary Search Tree using array elements.
Note: inorder Traversal of a BST gives a sorted array!
1) Get the Middle of the array and make it root.
2) Recursively do same for left half and right half.
      a) Get the middle of left half and make it left child of the root
          created in step 1.
      b) Get the middle of right half and make it right child of the
          root created in step 1.
'''

class Node:
    def __init__(self, val):
        self.right = None
        self.left = None
        self.val = val


def innOrder(root):
    if root:
        innOrder(root.left)
        print(root.val)
        innOrder(root.right)

def preOrder(root):
    if root:
        print(root.val)
        preOrder(root.left)
        preOrder(root.right)
def postOrder(root):
    if root:
        postOrder(root.left)
        postOrder(root.right)
        print(root.val)


def BSTfromArr(a, l, h):
    if l> h:
        return
    if l==h:
        root = Node(a[l])
        return root
    m = int((l+h)/2)
    root = Node(a[m])
    root.right = BSTfromArr(a, m+1,h)
    root.left = BSTfromArr(a, l, m-1)
    return root

def BSTfromSortedArr1(a, l, h):
    if l >h:
        return
    if l == h:
        root = Node(a[l])
        return root
    m = int((l+h)/2)
    root = Node(a[m])
    root.right = BSTfromSortedArr1(a, m+1, h)
    root.left = BSTfromSortedArr1(a, l, m-1)
    return root
a=[1,2,3,4,5,6,7,8]
root = BSTfromArr(a, 0, len(a)-1)
innOrder(root)
print('=====')
preOrder(root)
print('===')
postOrder(root)