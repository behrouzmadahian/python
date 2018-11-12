import numpy as np

'''
A program to check if a binary tree is BST or not
'''

class Node:
    def __init__(self, val):
        self.right = None
        self.left = None
        self.val = val


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

def isBSTUtil(root, mini, maxi):
    if root is None:
        return True
    if root.val < mini or root.val >maxi:
        return False
    return isBSTUtil(root.left, mini, root.val-1) and isBSTUtil(root.right, root.val +1 , maxi)


def isBSTutil1(root, mini, maxi):
    if root is None:
        return True
    if root.val < mini or root.val > maxi:
        return False
    return isBSTutil1(root.left, mini, root.val -1) and isBSTutil1(root.right, root.val+1, maxi)

root = Node(10)
bInsertion(root, 5)
bInsertion(root, 40)
bInsertion(root, 1)
bInsertion(root, 7)
bInsertion(root, 50)
#root.val =2
print(isBSTUtil(root, -np.inf, np.inf))