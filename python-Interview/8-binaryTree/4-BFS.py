'''
There are basically two functions in this method. One is to print all nodes at a given level (printGivenLevel),
and other is to print level order traversal of the tree (printLevelorder).
printLevelorder makes use of printGivenLevel to print nodes at all levels one by one starting from root.
'''

class Node:
    def __init__(self, key):
        self.left = None
        self.right = None
        self.val = key

""" Compute the height of a tree--the number of nodes
    along the longest path from the root node down to
    the farthest leaf node
"""
def height(node):
    if node is None:
        return 0
    else:
        # compute the height of each subtree
        lheight = 1 + height(node.left)
        rheight = 1 + height(node.right)
        if lheight > rheight:
            return lheight
        else:
            return rheight

def isBalanced(root):
    if root is None:
        return 0
    lheight = 1 + height(root.left)
    rheight = 1 + height(root.right)
    return lheight == rheight

def printGivenLevel(root,level):
    if root is None:
        return
    if level == 1:
        print(root.val)
    elif level > 1:
        printGivenLevel(root.left, level-1)
        printGivenLevel(root.right, level-1)

def printLevelOrder(root):
    h = height(root)
    for i in range(1, h+1):
        printGivenLevel(root, i)
        print('--')

root = Node(1)
root.left = Node(2)
root.right = Node(3)
root.left.left = Node(4)
root.left.right = Node(5)
root.right.left = Node(6)
printGivenLevel(root, 3)
print('====')
printLevelOrder(root)
print(isBalanced(root))