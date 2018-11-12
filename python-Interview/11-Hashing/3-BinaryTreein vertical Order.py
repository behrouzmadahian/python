'''
Given a binary tree, print it vertically.
Lets define a metric: Horizontal distance(HD).
1. Horizontal distance of the root is zero.
2. a right edge ( edge connecting to right subtree) is considered a  +1 horizontal distance.
3.  a left edge is considered a -1 horizontal distance
4. two nodes with the same horizontal distance are considered  to be on the same  vertical line
and should be printed at the same time.

We can do preorder traversal of the given Binary Tree. While traversing the tree,
we can recursively calculate HDs. We initially pass the horizontal distance as 0 for root.
For left subtree, we pass the Horizontal Distance as Horizontal distance of root minus 1.
For right subtree, we pass the Horizontal Distance as Horizontal Distance of root plus 1.
For every HD value, we maintain a list of nodes in a hash map. Whenever we see a node in traversal,
we go to the hash map entry and add the node to the hash map using HD as a key in map.
'''
class Node:
    def __init__(self, val):
        self.key = val
        self.right = None
        self.left = None

# Utility function to store vertical order in map 'm'
# 'hd' is horizontal distance of current node from root
# 'hd' is initially passed as 0
# m is a dictionary!
def getVerticalOrder(root, hd, m):
    if root is None:
        return
    # store current node in map 'm'
    try:
        m[hd].append(root.key) # if already exist, a node with this HD value in dict
    except:
        m[hd] = [root.key]
    # store nodes in left subtree
    getVerticalOrder(root.left, hd-1, m)
    # store nodes in right subtree
    getVerticalOrder(root.right, hd+1, m)
# main function to print vertical order:
def printVerticalOrder(root):
    m = dict()
    hd = 0
    getVerticalOrder(root, hd,m)

    for key in sorted(m):
        print(key, m[key])

root = Node(1)
root.left = Node(2)
root.right = Node(3)
root.left.left = Node(4)
root.left.right = Node(5)
root.right.left = Node(6)
root.right.right = Node(7)
root.right.left.right = Node(8)
root.right.right.right = Node(9)
printVerticalOrder(root)