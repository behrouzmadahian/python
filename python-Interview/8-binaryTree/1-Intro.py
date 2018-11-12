'''
Unlike Arrays, Linked Lists, Stack and queues, which are linear data structures, trees are hierarchical data structures.
The topmost node is called root of the tree. The elements that are directly under an element are called its children.
 The element directly above something is called its parent.
 elements with no children are called leaves.
 Binary Tree: A tree whose elements have at most 2 children is called a binary tree.

 1- The maximum number of nodes at level ‘i’ of a binary tree is 2**(i-1)
 2- Maximum number of nodes in a binary tree of height ‘h’ is 2**(h) – 1
 note height starts from root(h) and at leaves becomes 1!
 3- In a Binary Tree with N nodes, minimum possible height or minimum number of levels is   Log(N+1) <base 2>
 4- A Binary Tree with L leaves has at least    LogL  + 1   levels < base of log is 2!>
 5- In Binary tree, number of leaf nodes is always one more than nodes with two children.
 ##
 Types of Binary Tree:
 1-Full Binary Tree:
    A Binary Tree is full if every node has 0 or 2 children.
    We can also say a full binary tree is a binary tree in which all nodes except leaves have two children.
    in a Full Binary tree, number of leaf nodes is number of internal nodes plus 1

 2-Complete Binary Tree:
    A Binary Tree is complete Binary Tree if all levels are completely filled except possibly the last level
    and the last level has all keys as left as possible

 3- Perfect Binary Tree:
    A Binary tree is Perfect Binary Tree in which all internal nodes have two children and all leaves are at same level.
    A Perfect Binary Tree of height h (where height is number of nodes on path from root to leaf) has 2**h – 1 node.

 4- Balanced Binary Tree
    A binary tree is balanced if height of the tree is O(Log n) where n is number of nodes

Height for a Balanced Binary Tree is O(Log n).
'''

class Node:
    def __init__(self, key):
        self.left = None
        self.right = None
        self.val = key

root = Node(1)
lChild = Node(2)
rChild = Node(3)
root.left = lChild
root.right = rChild
