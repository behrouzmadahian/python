'''
Given a Binary Tree, convert it to a Binary Search Tree. The conversion must be done in such a
way that keeps the original structure of Binary Tree.
1) Create a temp array arr[] that stores inorder traversal of the tree. This step takes O(n) time.
2) Sort the temp array arr[]. Time complexity of this step depends upon the sorting algorithm.
 In the following implementation, Quick Sort is used which takes (n^2) time. This can be done in O(nLogn)
  time using Heap Sort or Merge Sort.
3) Again do inorder traversal of tree and copy array elements to tree nodes one by one. This step takes O(n) time.
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

def InorderNorecursion(root):
    current  = root
    stack = []
    done = 0
    inorderArr =[]
    while not done:
        if current is not None:
            stack.append(current)
            current = current.left
        else:
            if len(stack)>0:
                current = stack.pop()
                inorderArr.append(current.val)
                current = current.right
            else:
                done = 1
    return inorderArr


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

def InorderInsert(sortedArr, root):
    current  = root
    stack = []
    done = 0
    i = 0
    while not done:
        if current is not None:
            stack.append(current)
            current = current.left
        else:
            if len(stack)>0:
                current = stack.pop()
                current.val = sortedArr[i]
                i += 1
                current = current.right
            else:
                done = 1
    return root

def BTtoBST(root):
    a = InorderNorecursion(root)
    a.sort()
    print(a,'====')
    root = InorderInsert(a, root)
    return root

root = Node(10)
root.right = Node(7)
root.left = Node(2)
root.left.left = Node(8)
root.left.right = Node(4)

root = BTtoBST(root)
print(InorderNorecursion(root))