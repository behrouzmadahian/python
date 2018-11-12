'''
Given a Binary Search Tree (BST), convert it to a Binary Tree such that every key of the original BST
 is changed to key plus sum of all greater keys in BST.
 Solution: traverse the tree: reverse inorder: right root left
 key track of the sum of the nodes visited so far. for every node currently being visited,
 first and the node to the sum then make the node the new sum
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



def addSumGreater(root):
    current  = root
    stack = []
    done = 0
    sum = 0
    while not done:
        if current is not None:
            stack.append(current)
            current = current.right
        else:
            if len(stack)>0:
                current = stack.pop()
                sum += current.val
                current.val = sum
                #print(current.val)
                current = current.left
            else:
                done = 1
    return root

def addSumGreater2(root):
    stack =[]
    sum = 0
    done = False
    current = root
    while not done:
        if current is not None:
            stack.append(current)
            current = current.right
        else:
            if len(stack)>0:
                current = stack.pop()
                sum += current.val
                current.val = sum
                current = current.left
            else:
                done =True

root = Node(10)
bInsertion(root, 5)
bInsertion(root, 40)
bInsertion(root, 1)
bInsertion(root, 7)
bInsertion(root, 50)
innOrder(root)
print('=========')
addSumGreater(root)
innOrder(root)