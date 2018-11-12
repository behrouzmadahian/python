'''
Traversal time complexity: O(n)
'''
class Node:
    def __init__(self, key):
        self.left = None
        self.right = None
        self.val = key

def printInOrder(root):
        # prints in order traversal (Left root right)
        if root:
            printInOrder(root.left)
            print(root.val)
            printInOrder(root.right)
def printPreOrder(root):
    if root:
        print(root.val)
        printPreOrder(root.left)
        printPreOrder(root.right)
def printPostOrder(root):
    if root:
        printPostOrder(root.left)
        printPostOrder(root.right)
        print(root.val)

def printInorderNorecursion(root):
    current  = root
    stack = []
    done = 0
    while not done:
        if current is not None:
            stack.append(current)
            current = current.left
        else:
            if len(stack)>0:
                current = stack.pop()
                print(current.val)
                current = current.right
            else:
                done = 1
def iterativePreorder(root):
    stack = []
    stack.append(root)
    while len(stack) > 0:
        node = stack.pop()
        print(node.val)
        if  node.right is not None:
            stack.append(node.right)
        if node.left is not None:
            stack.append(node.left)

def iterativePostOrder(root):
    stack1 = []
    stack2 = []
    stack1.append(root)
    while len(stack1) > 0:
        current = stack1.pop()
        stack2.append(current)
        if current.left:
            stack1.append(current.left)
        if current.right:
            stack1.append(current.right)
    print([node.val for node in stack2[::-1]])

root = Node(1)
root.left = Node(2)
root.right = Node(3)
root.left.left = Node(4)
root.left.right = Node(5)
root.right.right =Node(7)
root.right.left = Node(6)

print('--')
print('PreOrder DFS')
printPreOrder(root)
print('---')

iterativePreorder(root)
print('PostOrder DFS')

printPostOrder(root)
print('---')
iterativePostOrder(root)
print('==============')

print('=============')
print('inOrder DFS')

printInOrder(root)
print('----')
printInorderNorecursion(root)

print('--------------------')
