
class Node:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None

def printGivenLevel(root, level):
    if root is None:
        return None
    if level ==1:
        print(root.val)
    elif level >1:
        printGivenLevel(root.left, level-1)
        printGivenLevel(root.right, level-1)

def height(root):
    if root is None:
        return 0
    lHeight = 1+ height(root.left)
    rHeight = 1+ height(root.right)
    return max(lHeight, rHeight)

def BFS(root):
    h = height(root)
    for i in range(1, h+1):
        printGivenLevel(root, i)

root = Node(1)
root.left = Node(2)
root.right = Node(3)
root.left.left = Node(4)
root.left.right = Node(5)
root.right.left = Node(6)
printGivenLevel(root, 3)
print('====')
BFS(root)
