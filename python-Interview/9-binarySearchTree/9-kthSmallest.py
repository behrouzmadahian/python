
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
def inorder(root):
    if root:
        inorder(root.left)
        print(root.val)
        inorder(root.right)
def kSmallest(root, k):
    stack =[]
    current = root
    done = False
    i = 0
    kthSmallets = None
    while not done:
        if current is not None:
            stack.append(current)
            current = current.left
        else:
            if len(stack) >0:
                current = stack.pop()
                print(current.val,'==')
                i += 1
                if i ==k:
                    kthSmallets = current.val
                    break
                current = current.right
            else:
                done =True
    return kthSmallets



root = Node(10)
bInsertion(root, 5)
bInsertion(root, 40)
bInsertion(root, 1)
bInsertion(root, 7)
bInsertion(root, 50)
#inorder(root)

print(kSmallest(root, 3))



