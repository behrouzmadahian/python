'''
When we delete a node, three possibilities arise.
1) Node to be deleted is leaf: Simply remove from the tree
2) Node to be deleted has only one child: Copy the child to the node and delete the child
3) Node to be deleted has two children: Find inorder successor of the node. Copy contents of
the inorder successor to the node and delete the inorder successor. Note that inorder predecessor can also be used.
iE: find the minimum value in the right subtree of a node and copy it to the node!

'''
class Node:
    def __init__(self, key):
        self.val = key
        self.left = None
        self.right = None

# Given non-empty binary search tree, return the node with minimum key value found in that tree.
def minValueNode(node):
    current = node
    while current.left is not None:
        current = current.left
    return current

def inorder(root):
    if root:
        inorder(root.left)
        print(root.val)
        inorder(root.right)

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

# given  a binary search tree and a key, this function deletes the key and returns the new root
def deleteNode(root, key):
    if root is None:
        return root
    # If key is same as root's key, then this is the node
     # to be deleted
    if root.val == key:
        # Leaf node:
        if root.left is None and root.right is None:
            root = None
            return root
        # Node with only one children:
        if root.left is None:
            tmp = root.right
            root = None
            return tmp
        elif root.right is None:
            tmp = root.left
            root = None
            return tmp
        # Node with two children
        temp = minValueNode(root.right)
        root.val = temp.val
        root.right = deleteNode(root.right, temp.val)

    elif root.val > key:
        root.left = deleteNode(root.left, key)
    else:
        root.right = deleteNode(root.right, key)
    return root


root = Node(50)
bInsertion(root, 30)
bInsertion(root, 20)
bInsertion(root, 40)
bInsertion(root, 70)
bInsertion(root, 60)
bInsertion(root, 80)
inorder(root)
root  =deleteNode(root, 20)
print('--')
inorder(root)