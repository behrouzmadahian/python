'''
Following is a stack based iterative solution that works in O(n) time.
1. Create an empty stack.
2. Make the first value as root. Push it to the stack.
3. Keep on popping while the stack is not empty and the next value is greater than stack’s top value.
Make this value as the right child of the last popped node. Push the new node to the stack.

4. If the next value is less than the stack’s top value, make this value as the left child of the stack’s top node.
 Push the new node to the stack.
5. Repeat steps 2 and 3 until there are items remaining in pre[].
O(n)
'''
class Node:
    def __init__(self, val):
        self.right = None
        self.left = None
        self.val = val
def inOrder(root):
    if root:
        inOrder(root.left)
        print(root.val)
        inOrder(root.right)
# Preorder ={10, 5, 1, 7, 40, 50} (root left right)
def BSTfromPreorderIter(preOrder):
    stack = []
    root = Node(preOrder[0])
    stack.append(root)
    for i in range(1, len(preOrder)):
        current = None
        while len(stack) > 0 and stack[-1].val < preOrder[i] :
            current = stack.pop()
            print(current.val,'===')
        if current:
            current.right = Node(preOrder[i])
            stack.append(current.right)
        else:
            left = Node(preOrder[i])
            stack[-1].left =left
            stack.append(left)
    return root

preorder = [10, 5, 1, 7, 40, 50]
root = BSTfromPreorderIter(preorder)
inOrder(root)

