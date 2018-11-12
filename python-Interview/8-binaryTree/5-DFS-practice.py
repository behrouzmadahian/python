class Node:
    def __init__(self, val):
        self.val = val
        self. left = None
        self.right  = None

def inOrder_iter(root):
    current = root
    stack = []
    done = False
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
                done =True

def preorder_iter(root):
    current = root
    stack = []
    stack.append(current)
    while len(stack)>0:
        current = stack.pop()
        print (current.val)
        if current.right is not None:
            stack.append(current.right)
        if current.left is not None:
            stack.append(current.left)

def post_order_iter(root):
    stack1 =[]
    stack2 = []
    stack1.append(root)
    while len(stack1)>0:
        current = stack1.pop()
        stack2.append(current)
        if current.left is not None:
            stack1.append(current.left)
        if current.right is not None:
            stack1.append(current.right)
    print([item.val for item in stack2 [::-1]])

root = Node(1)
root.left = Node(2)
root.right = Node(3)
root.left.left = Node(4)
root.left.right = Node(5)
root.right.right =Node(7)
root.right.left = Node(6)
inOrder_iter(root)
print('==')
preorder_iter(root)
print('==')
post_order_iter(root)

