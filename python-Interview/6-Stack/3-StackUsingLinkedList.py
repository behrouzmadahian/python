class StackNode:
    def __init__(self, data):
        self.data = data
        self.next = None

class Stack:
    def __init__(self):
        self.root = None
    def isEmpty(self):
        return True if self.root is None else False
    def push(self,data):
        node = StackNode(data)
        node.next = self.root
        self.root = node
        print('Pushed data to stack=', data)
    def pop(self):
        if self.isEmpty():
            return None
        else:
            tmp = self.root.data
            self.root = self.root.next
            return tmp
    def peek(self):
        if self.isEmpty():
            return None
        else:
            return  self.root.data
    def printS(self):
        tmp = self.root
        while tmp:
            print(tmp.data)
            tmp = tmp.next


stack =Stack()
stack.push(10)
stack.push(20)
stack.push(30)
stack.printS()