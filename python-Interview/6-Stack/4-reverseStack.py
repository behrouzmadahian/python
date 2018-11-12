'''
reversea stack using recursion.
The idea of the solution is to hold all values in Function Call Stack until the stack becomes empty.
When the stack becomes empty, insert all held items one by one at the bottom of the stack.
'''

class Stack:
    def __init__(self):
        self.stack = []
    def isEmpty(self):
        return len(self.stack)==0
    def push(self, item):
        self.stack.append(item)
    def pop(self):
        if not self.isEmpty():
            return self.stack.pop()
        return None
    def printItems(self):
        for item in self.stack:
            print (item)
    def insertAtButtom(self, item):
        # insert an element at the bottom of stack!
        if self.isEmpty():
            self.push(item)
        else:
            tmp = self.pop()
            self.insertAtButtom(item)
            self.push(tmp)
    def reveseStack(self):
        if not self.isEmpty():
            tmp = self.pop()
            self.reveseStack()
            self.insertAtButtom(tmp)


stack = Stack()
stack.push(4)
stack.push(3)
stack.push(2)
stack.push(1)
stack.printItems()
stack.reveseStack()
stack.printItems()