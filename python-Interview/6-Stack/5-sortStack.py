
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
    def peek(self):
        return self.stack[-1]
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
    def sortedInsert(self, item):
        if self.isEmpty() or item > self.peek():
            self.push(item)
        else:
            tmp = self.pop()
            self.sortedInsert(item)
            self.push(tmp)
    def sortstack(self):
        if not self.isEmpty():
            tmp = self.pop()
            self.sortstack()
            self.sortedInsert(tmp)





stack = Stack()
stack.push(4)
stack.push(3)
stack.push(2)
stack.push(1)
stack.push(10)
stack.printItems()
stack.sortstack()
stack.printItems()