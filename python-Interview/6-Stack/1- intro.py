'''
Stack is a linear data structure which follows a particular order in which the operations are performed.
The order may be LIFO(Last In First Out) or FILO(First In Last Out).

Push: Adds an item in the stack. If the stack is full, then it is said to be an Overflow condition.
Pop: Removes an item from the stack. The items are popped in the reversed order in which they are pushed.
If the stack is empty, then it is said to be an Underflow condition.
Peek or Top: Returns top element of stack.
isEmpty: Returns true if stack is empty, else false.
push(), pop(), esEmpty() and peek() all take O(1) time.
There are two ways to implement a stack:
Using array
Using linked list
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
s = Stack()
s.push(10)
s.push(12)
s.printItems()
print(s.pop())
