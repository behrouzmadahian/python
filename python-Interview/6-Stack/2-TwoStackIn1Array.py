'''
Assume total length of array is n.
solution start one stack from the beginning and second one from the end
'''
class TwoStacks:
    def __init__(self, n):
        self.size = n
        self.stack = [None]*n
        self.top1 = -1
        self.top2 = self.size
    def push1(self, x):
         # check to see if there is at least one empty space in array
        if self.top1 < self.top2 - 1:
            self.top1 += 1
            self.stack[self.top1]= x
        else:
            print('Stack Overflow')
            exit(1)
    def push2(self, x):
        if self.top1 < self.top2 - 1:
            self.top2 -=1
            self.stack[self.top2] =x
        else:
            print('Stack Overflow')
            exit(1)
    def pop1(self):
        if self.top1 > 0:
            x = self.stack[self.top1]
            self.top1 -=1
            return x
        else:
            print('Stack Underflow')
            exit(1)
    def pop2(self):
        if self.top2<self.size:
            x = self.stack[self.top2]
            self.top2 += 1
            return x
        else:
            print('Stack Underflow')
            exit(1)
ts = TwoStacks(5)
ts.push1(5)
ts.push2(10)
ts.push2(15)
ts.push1(11)
ts.push2(7)
print("Popped element from stack1 is " + str(ts.pop1()))
ts.push2(40)
print("Popped element from stack2 is " + str(ts.pop2()))