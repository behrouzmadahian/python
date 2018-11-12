'''
Given two Linked Lists, create union and intersection lists that contain union and intersection of the elements
 present in the given lists. Order of elments in output lists doesn’t matter.
'''
class Node:
    def __init__(self, val):
        self.val = val
        self.next = None
class linkedList:
    def __init__(self):
        self.head = None
    def push(self,val):
        current = self.head
        while current.next:
            current = current.next
        current.next = Node(val)

def intersection(ll1, ll2):
    h = {}
    current = ll1.head
    results ={}
    while current:
        try:
            h[current.val]
        except:
            h[current.val] = current
        current = current.next
    current = ll2.head
    while current:
        try:
            results[h[current.val]] = current.val
        except:
            pass
        current = current.next
    for item in results:
        print(results[item])

def union(ll1,ll2):
    h = {}
    current = ll1.head
    while current:
        try:
            h[current.val]
        except:
            h[current.val] = current
        current = current.next
    current = ll2.head
    while current:
        try:
            h[current.val]
        except:
            h[current.val] = current
        current = current.next
    for key in h:
        print(h[key].val)
ll1 = linkedList()
ll1.head = Node(1)
ll1.push(2)
ll1.push(3)
ll1.push(4)
ll2 = linkedList()
ll2.head = Node(1)
ll2.push(5)
ll2.push(7)
ll2.push(4)
ll2.push(4)

intersection(ll1,ll2)
print('===')

union(ll1,ll2)


