'''
Like arrays, Linked List is a linear data structure. Unlike arrays,
linked list elements are not stored at contiguous location; the elements are linked using pointers.
Advantages:
dynamic size.
ease of insertion and deletion.+
Drawbacks:
1) Random access is not allowed. We have to access elements sequentially starting from the first node.
    So we cannot do binary search with linked lists.
2) Extra memory space for a pointer is required with each element of the list.
Each node in a list consists of at least two parts:
1) data
2) pointer to the next node
'''
#node class:
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

#LinkedListClass
class LinkedList:
    # function to initialize the linkedList object
    def __init__(self):
        self.head = None
        
    def printList(self):
        temp = self.head
        while temp:
            print(temp.data)
            temp = temp.next
# a simple LinkedLinst
if __name__ == '__main__':
    llist = LinkedList()
    llist.head = Node(1)
    second = Node(2)
    third = Node(3)
    llist.head.next = second
    second.next = third
    print(llist.head.next.next.data)
    llist.printList()