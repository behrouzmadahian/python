'''
A node can be added in three ways
1) At the front of the linked list
2) After a given node.
3) At the end of the linked list.
'''


class Node:
    def __init__(self, data):
        self.data = data
        self.next = None


# LinkedListClass
class LinkedList:
    # function to initialize the linkedList object
    def __init__(self):
        self.head = None

    def printList(self):
        temp = self.head
        while temp:
            print(temp.data)
            temp = temp.next
    def push_inFront(self, new_data):
        new_node = Node(new_data)
        new_node.next = self.head
        self.head = new_node

    def pushAfter_node(self, prevNode, new_data):
        new_node = Node(new_data)
        new_node.next = prevNode.next
        prevNode.next = new_node

    def push_atEnd(self, new_data):
        '''
        we have to traverse the list till end and then change the next of last node to new node.
        '''
        new_node = Node(new_data)
        if self.head is None:
            self.head = new_node
        tmp = self.head
        while tmp.next:
            tmp = tmp.next
        tmp.next = new_node

llist = LinkedList()
first = Node(3);second = Node(4);third = Node(5)
llist.head = first;llist.head.next = second;second.next = third
llist.printList()
# Add a node to the front: the new element becomes the head
print('=========')
llist.push_inFront(10)
llist.printList()

llist.pushAfter_node(second, 25)
llist.printList()
llist.push_atEnd(33)
llist.printList()
