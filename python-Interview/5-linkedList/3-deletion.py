'''
To delete a node from linked list, we need to do following steps.
1) Find previous node of the node to be deleted.
2) Changed next of previous node.
3) Free memory for the node to be deleted.
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

    def deleteNode(self, key):
        tmp = self.head
        if self.head is not None and tmp.data == key:
            self.head = tmp.next
            return
        while tmp is not None:
            if tmp.data == key:
                break
            prev = tmp
            tmp = tmp.next
        if tmp ==None:
            return
        prev.next = tmp.next
        tmp.next =None

llist = LinkedList()
llist.head = Node(1)
llist.push_atEnd(2)
llist.push_atEnd(3)
llist.push_atEnd(4)
llist.printList()
print('==')
llist.deleteNode(4)
llist.printList()