'''
Given pointer to the head node of a linked list, the task is to reverse the linked list.
We need to reverse the list by changing links between nodes.
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
        if tmp == None:
            return
        prev.next = tmp.next
        tmp.next = None

    def get_count(self):
        tmp = self.head
        count = 0
        while tmp:
            count += 1
            tmp = tmp.next
        return count

    def get_count_recursive(self, head):
        if head == None:
            return 0
        return 1 + self.get_count_recursive(head.next)

    def search(self, value):
        tmp = self.head
        while tmp:
            if tmp.data == value:
                return True
            tmp = tmp.next
        return False
    def reverse_L(self):
        prev = None
        current =self.head
        while current is not None:
            next = current.next
            current.next = prev
            prev = current
            current = next
        self.head =prev

llist = LinkedList()
llist.head = Node(5)
llist.push_atEnd(10)
llist.push_atEnd(25)
llist.push_atEnd(35)
llist.push_atEnd(45)
llist.reverse_L()
llist.printList()

