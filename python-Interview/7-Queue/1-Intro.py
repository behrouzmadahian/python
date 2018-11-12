'''
The order is First In First Out (FIFO).
A good example of queue is any queue of consumers for a resource where the consumer that came first is served first.
The difference between stacks and queues is in removing. In a stack we remove the item the most recently added;
 in a queue, we remove the item the least recently added.

Operations on Queue:
Mainly the following four basic operations are performed on queue:

Enqueue: Adds an item to the queue. If the queue is full, then it is said to be an Overflow condition.
Dequeue: Removes an item from the queue. The items are popped in the same order in which they are pushed.
         If the queue is empty, then it is said to be an Underflow condition.
Front: Get the front item from queue.
Rear: Get the last item from queue.
Queue is used when things don’t have to be processed immediately,
but have to be processed in First In First Out order like Breadth First Search.
'''

class Queue:
    def __init__(self):
        self.Queue = []
    def enqueue(self, item):
        self.Queue.append(item)
    def isEmpty(self):
        return len(self.Queue)==0
    def dequeue(self):
        if self.isEmpty():
            print('Underflow condition- Queue is empty nothing to dequeue')
        else:
            self.Queue.pop(0)
    def front(self):
        # return the first item in Queue
        return self.Queue[0]
    def rear(self):
        # return the last item in Queue
        return self.Queue[-1]

    def printQ(self):
        print(self.Queue)


q = Queue()
q.enqueue(10)
q.enqueue(15)
q.enqueue(5)
q.enqueue(20)
q.printQ()
q.dequeue()
q.dequeue()
q.printQ()


