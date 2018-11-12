'''
Priority Queue is an extension of queue with following properties.
1) Every item has a priority associated with it.
2) An element with high priority is dequeued before an element with low priority.
3) If two elements have the same priority, they are served according to their order in the queue.
A typical priority queue supports following operations.
insert(item, priority): Inserts an item with given priority.
getHighestPriority(): Returns the highest priority item.
deleteHighestPriority(): Removes the highest priority item.
'''
class PriorityQueue:
    def __init__(self):
        self.PQ = []
    def insert(self, item, priority):
        self.PQ.append((item,priority))
    def getHighestPriority(self):
        maxPriorityItem = [-1, -1]
        for item in self.PQ:
            if item[1]> maxPriorityItem[1]:
                maxPriorityItem[0], maxPriorityItem[1] = item[0], item[1]
        return maxPriorityItem
    def deleteHighestPriority(self):
        MaxPriority_inds = []
        maxInd = None
        maxPriority =-1
        for i in range(len(self.PQ)):
            if self.PQ[i][1] > maxPriority:
                maxPriority = self.PQ[i][1]
                maxInd = i
        MaxPriority_inds.append(maxInd)
        for i in range(len(self.PQ)):
            if i != maxInd:
                if self.PQ[i][1] == maxPriority:
                    MaxPriority_inds.append(i)
        ind_toreturn = min(MaxPriority_inds)
        return self.PQ[ind_toreturn][0]

pq =PriorityQueue()
pq.insert(1,2)
pq.insert(2,1)
pq.insert(3,3)
print(pq.getHighestPriority())
print('==')
print(pq.deleteHighestPriority())