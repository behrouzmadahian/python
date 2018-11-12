'''
Given a directed graph, check whether the graph contains a cycle or not. Your function should return true
 if the given graph contains at least one cycle, else return false. For example, the following graph
  contains three cycles 0->2->0, 0->1->2->0 and 3->3, so your function must return true.
Depth First Traversal can be used to detect cycle in a Graph. DFS for a connected graph produces a tree.
 There is a cycle in a graph only if there is a back edge present in the graph. A back edge is an
  edge that is from a node to itself (self loop) or one of its ancestor in the tree produced by DFS.
'''
from collections import  defaultdict
class Graph:
    def __init__(self, vertices):
        self.V = vertices
        self.graph = defaultdict(list)
    def addEdge(self, u, v):
        self.graph[u].append(v)
    def isCyclicUtil(self, v, visited, rec_stack):
        # mark current node as visited
        # add it to recursion stack
        visited[v] = True
        rec_stack[v] = True
        # recur for all neighbors
        # if any neighbor is visited and in recStack then graph is cyclic
        for neighbor in self.graph[v]:
            if visited[neighbor] == False:
                if self.isCyclicUtil(neighbor, visited, rec_stack)== True:
                    return True
            elif rec_stack[neighbor] == True:
                return True
        #the node needs to be popped from recursion stack before function ends
        rec_stack[v] = False
        return False
    def isCyclic(self):
        visited = [False]*self.V
        recStack = [False] * self.V
        for node in range(self.V):
            if visited[node] == False:
                if self.isCyclicUtil(node, visited, recStack):
                    return True
        return False
g = Graph(4)
g.addEdge(0, 1)
g.addEdge(0, 2)
g.addEdge(1, 2)
g.addEdge(2, 0)
g.addEdge(2, 3)
g.addEdge(3, 3)
if g.isCyclic() == 1:
    print ("Graph has a cycle")
else:
    print("Graph has no cycle")