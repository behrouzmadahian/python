'''
Given an undirected graph, how to check if there is a cycle in the graph.
We do a DFS traversal of the given graph. For every visited vertex ‘v’, if there is an adjacent ‘u’
such that u is already visited and u is not parent of v, then there is a cycle in graph.
O(n)
'''
from collections import  defaultdict
class Graph(object):
    def __init__(self, vertices):
        self.graph = defaultdict(list)
        self.V = vertices
    def addEdge(self, u, v):
        self.graph[u].append(v)
        self.graph[v].append(u)
    ## A recursive function that uses visited[] and parent to detect
    # cycle in sub graph reachable from vertex v.
    def isCyclicUtil(self, v, visited, parent):
        visited[v] = True
        for i in  self.graph[v]:
            ## If an adjacent vertex is visited and not parent of current vertex,
            # then there is a cycle
            if visited[i] == True:
                if parent != i:
                    return True
            else:
                if self.isCyclicUtil(i, visited, v):
                    return True
        return False
    def isCyclic(self):
        visited = [False] * self.V
        for i in range(self.V):
            if visited[i]== False:
                if self.isCyclicUtil(i, visited, -1):
                    return True
        return False

g = Graph(5)
g.addEdge(1, 0)
g.addEdge(0, 2)
g.addEdge(2, 0)
g.addEdge(0, 3)
g.addEdge(3, 4)
print(g.isCyclic())


