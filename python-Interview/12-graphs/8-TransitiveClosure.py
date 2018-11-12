'''
Given a directed graph, find out if a vertex v is reachable from another vertex u for all vertex pairs (u, v)
 in the given graph. Here reachable means that there is a path from vertex u to v.
 The reach-ability matrix is called transitive closure of a graph.

 1. Create a matrix tc[V][V] that would finally have transitive closure of given graph.
     Initialize all entries of tc[][] as 0.
Call DFS for every node of graph to mark reachable vertices in tc[][].
In recursive calls to DFS, we don’t call DFS for an adjacent vertex if it is already marked as reachable in tc[][].
'''
from collections import  defaultdict
class Graph:
    def __init__(self, nVerticies):
        self.V = nVerticies
        # default dictionary to store graph
        self.graph = defaultdict(list)
        # tc for storing transitive closure matrix:
        self.tc = [[0 for j in range(self.V)] for i in range(self.V)]
    def addEdge(self, u, v):
        self.graph[u].append(v)

    def DFSUtil(self, s, v):
        # when calling the function we set s=v ( diagonal elements are all one!)
        # mark reachability from s to v as true
        self.tc[s][v] = 1
        # find all the verticies reachable through v
        for node in self.graph[v]:
            if self.tc[s][node] == 0:
                self.DFSUtil(s, node)

    def transitiveClosure(self):
        for i in range(self.V):
            self.DFSUtil(i, i)
        print(self.tc)
g = Graph(4)
g.addEdge(0, 1)
g.addEdge(0, 0)
g.addEdge(1, 0)
g.addEdge(2, 0)
g.addEdge(2, 3)
g.addEdge(3, 3)
g.transitiveClosure()