'''
Topological sorting for Directed Acyclic Graph (DAG) is a linear ordering of
vertices such that for every directed edge uv, vertex u comes before v in the ordering.
 Topological Sorting for a graph is not possible if the graph is not a DAG.
 There can be more than one topological sorting for a graph!
The first vertex in topological sorting is always a vertex with in-degree as 0
(a vertex with no in-coming edges).
In DFS, we print a vertex and then recursively call DFS for its adjacent vertices.
In topological sorting, we need to print a vertex before its adjacent vertices.
In topological sorting, we use a temporary stack. We don’t print the vertex immediately,
we first recursively call topological sorting for all its adjacent vertices,
then push it to a stack. Finally, print contents of stack. Note that a vertex
is pushed to stack only when all of its adjacent vertices
(and their adjacent vertices and so on) are already in stack.
'''
from collections import  defaultdict
class Graph(object):
    def __init__(self, vertices):
        self.V = vertices
        self.graph = defaultdict(list)
    def addEdge(self, u, v):
        self.graph[u].append(v)
    def topologicalUtil(self, v, visited, stack):
        visited[v] = True
        for node in self.graph[v]:
            if visited[node] ==False:
                self.topologicalUtil(node, visited, stack)
        stack.append(v)
    def topologicalOrder(self):
        visited = [False] * self.V
        stack =[]
        for i in range(self.V):
            if visited[i]== False:
                self.topologicalUtil(i, visited, stack)
        for item in reversed(stack):
            print(item)
g = Graph(6)
g.addEdge(5,2)
g.addEdge(5,0)
g.addEdge(4,0)
g.addEdge(4,1)
g.addEdge(2,3)
g.addEdge(3,1)
g.topologicalOrder()