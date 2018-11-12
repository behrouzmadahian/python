'''
Depth First Traversal (or Search) for a graph is similar to Depth First Traversal of a tree. The only catch here is,
unlike trees, graphs may contain cycles, so we may come to the same node again.
 To avoid processing a node more than once, we use a boolean visited array.
 The graph might not be connected, In order to traverse the whole graph we do DFS from each node!
 DFS1 codes this!
'''
from collections import  defaultdict
class Graph:
    def __init__(self):
        self.graph = defaultdict(list)
    def addEdge(self, u, v):
        self.graph[u].append(v)

    def DFSUtil(self, v, visited):
        # mark the current node as visited and print it
        visited[v] = True
        print (v)
        # recur for all the vertices adjacent to this vertex:
        for i in self.graph[v]:
            if visited[i] == False:
                self.DFSUtil(i, visited)
    def DFS(self, s):
        visited = [False] * len(self.graph)
        # call the recursive helper to print DFS traversal
        self.DFSUtil(s, visited)
    def DFS1(self):
        visited = [False] * len(self.graph)
        for i in range(len(self.graph)):
            if visited[i] == False:
                self.DFSUtil(i, visited)
    def DFS_util_redo(self, s, visited):
        # visit all nodes from S, DFS
        print(s)
        visited[s] = True
        for node in self.graph[s]:
            if visited[node] == False:
                self.DFS_util_redo(node, visited)
    def DFS_redo(self):
        visited = [False] * len(self.graph)
        for node in self.graph:
            if visited[node] == False:
                self.DFS_util_redo(node, visited)



g = Graph()
g.addEdge(0, 1)
g.addEdge(0, 2)
g.addEdge(1, 2)
g.addEdge(2, 0)
g.addEdge(2, 3)
g.addEdge(3, 3)
g.addEdge(4,5)
g.addEdge(5,5)
g.DFS(0)
print(g.graph)
print('==')
g.DFS1()
print('==')
g.DFS_redo()