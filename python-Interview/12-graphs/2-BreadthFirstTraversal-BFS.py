'''
Breadth First Traversal (or Search) for a graph is similar to Breadth First Traversal of a tree
The only catch here is, unlike trees, graphs may contain cycles, so we may come to the same node again.
 To avoid processing a node more than once, we use a boolean visited array.
 For simplicity, it is assumed that all vertices are reachable from the starting vertex.

 Note that the above code traverses only the vertices reachable from a given source vertex.
  All the vertices may not be reachable from a given vertex (example Disconnected graph).
  To print all the vertices, we can modify the BFS function to do traversal starting from
  all nodes one by one (Like the DFS modified version).

Time Complexity: O(V+E) where V is number of vertices in the graph and E is number of edges in the graph.

'''
# Program to print BFS traversal from a given source
# vertex. BFS(s) traverses vertices reachable from s.
from collections import defaultdict
# This class represents a directed graph using adjacency  list representation
class Graph:
    def __init__(self):
        self.graph = defaultdict(list)
    def addEdge(self, u, v):
        self.graph[u].append(v)
    def BFS(self, s):
        # mark all verticies as not visited
        visited = [False]* len(self.graph)
        # createa queue for BFS:
        queue = []
        # mark the source node as visited and enqueue it:
        queue.append(s)
        visited[s] = True
        while queue:
            s = queue.pop(0)
            print(s)
            # get all adjacent vertices of the dequeued vertex s
            # if a adjacent has not been visited,
            # then mark it visited and enqueue it
            for i in self.graph[s]:
                if visited[i] == False:
                    queue.append(i)
                    visited[i] = True

    def BFS_redo(self, s):
        visited = [False]* len(self.graph)
        queue = [s]
        visited[s]= True
        while queue:
            current = queue.pop(0)
            print(current)
            for node in self.graph[current]:
                if visited[node] ==False:
                    queue.append(node)
                    visited[node] = True
    def BFS_allNodes(self):
        visited = [False] * len(self.graph)
        for s in self.graph:
            if visited[s] ==False:
                queue = [s]
                visited[s] =True
                while queue:
                    current = queue.pop(0)
                    print(current)
                    for node in self.graph[current]:
                        if visited[node] == False:
                            queue.append(node)
                            visited[node] =True


g = Graph()
g.addEdge(0, 1)
g.addEdge(0, 2)
g.addEdge(1, 2)
g.addEdge(2, 0)
g.addEdge(2, 3)
g.addEdge(3, 3)
g.addEdge(4, 5)
g.addEdge(5, 6)
g.addEdge(6,6)
g.BFS(2)
print('==')
g.BFS_redo(0)
print('===')
g.BFS_allNodes()