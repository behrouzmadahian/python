'''
Given an Undirected graph G and an integer K, K-cores of the graph are connected components that are
left after all vertices of degree less than k have been removed.
The standard algorithm to find a k-core graph is to remove all the vertices that have degree less than- ‘K’
from the input graph. We must be careful that removing a vertex reduces the degree of all the vertices adjacent to it,
 hence the degree of adjacent vertices can also drop below-‘K’. And thus, we may have to remove those vertices also.
 To implement above algorithm, we do a modified DFS on the input graph and delete all the vertices having degree less
  than ‘K’, then update degrees of all the adjacent vertices, and if
   their degree falls below ‘K’ we will delete them too.
   O(V + E) where V is number of vertices and E is number of edges.
'''
from collections import  defaultdict
class Graph:
    def __init__(self, verticies):
        self.V = verticies # number of vertices
        self.graph = defaultdict(list)

    # function to add an edge to undirected graph
    def addEdge(self, u, v):
        self.graph[u].append(v)
        self.graph[v].append(u)
    # a recursive function to call DFS starting from v.
    # it returns true if vDegree of v after processing is less than k else False
    # also, updates vDegree of adjacent if vDegree of v is less than k
    # and if vDegree of a processed adjacent becomes less than k, then it reduces vDegree of v also
    def DFSUtil(self, v, visited, vDegree, k):
        # mark the current node as visited
        visited[v] = True
        # recur for all vertices adjacent to v
        # if vDegree of v is less than k, then vDegree of adjacent must be reduced( since later we remove v!)
        for node in self.graph[v]:
            if vDegree[v] < k:
                vDegree[node] -= 1
            if visited[node] == False:
                # If vDegree of adjacent after processing becomes
                # less than k, then reduce vDegree of v also
                if self.DFSUtil(node, visited, vDegree, k):
                    vDegree[v] -=1
        return vDegree[v] < k # TRUE if degree of v less than k

    def printKCores(self, k):
        # initialization:
        visited = [False]* self.V
        # store vDegrees of al verticies:
        vDegree = [0] * self.V
        for i in self.graph:
            vDegree[i] = len(self.graph[i])
        # choose any vertex as starting vertex
        self.DFSUtil(0, visited, vDegree, k)
        # DFS traversal to update vDegree of all vertices in case they are unconnected
        for i in range(self.V):
            if visited[i] == False:
                self.DFSUtil(i, visited, vDegree, k)
        #printing k cores:
        for v in range(self.V):
            if vDegree[v] >= k:
                print(str("\n [ ") + str(v) + str(" ]"))
                # Traverse adjacency list of v and print only
                # those adjacent which have vvDegree >= k
                # after DFS
                for i in self.graph[v]:
                    if vDegree[i] >= k:
                        print( "-> " + str(i))


k = 3
g1 = Graph (9)
g1.addEdge(0, 1)
g1.addEdge(0, 2)
g1.addEdge(1, 2)
g1.addEdge(1, 5)
g1.addEdge(2, 3)
g1.addEdge(2, 4)
g1.addEdge(2, 5)
g1.addEdge(2, 6)
g1.addEdge(3, 4)
g1.addEdge(3, 6)
g1.addEdge(3, 7)
g1.addEdge(4, 6)
g1.addEdge(4, 7)
g1.addEdge(5, 6)
g1.addEdge(5, 8)
g1.addEdge(6, 7)
g1.addEdge(6, 8)
g1.printKCores(k)