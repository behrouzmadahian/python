class Graph(object):
    def __init__(self, vertices):
        self.V = vertices
        self.graph = [[0 for _ in range(self.V)] for i in range(self.V)]
    def printSolution(self, dist):
        print('Vertex tDistant from Source=')
        for node in range(self.V):
            print(node, dist[node])

    # A utility function to find the vertex with
    # minimum distance value, from the set of vertices
    # not yet included in shortest path tree
    def minDist(self, dist, sptSet):
        # initialize the minimum distance for next node
        min = float('inf')
        # search nearest vertext not in the shortes  path tree
        for v in range(self.V):
            if dist[v] < min and sptSet[v] == False:
                min = dist[v]
                min_index = v
        return min_index
    # Funtion that implements Dijkstra's single source
    # shortest path algorithm for a graph represented
    # using adjacency matrix representation
    def dijikstra(self, src):
        dist = [float('inf')] * self.V
        dist[src] = 0
        sptSet = [False] *self.V
        for count in range(self.V):
            u = self.minDist(dist, sptSet)
            # put the min distance vertex in the shortest path tree
            sptSet[u] = True
            # Update dist value of the adjacent vertices
            # of the picked vertex only if the current
            # distance is greater than new distance and
            # the vertex is not in the shortest path tree
            for v in range(self.V):
                if self.graph[u][v] > 0  and sptSet[v] ==False  and dist[v] > dist[u] + self.graph[u][v]:
                    dist[v] = dist[u] +self.graph[u][v]
        self.printSolution(dist)


g = Graph(9)
g.graph = [[0, 4, 0, 0, 0, 0, 0, 8, 0],
           [4, 0, 8, 0, 0, 0, 0, 11, 0],
           [0, 8, 0, 7, 0, 4, 0, 0, 2],
           [0, 0, 7, 0, 9, 14, 0, 0, 0],
           [0, 0, 0, 9, 0, 10, 0, 0, 0],
           [0, 0, 4, 14, 10, 0, 2, 0, 0],
           [0, 0, 0, 0, 0, 2, 0, 1, 6],
           [8, 11, 0, 0, 0, 0, 1, 0, 7],
           [0, 0, 2, 0, 0, 0, 6, 7, 0]
           ]

g.dijikstra(0)

