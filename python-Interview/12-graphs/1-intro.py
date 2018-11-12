'''
Graph is a data structure that consists of following two components:
1. A finite set of vertices also called as nodes.
2. A finite set of ordered pair of the form (u, v) called as edge. The pair is ordered because (u, v)
is not same as (v, u) in case of directed graph(di-graph). The pair of form (u, v) indicates that
there is an edge from vertex u to vertex v. The edges may contain weight/value/cost.
Following two are the most commonly used representations of graph.
1. Adjacency Matrix
2. Adjacency List
Adjacency Matrix is a 2D array of size V x V where V is the number of vertices in a graph. Let the 2D array
be adj[][], a slot adj[i][j] = 1 indicates that there is an edge from vertex i to vertex j. Adjacency matrix
 for undirected graph is always symmetric. Adjacency Matrix is also used to represent weighted graphs.
  If adj[i][j] = w, then there is an edge from vertex i to vertex j with weight w.
'''