from itertools import combinations
from collections import deque
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


# Depth-first search
def dfs(graph, start, visited=None):
    if visited is None:
        visited = set()
    visited.add(start)
    for next in graph[start] - visited:
        dfs(graph, next, visited)
    return visited


# Breadth-first search
def bfs(graph, root):
    distances = {}
    distances[root] = 0
    q = deque([root])
    while q:
        current = q.popleft()
        for neighbor in graph[current]:
            if neighbor not in distances:
                distances[neighbor] = distances[current] + 1
                q.append(neighbor)
    return distances


v = 100  # Number of vertices
e = 200  # Number of edges

# Initializing random adjacency matrix
adjMatrix = []
for i in range(v):
    adjMatrix.append([0] * v)
edges_pot = set(combinations(range(v), 2))
numb_of_added_edges = 0
while numb_of_added_edges != e:
    r = np.random.randint(len(edges_pot))
    edge = list(edges_pot)[r]
    adjMatrix[edge[0]][edge[1]] = 1
    adjMatrix[edge[1]][edge[0]] = 1
    numb_of_added_edges += 1
    edges_pot.remove(edge)

# Plotting
graph = nx.convert_matrix.from_numpy_matrix(np.array(adjMatrix))
nx.draw_networkx(graph)
plt.title('Visualization of the graph')
plt.tight_layout()
plt.show()

# Transferring the matrix into an adjacency list
adjlist = []
for row in range(v):
    row_set = set()
    for column in range(v):
        if adjMatrix[row][column]:
            row_set.add(column)
    adjlist.append(row_set)

# Finding the shortest path between two random vertices and printing one connected component
unvisited_vertices = set([i for i in range(v)])
short_dist = None
rand_vertex1 = list(unvisited_vertices)[np.random.randint(len(unvisited_vertices))]
rand_vertex2 = list(unvisited_vertices - {rand_vertex1})[np.random.randint(len(unvisited_vertices) - 1)]
visited_vertices = dfs(adjlist, rand_vertex1)
print('Connected components:')
print(visited_vertices)
if rand_vertex1 in visited_vertices and rand_vertex2 in visited_vertices:
    short_dist = bfs(adjlist, rand_vertex1)[rand_vertex2]
unvisited_vertices -= visited_vertices

# Printing other connected components
while len(unvisited_vertices):
    start = list(unvisited_vertices)[np.random.randint(len(unvisited_vertices))]
    visited_vertices = dfs(adjlist, start)
    print(visited_vertices)
    unvisited_vertices -= visited_vertices
print()

# Printing the shortest path between two random vertices
if short_dist is None:
    print(f'There is no path between {rand_vertex1} and {rand_vertex2}')
else:
    print(f'The shortest path between {rand_vertex1} and {rand_vertex2} is ', short_dist)
print()

# Printing first five rows of adjacency matrix
print('The part of adjacency matrix')
for row in range(v):
    if row == 6:
        break
    for column in range(v):
        print(adjMatrix[row][column], end=' ')
    print()
print()

# Printing first five rows of adjacency list
print('The part of adjacency list')
for row in range(v):
    if row == 6:
        break
    print(row, ':', adjlist[row])
