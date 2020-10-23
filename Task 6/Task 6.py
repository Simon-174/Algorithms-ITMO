from itertools import combinations
import random
import networkx as nx
import numpy as np
from timeit import default_timer
import matplotlib.pyplot as plt


v = 40  # Number of vertices
edges = set(combinations(range(v), 2))  # The set of edges that have not been added yet
max_weight = 100  # Maximum weight of edges
number_of_iterations = 3  # Variable for counting average execution time

# Initializing a graph
graph = nx.Graph()
graph.add_nodes_from(range(v))

# Lists for plotting
x_data = [i for i in range(v, len(edges)+1)]  # The number of edges
y_data_dij = []  # The set of execution times for Dijkstra algorithm
y_data_bf = []  # The set of execution times for Bellman Ford algorithm

# Variables for average execution times
dij_avg_time = 0
bf_avg_time = 0

# Adding edges to make a graph connected
for j in range(v-1):
    edge = (j, np.random.randint(j+1, v))
    graph.add_edge(*edge, weight=np.random.randint(1, max_weight + 1))
    edges.remove(edge)

# Till the number of edges is maximum for given number of vertices:
# Add one random edge, count execution times
while len(edges):
    r = np.random.randint(len(edges))
    edge = list(edges)[r]

    graph.add_edge(*edge, weight=np.random.randint(1, max_weight + 1))
    edges.remove(edge)

    for i in range(number_of_iterations):
        rand_vertex = np.random.randint(v)

        start = default_timer()
        for j in range(v):
            nx.dijkstra_path(graph, rand_vertex, j)
        dij_avg_time += default_timer() - start

        start = default_timer()
        for k in range(v):
            nx.bellman_ford_path(graph, rand_vertex, k)
        bf_avg_time += default_timer() - start

    dij_avg_time /= number_of_iterations
    bf_avg_time /= number_of_iterations

    y_data_dij.append(dij_avg_time)
    y_data_bf.append(bf_avg_time)

# Plotting
plt.plot(x_data, y_data_dij, label='Dijkstra')
plt.plot(x_data, y_data_bf, label='Bellman Ford')
plt.title('Dependence of execution time on number of edges')
plt.xlabel('Number of edges')
plt.ylabel('Execution time, sec')
plt.legend()
plt.tight_layout()
plt.show()

# Creating cell grid
grid = nx.generators.lattice.grid_graph(dim=[range(10), range(10)])

# Making obstacle cells
nodes_to_del = [(node // 10, node % 10) for node in np.random.choice(10 * 10, 30)]
grid.remove_nodes_from(nodes_to_del)

y_data_astar = []  # The set of execution times for A Star algorithm
number_of_experiments = 5

# Counting execution time for A star algorithm number_of_experiments times
for n in range(number_of_experiments):
    rand_index1, rand_index2 = random.sample(range(len(grid.nodes)), 2)
    vertex1, vertex2 = np.array(grid.nodes)[rand_index1], np.array(grid.nodes)[rand_index2]

    start = default_timer()
    path = nx.astar_path(grid, tuple(vertex1), tuple(vertex2))
    y_data_astar.append(default_timer() - start)

# Plotting
plt.scatter(range(1, number_of_experiments + 1), y_data_astar)
plt.title('Execution time of A Star algorithm')
plt.xlabel('Ordinal number of experiment')
plt.ylabel('Execution time, sec')
plt.tight_layout()
plt.show()
