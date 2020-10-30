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


# Variables for average execution times
dij_avg_time = 0

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
        nx.floyd_warshall(graph)
        dij_avg_time += default_timer() - start



    dij_avg_time /= number_of_iterations

    y_data_dij.append(dij_avg_time)

# Plotting
plt.plot(x_data, y_data_dij, label='Floyd_warshall')
plt.title('Dependence of execution time on number of edges')
plt.xlabel('Number of edges')
plt.ylabel('Execution time, sec')
plt.legend()
plt.tight_layout()
plt.show()


