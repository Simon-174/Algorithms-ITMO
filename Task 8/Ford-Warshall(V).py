from itertools import combinations
import networkx as nx
import numpy as np
from timeit import default_timer
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Defining approximation function
def cub_parabola(x, a, b):
    x_cubed = np.array([x[i]**3 for i in range(len(x))])
    return a*x_cubed + b

v = 40  # Number of vertices
max_v = 100

# Lists for plotting
x_data = [i for i in range(v, max_v+1)]  # The number of edges
y_data_dij = []  # The set of execution times for Dijkstra algorithm

while v != max_v + 1:
    edges = set(combinations(range(v), 2))  # The set of edges that have not been added yet
    max_weight = 100  # Maximum weight of edges
    number_of_iterations = 3  # Variable for counting average execution time

    # Initializing a graph
    graph = nx.Graph()
    graph.add_nodes_from(range(v))


    # Variables for average execution times
    dij_avg_time = 0

    # Adding edges to make a graph connected
    for j in range(v-1):
        edge = (j, np.random.randint(j+1, v))
        graph.add_edge(*edge, weight=np.random.randint(1, max_weight + 1))
        edges.remove(edge)

    # Till the number of edges is maximum for given number of vertices:
    # Add one random edge, count execution times
    k = 0
    while k != 500:
        r = np.random.randint(len(edges))
        edge = list(edges)[r]

        graph.add_edge(*edge, weight=np.random.randint(1, max_weight + 1))
        edges.remove(edge)
        k += 1

    for i in range(number_of_iterations):
        start = default_timer()
        nx.floyd_warshall(graph)
        dij_avg_time += default_timer() - start

    dij_avg_time /= number_of_iterations

    y_data_dij.append(dij_avg_time)
    v += 1
    print(v)

# Finding parameters of approximation function
parameters = curve_fit(cub_parabola, x_data, y_data_dij)[0]

# Plotting
plt.plot(x_data, y_data_dij, label='Floyd_warshall')
plt.plot(x_data, cub_parabola(x_data, *parameters), 'r', label='approximation')
plt.title('Dependence of execution time on number of vertices')
plt.xlabel('Number of edges')
plt.ylabel('Execution time, sec')
plt.legend()
plt.tight_layout()
plt.show()


