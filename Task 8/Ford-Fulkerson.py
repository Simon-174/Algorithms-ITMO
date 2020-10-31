import networkx as nx
import numpy as np
import random
from itertools import permutations
from timeit import default_timer
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt


# Defining approximation function
def line(x, a, b):
    return np.multiply(a, x) + b


def rand_G(n, m):
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for pair in random.sample([*permutations(range(n), 2)], m):
        G.add_edge(*pair, weight=random.randint(10, 100))
    return G


def fs(s, t, parent):
    visited_list = [False] * v
    q = []
    q.append(s)
    visited_list[s] = True
    while q:
        u = q.pop(0)
        for index, val in enumerate(graphmatrix[u]):
            if not visited_list[index] and val > 0:
                q.append(index)
                visited_list[index] = True
                parent[index] = u
    return visited_list[t]


def fl(graphmatrix, source, sink):
    parent = [-1] * v
    max_flow = 0
    while fs(source, sink, parent):
        path_flow = float("Inf")
        s = sink
        while s != source:
            path_flow = min(path_flow, graphmatrix[parent[s]][s])
            s = parent[s]
            max_flow += path_flow
            n = sink
        while n != source:
            p = parent[n]
            graphmatrix[p][n] -= path_flow
            graphmatrix[n][p] += path_flow
            n = parent[n]
    return max_flow


v = 40
m_max = 700
number_of_iterations = 20
x_data = [i for i in range(1, m_max + 1)]
times = []

for m in range(1, m_max + 1):
    graph = rand_G(v, m)
    matrix = nx.adjacency_matrix(graph).todense()
    graphmatrix = matrix.tolist()
    source = 0
    sink = 5
    t1 = default_timer()
    for i in range(number_of_iterations):
        fl(graphmatrix, source, sink)
    t2 = default_timer()
    times.append((t2 - t1) / number_of_iterations)

times = np.array(times)
times = np.multiply(times, 1000)

# Finding parameters of approximation function
parameters = curve_fit(line, x_data, times)[0]

plt.plot(x_data, times, label='experimental data')
plt.plot(x_data, line(x_data, *parameters), 'r', label='approximation')
plt.title("Ford-Fulkerson algorithm")
plt.legend()
plt.xlabel('Number of edges')
plt.ylabel('Execution time, ms')
plt.show()
