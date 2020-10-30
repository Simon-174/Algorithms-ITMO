import networkx as nx
import numpy as np
import random
from itertools import permutations
import timeit
from timeit import default_timer
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt

# Defining approximation function
def line(x, a, b):
    return np.multiply(a, x) + b

def random_graph(n, m):
     graph = nx.Graph()
     N_range = range(n)
     graph.add_nodes_from(N_range)
     for pair in random.sample([*permutations(N_range, 2)], m):
        graph.add_edge(*pair, weight=random.randint(10, 100))
     return graph
n = 40
m_max = 700
number_of_iterations = 20
x_data = [i for i in range(1, m_max + 1)]
times = []
for m in range(1, m_max + 1):
    graph1 = random_graph(n, m)
    matrix = nx.adjacency_matrix(graph1).todense()
    graphmatrix=matrix.tolist()
    def fs(s, t, parent):
         visited = [False] * n
         queue = []
         queue.append(s)
         visited[s] = True
         while queue:
             u = queue.pop(0)
             for ind, val in enumerate(graphmatrix[u]):
                 if visited[ind] == False and val > 0:
                     queue.append(ind)
                     visited[ind] = True
                     parent[ind] = u
         return True if visited[t] else False
    def fl(graphmatrix, source, sink):
         parent = [-1] * n
         max_flow = 0
         while fs(source, sink, parent):
             path_flow = float("Inf")
             s = sink
             while (s != source):
                 path_flow = min(path_flow, graphmatrix[parent[s]][s])
                 s = parent[s]
                 max_flow += path_flow
                 v = sink
             while (v != source):
                 u = parent[v]
                 graphmatrix[u][v] -= path_flow
                 graphmatrix[v][u] += path_flow
                 v = parent[v]
         return max_flow
    source = 0
    sink = 5
    t1=default_timer()
    for i in range(number_of_iterations):
        fl(graphmatrix, source, sink)
    t2=default_timer()
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
