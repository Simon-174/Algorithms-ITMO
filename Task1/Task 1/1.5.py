import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer
from scipy.optimize import curve_fit


# Implementing the algorithm in question
def bubble_sort(v):
    sorted_v = v.copy()
    for i in range(len(sorted_v) - 1):
        for j in range(0, len(sorted_v) - i - 1):
            if sorted_v[j] > sorted_v[j + 1]:
                sorted_v[j], sorted_v[j + 1] = sorted_v[j + 1], sorted_v[j]
    return list(sorted_v)


# Defining approximation function
def parabola(x, a, b):
    return np.multiply(a, np.multiply(x, x)) + b


# Counting average execution time for each n
time_list = []
for n in range(1, 2001):
    v = np.random.rand(n)
    exec_time = 0
    for i in range(5):
        start = default_timer()
        bubble_sort(v)
        exec_time += (default_timer() - start)
    time_list.append(exec_time / 5)

# Creating the x-axis for plot making
xdata = [n for n in range(1, 2001)]

# Finding parameters of approximation function
parameters = curve_fit(parabola, xdata, time_list)[0]

# Plotting
plt.plot(xdata, time_list, 'b', label='experimental data')
plt.plot(xdata, parabola(xdata, *parameters), 'r', label='approximation')
plt.title("Bubble Sort")
plt.xlabel('n')
plt.ylabel('Time, sec')
plt.legend()
plt.tight_layout()
plt.show()
