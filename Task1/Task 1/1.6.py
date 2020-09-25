import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer
from scipy.optimize import curve_fit


# Implementing the algorithm in question
def quick_sort(v):
    return np.sort(v)


# Defining approximation function
def n_logn(x, a, b):
    log2x = np.log2(x)
    return np.multiply(a, np.multiply(x, log2x)) + b


# Counting average execution time for each n
time_list = []
for n in range(1, 2001):
    v = np.random.rand(n)
    exec_time = 0
    for i in range(5):
        start = default_timer()
        quick_sort(v)
        exec_time += (default_timer() - start)
    time_list.append(exec_time / 5)

# Creating the x-axis for plot making
xdata = [n for n in range(1, 2001)]

# Finding parameters of approximation function
parameters = curve_fit(n_logn, xdata, time_list)[0]

# Plotting
plt.plot(xdata, time_list, 'b', label='experimental data')
plt.plot(xdata, n_logn(xdata, *parameters), 'r', label='approximation')
plt.title("Quick Sort")
plt.xlabel('n')
plt.ylabel('Time, sec')
plt.legend()
plt.tight_layout()
plt.show()
