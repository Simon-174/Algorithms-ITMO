import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer
from scipy.optimize import curve_fit


# Implementing the algorithm in question
def mat_product(v):
    size = len(v)
    a = np.random.rand(size, size)
    b = np.random.rand(size, size)
    return np.matmul(a, b)


# Defining approximation function
def cub_parabola(x, a, b):
    x_cubed = np.array([x[i]**3 for i in range(len(x))])
    return a*x_cubed + b


# Counting average execution time for each n
time_list = []
for n in range(1, 2001):
    v = np.random.rand(n)
    exec_time = 0
    for i in range(5):
        start = default_timer()
        mat_product(v)
        exec_time += (default_timer() - start)
    time_list.append(exec_time / 5)

# Creating the x-axis for plot making
xdata = [n for n in range(1, 2001)]

# Finding parameters of approximation function
parameters = curve_fit(cub_parabola, xdata, time_list)[0]

# Plotting
plt.plot(xdata, time_list, 'b', label='experimental data')
plt.plot(xdata, cub_parabola(xdata, *parameters), 'r', label='approximation')
plt.title("the usual matrix product")
plt.xlabel('n')
plt.ylabel('Time, sec')
plt.legend()
plt.tight_layout()
plt.show()
