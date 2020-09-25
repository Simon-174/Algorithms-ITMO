import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer
from scipy.optimize import curve_fit
from decimal import Decimal


# Implementing the algorithm in question
def polynom_Horner(v):
    i = len(v) - 1
    answer = v[i]
    while i > 0:
        answer = v[i-1] + answer*1.5
        i -= 1
    return answer


# Defining approximation function
def line(x, a, b):
    return np.multiply(a, x) + b


# Counting average execution time for each n
time_list = []
for n in range(1, 2001):
    v = np.random.rand(n)
    exec_time = 0
    for i in range(5):
        start = default_timer()
        polynom_Horner(v)
        exec_time += (default_timer() - start)
    time_list.append(exec_time / 5)

# Creating the x-axis for plot making
xdata = [n for n in range(1, 2001)]

# Finding parameters of approximation function
parameters = curve_fit(line, xdata, time_list)[0]

# Plotting
plt.plot(xdata, time_list, 'b', label='experimental data')
plt.plot(xdata, line(xdata, *parameters), 'r', label='approximation')
plt.title("Horner’s method of calculation P(x)")
plt.xlabel('n')
plt.ylabel('Time, sec')
plt.legend()
plt.tight_layout()
plt.show()
