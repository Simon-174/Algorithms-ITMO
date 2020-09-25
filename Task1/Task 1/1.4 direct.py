import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer
from scipy.optimize import curve_fit
from decimal import Decimal


# Implementing the algorithm in question
def dec_pow(q, p):
    q = Decimal(q)
    answer = Decimal(1)
    for i in range(p):
        answer *= q
    return answer


def polynom_direct(v):
    answer = Decimal(0)
    for i in range(len(v)):
        answer += Decimal(v[i]) * dec_pow(1.5, i)
    return answer


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
        polynom_direct(v)
        exec_time += (default_timer() - start)
    time_list.append(exec_time / 5)

# Creating the x-axis for plot making
xdata = [n for n in range(1, 2001)]

# Finding parameters of approximation function
parameters = curve_fit(parabola, xdata, time_list)[0]

# Plotting
plt.plot(xdata, time_list, 'b', label='experimental data')
plt.plot(xdata, parabola(xdata, *parameters), 'r', label='approximation')
plt.title("direct calculation of P(x)")
plt.xlabel('n')
plt.ylabel('Time, sec')
plt.legend()
plt.tight_layout()
plt.show()
