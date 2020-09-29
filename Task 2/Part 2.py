from math import ceil
from functools import partial
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt


# Implementing optimization methods
def exh_search(f, x1=-5.0, x2=5.0, y1=-5.0, y2=5.0, eps=0.01):
    n1 = ceil((x2 - x1) / eps)
    x_list = [x1 + k * (x2 - x1) / n1 for k in range(n1 + 1)]
    n2 = ceil((y2 - y1) / eps)
    y_list = [y1 + k * (y2 - y1) / n2 for k in range(n2 + 1)]
    r_min = (x_list[0], y_list[0])
    f_min = f((r_min[0], r_min[1]))
    numb_iter = 0
    for i in range(n1 + 1):
        for j in range(n2 + 1):
            if f((x_list[i], y_list[j])) < f_min:
                r_min = (x_list[i], y_list[j])
                f_min = f((r_min[0], r_min[1]))
            numb_iter += 1
    numb_func = numb_iter + 1
    return r_min, numb_iter, numb_func


def gauss(f, x1=-5.0, x2=5.0, y1=-5.0, y2=5.0, eps=0.01):
    n1 = ceil((x2 - x1) / (eps/2))
    x_list = [x1 + k * (x2 - x1) / n1 for k in range(n1 + 1)]
    n2 = ceil((y2 - y1) / (eps/2))
    y_list = [y1 + k * (y2 - y1) / n2 for k in range(n2 + 1)]
    x_min = x_list[len(x_list) // 2]
    y_min = y_list[len(y_list) // 2]
    f_min = f((x_min, y_min))
    r1 = (x_min, y_min)
    numb_iter = 0
    while True:
        for i in range(1, n1 + 1):
            if f((x_list[i], y_min)) < f_min:
                x_min = x_list[i]
                f_min = f((x_min, y_min))
            numb_iter += 1
        for j in range(1, n2 + 1):
            if f((x_min, y_list[j])) < f_min:
                y_min = y_list[j]
                f_min = f((x_min, y_min))
            numb_iter += 1
        r2 = (x_min, y_min)
        if (r1[0] - r2[0])**2 + (r1[1] - r2[1])**2 < eps**2:
            break
        r1 = r2
    numb_func = numb_iter + 1
    return r2, numb_iter, numb_func


# Defining approximation functions
def line(x, a, b):
    return a*x + b


def ratio(x, a, b):
    return a / (1 + b*x)


# Counting the sum of squared deviations for linear and rational approximants
def lsm_line(x_data, y_data, r):
    sum_of_squares = 0
    for i in range(101):
        sum_of_squares += (r[0]*x_data[i] + r[1] - y_data[i])**2
    return sum_of_squares


def lsm_ratio(x_data, y_data, r):
    sum_of_squares = 0
    for i in range(101):
        sum_of_squares += (r[0]/(1 + r[1]*x_data[i]) - y_data[i])**2
    return sum_of_squares


# Creating alpha and beta in the interval(0,1)
while True:
    alpha = np.random.uniform(0, 1)
    if alpha != 0:
        break
while True:
    beta = np.random.uniform(0, 1)
    if beta != 0:
        break

# Making noisy data
x_data = [k/100 for k in range(101)]
y_data = [alpha*x_data[k] + beta + np.random.normal() for k in range(101)]

# Making tuples to make a loop for plotting
titles = ('Linear approximation', 'Rational approximation')
lsm = (lsm_line, lsm_ratio)
f_approx = (line, ratio)
y1 = (-5.0, -0.998)

# Plotting graphs of approximants obtained by considering methods
for i in range(2):
    # Calculating function of deviations
    D = partial(lsm[i], x_data, y_data)
    # Obtaining parameters of approximation functions
    a_exh, b_exh = exh_search(D, y1=y1[i])[0]
    a_gauss, b_gauss = gauss(D, y1=y1[i])[0]
    a_nelder, b_nelder = minimize(D, [0, 0], method='Nelder-Mead').x
    # Calculating y values for plotting
    y_exh = [f_approx[i](x, a_exh, b_exh) for x in x_data]
    y_gauss = [f_approx[i](x, a_gauss, b_gauss) for x in x_data]
    y_nelder = [f_approx[i](x, a_nelder, b_nelder) for x in x_data]
    # Plotting
    plt.plot(x_data, y_data)
    plt.plot(x_data, y_gauss, label='gauss')
    plt.plot(x_data, y_exh, label='exhaustive search')
    plt.plot(x_data, y_nelder, label='Nelder-Mead')
    plt.title(titles[i])
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()
