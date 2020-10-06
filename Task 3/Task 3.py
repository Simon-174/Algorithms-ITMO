from functools import partial
import numpy as np
from scipy.optimize import minimize, least_squares
import matplotlib.pyplot as plt


# Implementing optimization methods
def grad_desc(f_dev, r_init=(0, 0), eps=0.0001):
    r = list(r_init)
    alpha = 10*eps
    numb_iter = 0
    while True:
        r_prev = r[:]
        r[0] = r_prev[0] - alpha * f_dev[0]((r_prev[0], r_prev[1]))
        r[1] = r_prev[1] - alpha * f_dev[1]((r_prev[0], r_prev[1]))
        numb_iter += 1
        if (r[0] - r_prev[0]) ** 2 + (r[1] - r_prev[1]) ** 2 < eps ** 2:
            break
    numb_f_calls = numb_iter * 2
    r_min = [(r_prev[0] + r[0]) / 2, (r_prev[1] + r[1]) / 2]
    return r_min, numb_iter, numb_f_calls


def newton_method(f_dev, f_dev2, r_start=(0, 0), eps=0.0001):
    df_dx = f_dev[0]
    df_dy = f_dev[1]
    d2f_dx2 = f_dev2[0]
    d2f_dxdy = f_dev2[1]
    d2f_dy2 = f_dev2[2]
    r = list(r_start)
    alpha = 0.01
    numb_iter = 0
    while True:
        r_prev = r[:]
        delta_x = (df_dy(r) * d2f_dxdy(r) - df_dx(r) * d2f_dy2(r)) / (d2f_dx2(r) * d2f_dy2(r) - d2f_dxdy(r) ** 2)
        delta_y = (df_dx(r) * d2f_dxdy(r) - df_dy(r) * d2f_dx2(r)) / (d2f_dx2(r) * d2f_dy2(r) - d2f_dxdy(r) ** 2)
        r[0] = r_prev[0] + alpha * delta_x
        r[1] = r_prev[1] + alpha * delta_y
        numb_iter += 1
        if (r[0] - r_prev[0]) ** 2 + (r[1] - r_prev[1]) ** 2 < eps ** 2:
            break
        numb_f_calls = numb_iter * 14
    r_min = ((r_prev[0] + r[0]) / 2, (r_prev[1] + r[1]) / 2)
    return r_min, numb_iter, numb_f_calls


# Defining approximation functions
def line(x, a, b):
    return a * x + b


def ratio(x, a, b):
    return a / (1 + b * x)


# Calculating arrays of residuals for Levenberg-Marquardt algorithm
def residuals_line(x_data, y_data, r):
    return np.array(np.multiply(r[0], x_data) + r[1] - y_data)


def residuals_ratio(x_data, y_data, r):
    return np.array(np.multiply(r[0], 1 / (1 + np.multiply(r[1], x_data))) - y_data)


# Obtaining the sum of deviations squared for line and rational approximants
def sum_of_squares_line(x_data, y_data, r):
    sum_of_squares = 0
    for i in range(101):
        sum_of_squares += (r[0] * x_data[i] + r[1] - y_data[i]) ** 2
    return sum_of_squares


def sum_of_squares_ratio(x_data, y_data, r):
    sum_of_squares = 0
    for i in range(101):
        sum_of_squares += (r[0] / (1 + r[1] * x_data[i]) - y_data[i]) ** 2
    return sum_of_squares


# Calculating partial derivatives of the sum of deviations squared for line and rational approximants
def sum_of_squares_line_dev_a(x_data, y_data, r):
    d_da_sum_of_squares = 0
    for i in range(101):
        d_da_sum_of_squares += r[0] * x_data[i] ** 2 + r[1] * x_data[i] - x_data[i] * y_data[i]
    d_da_sum_of_squares *= 2
    return d_da_sum_of_squares


def sum_of_squares_line_dev_b(x_data, y_data, r):
    d_db_sum_of_squares = 0
    for i in range(101):
        d_db_sum_of_squares += r[1] + r[0] * x_data[i] - y_data[i]
    d_db_sum_of_squares *= 2
    return d_db_sum_of_squares


def sum_of_squares_ratio_dev_a(x_data, y_data, r):
    d_da_sum_of_squares = 0
    for i in range(101):
        d_da_sum_of_squares += r[0] / (1 + r[1] * x_data[i]) ** 2 - y_data[i] / (1 + r[1] * x_data[i])
    d_da_sum_of_squares *= 2
    return d_da_sum_of_squares


def sum_of_squares_ratio_dev_b(x_data, y_data, r):
    d_db_sum_of_squares = 0
    for i in range(101):
        d_db_sum_of_squares += r[0] * x_data[i] * y_data[i] / (1 + r[1] * x_data[i]) ** 2 - r[0] ** 2 * x_data[i] / (
                1 + r[1] * x_data[i]) ** 3
    d_db_sum_of_squares *= 2
    return d_db_sum_of_squares


# Calculating second partial derivatives of the sum of deviations squared for line and rational approximants
def sum_of_squares_line_dev2_a2(x_data, y_data, r):
    d2_da2_sum_of_squares_a = 0
    for i in range(101):
        d2_da2_sum_of_squares_a += x_data[i] ** 2
    d2_da2_sum_of_squares_a *= 2
    return d2_da2_sum_of_squares_a


def sum_of_squares_line_dev2_ab(x_data, y_data, r):
    d2_dadb_sum_of_squares_ab = 0
    for i in range(101):
        d2_dadb_sum_of_squares_ab += x_data[i]
    d2_dadb_sum_of_squares_ab *= 2
    return d2_dadb_sum_of_squares_ab


def sum_of_squares_line_dev2_b2(x_data, y_data, r):
    return 202


def sum_of_squares_ratio_dev2_a2(x_data, y_data, r):
    d2_da2_sum_of_squares_a = 0
    for i in range(101):
        d2_da2_sum_of_squares_a += 1 / (1 + r[1] * x_data[i]) ** 2
    d2_da2_sum_of_squares_a *= 2
    return d2_da2_sum_of_squares_a


def sum_of_squares_ratio_dev2_ab(x_data, y_data, r):
    d2_dadb_sum_of_squares_ab = 0
    for i in range(101):
        d2_dadb_sum_of_squares_ab += x_data[i] * y_data[i] / (1 + r[1] * x_data[i]) ** 2 - 2 * r[0] * x_data[i] / (
                1 + r[1] * x_data[i]) ** 3
    d2_dadb_sum_of_squares_ab *= 2
    return d2_dadb_sum_of_squares_ab


def sum_of_squares_ratio_dev2_b2(x_data, y_data, r):
    sum_of_squares_b = 0
    for i in range(101):
        sum_of_squares_b += 3 * r[0] ** 2 * x_data[i] ** 2 / (1 + r[1] * x_data[i]) ** 4 - 2 * r[0] * x_data[i] ** 2 * \
                            y_data[i] / (1 + r[1] * x_data[i]) ** 3
    sum_of_squares_b *= 2
    return sum_of_squares_b


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
x_data = [k / 100 for k in range(101)]
y_data = [alpha * x_data[k] + beta + np.random.normal() for k in range(101)]

# Calculation function of deviations squared and its first and second derivatives
D = (partial(sum_of_squares_line, x_data, y_data), partial(sum_of_squares_ratio, x_data, y_data))

D_dev = ((partial(sum_of_squares_line_dev_a, x_data, y_data), partial(sum_of_squares_line_dev_b, x_data, y_data)),
         (partial(sum_of_squares_ratio_dev_a, x_data, y_data), partial(sum_of_squares_ratio_dev_b, x_data, y_data)))

D_dev2 = (
    (partial(sum_of_squares_line_dev2_a2, x_data, y_data),
     partial(sum_of_squares_line_dev2_ab, x_data, y_data),
     partial(sum_of_squares_line_dev2_b2, x_data, y_data)
     ),
    (partial(sum_of_squares_ratio_dev2_a2, x_data, y_data),
     partial(sum_of_squares_ratio_dev2_ab, x_data, y_data),
     partial(sum_of_squares_ratio_dev2_b2, x_data, y_data)
     )
)

# Calculating arrays of residuals for Levenberg-Marquardt algorithm
residuals_lists = (partial(residuals_line, x_data, y_data), partial(residuals_ratio, x_data, y_data))

# Making tuples to make a loop for plotting
approximation_functions = (line, ratio)
titles = ('Linear approximation', 'Rational approximation')

# Plotting graphs of approximants obtained by considering methods
for i in range(2):
    plt.plot(x_data, y_data)

    a, b = grad_desc(D_dev[i])[0]
    plt.plot(x_data, [approximation_functions[i](x, a, b) for x in x_data], label='grad')

    a, b = minimize(D[i], np.array([0, 0]), method='CG').x
    plt.plot(x_data, [approximation_functions[i](x, a, b) for x in x_data], label='conj_grad')

    a, b = least_squares(residuals_lists[i], [0, 0], method='lm').x
    plt.plot(x_data, [approximation_functions[i](x, a, b) for x in x_data], label='levenberg-marquardt')

    a, b = newton_method(D_dev[i], D_dev2[i])[0]
    plt.plot(x_data, [approximation_functions[i](x, a, b) for x in x_data], label='newton')

    plt.title(titles[i])
    plt.legend()
    plt.tight_layout()
    plt.show()
