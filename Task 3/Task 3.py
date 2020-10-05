from functools import partial
import numpy as np
from scipy.optimize import minimize, least_squares
import matplotlib.pyplot as plt


# Implementing optimization methods
def grad_desc(f_dev, r_start=(0, 0), eps=0.0001):
    r_now = list(r_start)
    alpha = eps
    numb_iter = 0
    while True:
        r_prev = r_now[:]
        r_now[0] = r_prev[0] - alpha * f_dev[0]((r_prev[0], r_prev[1]))
        r_now[1] = r_prev[1] - alpha * f_dev[1]((r_prev[0], r_prev[1]))
        numb_iter += 1
        if (r_now[0] - r_prev[0]) ** 2 + (r_now[1] - r_prev[1]) ** 2 < eps ** 2:
            break
    numb_f_dev = numb_iter * 2
    r_min = [(r_prev[0] + r_now[0]) / 2, (r_prev[1] + r_now[1]) / 2]
    return r_min, numb_iter, numb_f_dev


def newton_method(f_dev, f_dev2, r_start=(0, 0), eps=0.0001):
    df_dx = f_dev[0]
    df_dy = f_dev[1]
    d2f_dx2 = f_dev2[0]
    d2f_dxdy = f_dev2[1]
    d2f_dy2 = f_dev2[2]

    r = list(r_start)
    alpha = 0.5
    while True:
        r_prev = r
        delta_x = (df_dy(r) * d2f_dxdy(r) - df_dx(r) * d2f_dy2(r)) / (d2f_dx2(r) * d2f_dy2(r) - d2f_dxdy(r)**2)
        delta_y = (df_dx(r) * d2f_dxdy(r) - df_dy(r) * d2f_dx2(r)) / (d2f_dx2(r) * d2f_dy2(r) - d2f_dxdy(r)**2)
        r[0] = r_prev[0] + alpha*delta_x
        r[1] = r_prev[1] + alpha*delta_y
        if (r[0] - r_prev[0]) ** 2 + (r[1] - r_prev[1]) ** 2 < eps ** 2:
            break
    r_min = ((r_prev[0] + r[0]) / 2, (r_prev[1] + r[1]) / 2)
    return r_min
    # Defining approximation functions


def line(x, a, b):
    return a * x + b


def ratio(x, a, b):
    return a / (1 + b * x)


def residuals_line(x_data, y_data, r_ar):
    return np.array(np.multiply(r_ar[0], x_data) + r_ar[1] - y_data)


def residuals_ratio(x_data, y_data, r_ar):
    return np.array(np.multiply(r_ar[0], 1 / (1 + np.multiply(r_ar[1], x_data))) - y_data)


def lsm_line(x_data, y_data, r):
    sum_of_squares = 0
    for i in range(101):
        sum_of_squares += (r[0] * x_data[i] + r[1] - y_data[i]) ** 2
    return sum_of_squares


def lsm_ratio(x_data, y_data, r):
    sum_of_squares = 0
    for i in range(101):
        sum_of_squares += (r[0] / (1 + r[1] * x_data[i]) - y_data[i]) ** 2
    return sum_of_squares


# Counting the sum of squared deviations for linear and rational approximants
def lsm_line_dev_a(x_data, y_data, r):
    sum_of_squares_a = 0
    for i in range(101):
        sum_of_squares_a += r[0] * x_data[i] ** 2 + r[1] * x_data[i] - x_data[i] * y_data[i]
    sum_of_squares_a *= 2
    return sum_of_squares_a


def lsm_line_dev_b(x_data, y_data, r):
    sum_of_squares_b = 0
    for i in range(101):
        sum_of_squares_b += r[1] + r[0] * x_data[i] - y_data[i]
    sum_of_squares_b *= 2
    return sum_of_squares_b


def lsm_ratio_dev_a(x_data, y_data, r):
    sum_of_squares_a = 0
    for i in range(101):
        sum_of_squares_a += r[0] / (1 + r[1] * x_data[i]) ** 2 - y_data[i] / (1 + r[1] * x_data[i])
    sum_of_squares_a *= 2
    return sum_of_squares_a


def lsm_ratio_dev_b(x_data, y_data, r):
    sum_of_squares_b = 0
    for i in range(101):
        sum_of_squares_b += r[0] * x_data[i] * y_data[i] / (1 + r[1] * x_data[i]) ** 2 - r[0] ** 2 * x_data[i] / (1 + r[1] * x_data[i]) ** 3
    sum_of_squares_b *= 2
    return sum_of_squares_b


def lsm_line_dev2_a2(x_data, y_data, r):
    sum_of_squares_a = 0
    for i in range(101):
        sum_of_squares_a += x_data[i]**2
    sum_of_squares_a *= 2
    return sum_of_squares_a

def lsm_line_dev2_ab(x_data, y_data, r):
    sum_of_squares_ab = 0
    for i in range(101):
        sum_of_squares_ab += x_data[i]
    sum_of_squares_ab *= 2
    return sum_of_squares_ab

def lsm_line_dev2_b2(x_data, y_data, r):
    return 202


def lsm_ratio_dev2_a2 (x_data, y_data, r):
    sum_of_squares_a = 0
    for i in range(101):
        sum_of_squares_a += 1 / (1 + r[1]*x_data[i])**2
    sum_of_squares_a *= 2
    return sum_of_squares_a


def lsm_ratio_dev2_ab (x_data, y_data, r):
    sum_of_squares_ab = 0
    for i in range(101):
        sum_of_squares_ab += x_data[i]*y_data[i] / (1 + r[1]*x_data[i])**2 - 2*r[0]*x_data[i]/(1 + r[1]*x_data[i])**3
    sum_of_squares_ab *= 2
    return sum_of_squares_ab

def lsm_ratio_dev2_b2 (x_data, y_data, r):
    sum_of_squares_b = 0
    for i in range(101):
        sum_of_squares_b += 3*r[0]**2*x_data[i]**2 / (1 + r[1]*x_data[i])**4 - 2*r[0]*x_data[i]**2*y_data[i]/(1 + r[1]*x_data[i])**3
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

# D_line = partial(lsm_line, x_data, y_data)
# D_ratio = partial(lsm_ratio, x_data, y_data)

# D_line_dev = (partial(lsm_line_dev_a, x_data, y_data), partial(lsm_line_dev_b, x_data, y_data))
# D_ratio_dev = (partial(lsm_ratio_dev_a, x_data, y_data), partial(lsm_ratio_dev_b, x_data, y_data))

# a, b = grad_desc(D_line_dev)[0]
# y_approx_line = [line(x, a, b) for x in x_data]


# a, b = grad_desc(D_ratio_dev)[0]
# y_approx_ratio = [ratio(x, a, b) for x in x_data]


# a, b = minimize(D_line, np.array([0, 0]), method='CG').x
# y_approx_line = [line(x, a, b) for x in x_data]

# a, b = minimize(D_ratio, np.array([0, 0]), method='CG').x
# y_approx_ratio = [ratio(x, a, b) for x in x_data]


#D_line_lst = partial(residuals_line, x_data, y_data)
#a, b = least_squares(D_line_lst, [0, 0], method='lm').x
#y_approx_line = [line(x, a, b) for x in x_data]

#D_ratio_lst = partial(residuals_ratio, x_data, y_data)
#a, b = least_squares(D_ratio_lst, [0, 0], method='lm').x
#y_approx_ratio = [ratio(x, a, b) for x in x_data]


D_line_dev = (partial(lsm_line_dev_a, x_data, y_data), partial(lsm_line_dev_b, x_data, y_data))
D_line_dev2 = (partial(lsm_line_dev2_a2, x_data, y_data), partial(lsm_line_dev2_ab, x_data, y_data), partial(lsm_line_dev2_b2, x_data, y_data))
a, b = newton_method(D_line_dev, D_line_dev2)
y_approx_line = [line(x, a, b) for x in x_data]

D_ratio_dev = (partial(lsm_ratio_dev_a, x_data, y_data), partial(lsm_ratio_dev_b, x_data, y_data))
D_ratio_dev2 = (partial(lsm_ratio_dev2_a2, x_data, y_data), partial(lsm_ratio_dev2_ab, x_data, y_data), partial(lsm_ratio_dev2_b2, x_data, y_data))
a, b = newton_method(D_ratio_dev, D_ratio_dev2)
y_approx_ratio = [ratio(x, a, b) for x in x_data]

plt.plot(x_data, y_data)
plt.plot(x_data, y_approx_line)
plt.plot(x_data, y_approx_ratio)
plt.tight_layout()
plt.show()
