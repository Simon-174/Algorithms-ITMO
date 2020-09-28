from math import sin, ceil, sqrt
import matplotlib.pyplot as plt


# Defining considering functions
def f1(x):
    return x**3


def f2(x):
    return abs(x - 0.2)


def f3(x):
    return x * sin(1/x)


# Implementing optimization methods
def exh_search(f, a, b, eps=0.0001):
    n = ceil((b - a) / eps)
    x_list = [a + k * (b - a) / n for k in range(n + 1)]
    x_min = x_list[0]
    min_f = f(x_min)
    for i in range(1, n + 1):
        if f(x_list[i]) < min_f:
            x_min = x_list[i]
            min_f = f(x_min)
    numb_iter = n + 1
    numb_func = n + 1
    return x_min, numb_iter, numb_func


def dichotomy(f, a, b, eps=0.0001):
    delta = eps / 2
    numb_iter = 0
    while abs(a - b) > eps:
        x1 = (a + b - delta) / 2
        x2 = (a + b + delta) / 2
        fx1 = f(x1)
        fx2 = f(x2)
        if fx1 < fx2:
            b = x2
        else:
            a = x1
        numb_iter += 1
    numb_func = 2 * numb_iter
    x_min = (a + b) / 2
    return x_min, numb_iter, numb_func


def golden_section(f, a, b, eps=0.0001):
    x1 = a + (sqrt(5) - 1)/2 * (b - a)
    x2 = b - (sqrt(5) - 1)/2 * (b - a)
    fx1 = f(x1)
    fx2 = f(x2)
    numb_iter = 0
    while abs(a - b) > eps:
        if fx1 < fx2:
            a = x2
            x2 = x1
            fx2 = fx1
            x1 = a + (sqrt(5) - 1)/2 * (b - a)
            fx1 = f(x1)
        else:
            b = x1
            x1 = x2
            fx1 = fx2
            x2 = b - (sqrt(5) - 1)/2 * (b - a)
            fx2 = f(x2)
        numb_iter += 1
    numb_func = numb_iter + 2
    x_min = (a + b) / 2
    return x_min, numb_iter, numb_func


# Creating tuples to make a loop for plotting
f = (f1, f2, f3)
a = (0, 0, 0.01)
titles = ('Cubic parabola', 'f(x) = |x - 0.2|', 'f(x) = x * sin(1 / x)')

# Plotting graphs for each function
for i in range(3):
    # Set x-axis for plotting: for f1 and f2 0 <= x <= 1, for f3 0.01 <= x <= 1
    if i != 2:
        xdata = [n / 100 for n in range(-10, 101)]
    else:
        xdata = [n / 100 for n in range(1, 101)]
    # Creating tuples to output x_min, number of function calls and number of iterations for each method
    x_opt = (exh_search(f[i], a[i], 1)[0], dichotomy(f[i], a[i], 1)[0], golden_section(f[i], a[i], 1)[0])
    numb_iter = (str(exh_search(f[i], a[i], 1)[1]), str(dichotomy(f[i], a[i], 1)[1]), str(golden_section(f[i], a[i], 1)[1]))
    numb_func = (str(exh_search(f[i], a[i], 1)[2]), str(dichotomy(f[i], a[i], 1)[2]), str(golden_section(f[i], a[i], 1)[2]))
    # Set y-axis for plotting
    ydata = []
    for x in xdata:
        ydata.append(f[i](x))
    # Plotting
    plt.plot(xdata, ydata, 'b', label='graph of function')
    plt.scatter(x_opt[0], f[i](x_opt[0]), c='y', label='exhaustive search, numb_iter = ' + numb_iter[0] + ', numb_func = ' + numb_func[0])
    plt.scatter(x_opt[1], f[i](x_opt[1]), c='g', label='dichotomy, numb_iter = ' + numb_iter[1] + ', numb_func = ' + numb_func[1])
    plt.scatter(x_opt[2], f[i](x_opt[2]), c='r', label='golden section, numb_iter = ' + numb_iter[2] + ', numb_func = ' + numb_func[2])
    plt.title(titles[i])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()
