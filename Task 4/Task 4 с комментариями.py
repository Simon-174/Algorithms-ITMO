import numpy as np
from numpy.random import normal
from functools import partial
from scipy.optimize import minimize, least_squares, dual_annealing, differential_evolution
import matplotlib.pyplot as plt
from timeit import default_timer


# Defining the functions specified in the task condition
def f(x):
    return 1 / (x ** 2 - 3 * x + 2)


def y_k(x_k):
    if f(x_k) < -100:
        return -100 + normal()
    elif f(x_k) <= 100:
        return f(x_k) + normal()
    else:
        return 100 + normal()


# Defining approximation function
def rational_func(p, x):
    return (p[0] * x + p[1]) / (x ** 2 + p[2] * x + p[3])


# Calculating arrays of residuals
def residuals_list(x_k_list, y_k_list, p, func=rational_func):
    return [func(p, x_k_list[i]) - y_k_list[i] for i in range(len(x_k_list))]


# Obtaining the sum of deviations squared
def sum_of_squares(x_k_list, y_k_list, p):
    res_list = residuals_list(x_k_list, y_k_list, p)
    return sum([res_list[i] ** 2 for i in range(len(x_k_list))])


# Making noisy data
x_k_list = [3 * k / 1000 for k in range(1001)]
y_k_list = [y_k(x_k) for x_k in x_k_list]

# Defining function of deviations squared
D = partial(sum_of_squares, x_k_list, y_k_list)

# Plotting noisy data
plt.scatter(x_k_list, y_k_list, s=0.5, label='Noisy data')

# List of execution times for considering optimization methods
methods_times = []


# Usage of Nelder-Mead method and plotting approximation function obtained
start = default_timer()
nelder_opt = minimize(D, np.array([1, 1, 1, 1]), method='Nelder-Mead')
methods_times.append(default_timer() - start)
p_nelder = nelder_opt.x
plt.plot(x_k_list, [rational_func(p_nelder, x_k_list[i]) for i in range(len(x_k_list))], label='Nelder-Mead')

# Usage of Levenberg-Marquardt method and plotting approximation function obtained
start = default_timer()
lm_opt = least_squares(partial(residuals_list, x_k_list, y_k_list), [1, 1, 1, 1], method='lm')
methods_times.append(default_timer() - start)
p_lm = lm_opt.x
plt.plot(x_k_list, [rational_func(p_lm, x_k_list[i]) for i in range(len(x_k_list))], label='Levenberg-Marquardt')

# Usage of Simulated Annealing method and plotting approximation function obtained
start = default_timer()
anneal_opt = dual_annealing(D, [(-5, 5)] * 4)
methods_times.append(default_timer() - start)
p_anneal = anneal_opt.x
plt.plot(x_k_list, [rational_func(p_anneal, x_k_list[i]) for i in range(len(x_k_list))], label='Simulated Annealing')

# Usage of Differential Evolution method and plotting approximation function obtained
start = default_timer()
dif_evo_opt = differential_evolution(D, [(-5, 5)] * 4)
methods_times.append(default_timer() - start)
p_dif_evo = dif_evo_opt.x
plt.plot(x_k_list, [rational_func(p_dif_evo, x_k_list[i]) for i in range(len(x_k_list))], label='Differential Evolution')

plt.title('Rational approximation')
plt.legend()
plt.tight_layout()
plt.show()

# Diagram of function calls depending on methods
plt.bar(['Nelder\nMead', 'Levenberg\nMarquardt', 'Simulated\nAnnealing', 'Differential\nEvolution'],
        [nelder_opt.nfev, lm_opt.nfev, anneal_opt.nfev, dif_evo_opt.nfev], width=0.4)
plt.title('Number of function evaluations')
plt.show()

# Diagram of execution times depending on methods
plt.bar(['Nelder\nMead', 'Levenberg\nMarquardt', 'Simulated\nAnnealing', 'Differential\nEvolution'], methods_times, width=0.4)
plt.title('Execution time')
plt.show()
