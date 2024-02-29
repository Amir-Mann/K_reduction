from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy
import time

def gather_statistics(x, y):
    # can be generalized further
    x = np.array(x)
    x = x.reshape(-1, 1)
    regressor = LinearRegression()
    regressor.fit(x, y)
    r_squared = regressor.score(x, y)
    # m = 0
    # b = 0

    return (r_squared,)


# ---------- Helper Functions --------------
def pretty_print(matrix):
    s = [[str(e) for e in row] for row in matrix]
    lens = [max(map(len, col)) for col in zip(*s)]
    fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
    table = [fmt.format(*row) for row in s]
    print('\n'.join(table))


def max_index_of_matrix(matrix):
    y_max_index = None
    y_max_value = None
    for i, row in enumerate(matrix):
        x_max_index = np.argmax(row)
        x_max_value = row[x_max_index]
        if y_max_value == None or y_max_value < x_max_value:
            y_max_value = x_max_value
            y_max_index = (i, x_max_index)
    return y_max_index


def get_y(x1, x2, y_func):
    assert len(x1) == len(x2)
    y = []
    for i in range(len(x1)):
        y.append(y_func(x1[i], x2[i]))
    return y


def sigmoid_array(x):
    return 1 / (1 + np.exp(-x))


def plot_sigmoids(alphas, betas):
    max_x = int(max([- a/b for a, b in zip(alphas, betas)]) * 1.3)
    x = np.linspace(0.6 * max_x, max_x, max_x * 10)
    for a, b in zip(alphas, betas):
        y = sigmoid_array(a + b * x)
        plt.plot(x, y)

times = []
def correct_sigmoid(alpha, beta, sample_func, num_samples=10, v=3.36, **kwargs):
    """"
    Corrects a sigmoid using sampeling, and the assumption that the error is distributing T(v=v)
    alpha (float): any real number
    beta (float): smaller then 0
    sample_func (func(int, int)->(int)): a function which takes an k, num_samples and return the amount of successes,
                                         it samples the real probabilty distribution at that k, num_samples times.
    num_samples (int, optional): num_times to sample real distribution. Defaults to 10.
    return (tuple[float, float]): currected (alpha, beta)
    """
    #start = time.time()
    if num_samples == 0:
        return (alpha, beta)
    k_to_sample = round(- alpha / beta)
    r = - alpha / beta - k_to_sample # Round situation
    b = sample_func(k_to_sample, num_samples)
    def func_to_get_root_of(l):
        return (v + 1) / (v / (- l / beta + r) - l / beta + r) + beta * num_samples / (1 + np.exp(l)) - beta * b
    def func_to_minimize(s):
        return (v + 1) / 2 * np.log((1 + s ** 2 / v)) \
            + b * np.log(1 + np.exp(beta * (r - s))) \
            + (num_samples - b) * np.log(1 + np.exp(beta * (s - r)))
    
    result = scipy.optimize.minimize_scalar(func_to_minimize)
    #alpha_currection = - result / beta + r
    #alpha = alpha - beta * alpha_currection
    if result.success:
        prv_alpha = alpha
        alpha = alpha + beta * result.x
    else:
        print("\nFailed\n", end=", ")
    if "true_alpha" in kwargs and num_samples == 1001:
        true_alpha = kwargs["true_alpha"]
        true_beta = kwargs["true_beta"]
        x = np.linspace(-10, 10, 2000)
        y = [func_to_minimize(v) for v in x]
        plot_sigmoids([prv_alpha, alpha, true_alpha], [beta, beta, true_beta])
        plt.legend(["estimated", "currected", "true"], ncol=1, loc='center right', bbox_to_anchor=[1, 1], fontsize=6)
        plt.show()
        plt.plot(x, y)
        plt.scatter(result.x, func_to_minimize(result.x))
        plt.show()
    #times.append(time.time() - start)
    #print(sum(times) / len(times))
    return (alpha, beta)