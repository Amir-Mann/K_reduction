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
    min_x = int(min([- a/b for a, b in zip(alphas, betas)]) * 0.7)
    x = np.linspace(min_x, max_x, (max_x - min_x) * 10)
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
    return (tuple[float, float]): corrected (alpha, beta)
    """
    #start = time.time()
    if num_samples == 0:
        return (alpha, beta)
    k_to_sample = round(- alpha / beta)
    r = - alpha / beta - k_to_sample # Round leftovers
    b = sample_func(k_to_sample, num_samples)
    def func_to_minimize(s):
        return (v + 1) / 2 * np.log((1 + s ** 2 / v)) \
            + b * np.log(1 + np.exp(beta * (r - s))) \
            + (num_samples - b) * np.log(1 + np.exp(beta * (s - r)))
    
    result = scipy.optimize.minimize_scalar(func_to_minimize)
    if result.success:
        alpha = alpha + beta * result.x
    else:
        print("\nFailed\n", end=", ")
    return (alpha, beta)

counts_correct_sigmoid_double_sample = {"failed":0, "success":0}
def correct_sigmoid_double_sample(alpha, beta, sample_func, num_samples=10, v=3.36, var_beta=1, **kwargs):
    """"
    Corrects a sigmoid using sampeling, and the assumption that the error is distributing T(v=v)
    alpha (float): any real number
    beta (float): smaller then 0
    sample_func (func(int, int)->(int)): a function which takes an k, num_samples and return the amount of successes,
                                         it samples the real probabilty distribution at that k, num_samples times.
    num_samples (int, optional): num_times to sample real distribution. Defaults to 10.
    return (tuple[float, float]): corrected (alpha, beta)
    """
    #start = time.time()
    if num_samples == 0:
        return (alpha, beta)
    k_to_sample = round(- alpha / beta)
    r = - alpha / beta - k_to_sample # Round situation
    b = sample_func(k_to_sample, num_samples)
    def func_to_minimize(s):
        return (v + 1) / 2 * np.log((1 + s ** 2 / v)) \
            + b * beta * (r - s) \
            + (num_samples) * np.log(1 + np.exp(beta * (s - r)))
    
    result = scipy.optimize.minimize_scalar(func_to_minimize)
    #alpha_currection = - result / beta + r
    #alpha = alpha - beta * alpha_currection
    if result.success:
        prv_alpha = alpha
        alpha = alpha + beta * result.x
    else:
        print(f"\nFailed 1 correction\n", end=", ")
    k1_to_sample = round(- alpha / beta)
    if k1_to_sample == k_to_sample:
        if k1_to_sample < - alpha / beta:
            k1_to_sample = k_to_sample + 1
        else:
            k1_to_sample = k_to_sample - 1
    b1 = sample_func(k1_to_sample, num_samples)
    def func_to_minimize1(vars):
        s, m = vars
        alpha = (prv_alpha / beta + s) * m
        v0 = alpha + m * k_to_sample
        v1 = alpha + m * k1_to_sample
        return (v + 1) / 2 * np.log((1 + s ** 2 / v)) \
            + (beta - m) ** 2  / 2 / var_beta\
            - b * v0 - b1 * v1\
            + (num_samples) * (np.log(1 + np.exp(v0)) + np.log(1 + np.exp(v1)))
    
    result = scipy.optimize.minimize(func_to_minimize1, [result.x, beta])
    if result.success:
        s, m = result.x
        mid_alpha = alpha
        prv_beta = beta
        alpha = (prv_alpha / beta + s) * m
        beta = m
        if "true_alpha" in kwargs:
            true_alpha = kwargs["true_alpha"]
            true_beta = kwargs["true_beta"]
            plot_sigmoids([prv_alpha, mid_alpha, alpha, true_alpha], [prv_beta, prv_beta, beta, true_beta])
            plt.legend(["estimated", "corrected", "double_corrected", "true"], ncol=1, loc='center right', bbox_to_anchor=[1, 1], fontsize=6)
            plt.show()
        counts_correct_sigmoid_double_sample["success"] += 1
    else:
        counts_correct_sigmoid_double_sample["failed"] += 1
        frate = counts_correct_sigmoid_double_sample["failed"] / (counts_correct_sigmoid_double_sample["failed"] + counts_correct_sigmoid_double_sample["success"])
        print(f"\nFailed {frate}\n", end=", ")    #times.append(time.time() - start)
    #print(sum(times) / len(times))
    return (alpha, beta)


counts_correct_sigmoid_double_sample = {"failed":0, "success":0}
def correct_sigmoid_itertive(alpha, beta, sample_func, num_samples=10, v=3.36, var_beta=1, **kwargs):
    """"
    Corrects a sigmoid using sampeling, and the assumption that the error is distributing T(v=v)
    alpha (float): any real number
    beta (float): smaller then 0
    sample_func (func(int, int)->(int)): a function which takes an k, num_samples and return the amount of successes,
                                         it samples the real probabilty distribution at that k, num_samples times.
    num_samples (int, optional): num_times to sample real distribution. Defaults to 10.
    return (tuple[float, float]): corrected (alpha, beta)
    """
    #start = time.time()
    if num_samples == 0:
        return (alpha, beta)
    success_ks, fail_ks = [], []
    k_to_sample = round(- alpha / beta)
    for i in range(num_samples):
        d = round((3 + num_samples) / (3 + i))
        if d == 0:
            d = 1
        print("k", k_to_sample, end=", ")
        if sample_func(k_to_sample, 1) == 1:
            success_ks.append(k_to_sample)
            k_to_sample += d
        else:
            fail_ks.append(k_to_sample)
            k_to_sample -= d
    print(f"{len(success_ks)=}")
    success_ks = np.array(success_ks)
    fail_ks = np.array(fail_ks)
    def func_to_minimize1(vars):
        s, m = vars
        alphat = (alpha / beta + s) * m
        return (v + 1) / 2 * np.log((1 + s ** 2 / v)) \
            + (beta - m) ** 2  / 2 / var_beta\
            + np.sum(np.log(1 + np.exp(- (alphat + m * success_ks)))) \
            + np.sum(np.log(1 + np.exp(+ (alphat + m * fail_ks))))
    
    result = scipy.optimize.minimize(func_to_minimize1, [0, beta])
    if result.success:
        s, m = result.x
        prv_alpha = alpha
        prv_beta = beta
        alpha = (alpha / beta + s) * m
        beta = m
        if "true_alpha" in kwargs and len(success_ks) == num_samples:
            true_alpha = kwargs["true_alpha"]
            true_beta = kwargs["true_beta"]
            plot_sigmoids([prv_alpha, alpha, true_alpha], [prv_beta, beta, true_beta])
            plt.legend(["estimated", "iterative_correction", "true"], ncol=1, loc='center right', bbox_to_anchor=[1, 1], fontsize=6)
            plt.show()
        counts_correct_sigmoid_double_sample["success"] += 1
    else:
        counts_correct_sigmoid_double_sample["failed"] += 1
        frate = counts_correct_sigmoid_double_sample["failed"] / (counts_correct_sigmoid_double_sample["failed"] + counts_correct_sigmoid_double_sample["success"])
        print(f"\nFailed {frate}\n", end=", ")
    return (alpha, beta)