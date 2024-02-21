from sklearn.linear_model import LinearRegression
import numpy as np


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


def get_a_b(x1, x2, func_for_a, func_for_b):
    assert len(x1) == len(x2)
    a = []
    b = []
    for i in range(len(x1)):
        a.append(func_for_a(x1[i], x2[i]))
        b.append(func_for_b(x1[i], x2[i]))
    return a, b
