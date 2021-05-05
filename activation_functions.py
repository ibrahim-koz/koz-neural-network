from math import e, exp

import numpy as np


def tanh(x):
    return np.tanh(x)


def tanh_prime(x):
    meta = 1 - np.tanh(x) ** 2
    result = extend_result(meta)
    return result


def tanh_prime_v2(x):
    return 1 - np.tanh(x) ** 2


def relu(x):
    return np.maximum(0, x)


def relu_prime(x):
    meta = (x > 0).astype(x.dtype)
    result = extend_result(meta)
    return result


def elu(x, a=0.01):
    return a * (((e) ** 2) - 1) if x <= 0 else x


def elu_prime(x, a=0.01):
    return elu(x, a) + a if x <= 0 else 1


def softmax(x):
    shift_x = x - np.max(x)
    exps = np.exp(shift_x)
    product = exps / np.sum(exps)
    return product


def softmax_prime(x):
    probabilities = softmax(x)
    # we have all information to evaluate the derivative for each parameter.
    jacobean_matrix = softmax_grad(probabilities)
    return jacobean_matrix
    # because jacobean_matrix is symmetrical with respect to the diagonal, we don't have to take the transpose of the
    # result.


def softmax_grad(s):
    vector_s = s[0]
    square_dimensions = (vector_s.shape[0], vector_s.shape[0])
    jacobean_matrix = np.zeros(square_dimensions)
    for i in range(len(jacobean_matrix)):
        for j in range(len(jacobean_matrix)):
            if i == j:
                jacobean_matrix[i][j] = vector_s[i] * (1 - vector_s[i])
            else:
                jacobean_matrix[i][j] = -vector_s[i] * vector_s[j]
    return jacobean_matrix


def argmax(x):
    predictions = []
    for e in x:
        predictions.append(np.argmax(e, axis=1)[0])
    return np.array(predictions)


def sigmoid(x, b=1):
    def helper(x):
        if x >= 0:
            z = exp(-x * b)
            return 1 / (1 + z)
        else:
            z = exp(x * b)
            return z / (1 + z)

    z = np.array([list(map(helper, x[0]))])
    # z = 1 / (1 + np.exp(-x * b))
    return z


def swish(x, b=1):
    z = x * sigmoid(x, b)
    return z


def swish_prime(x, b=1):
    meta = sigmoid(x, b) + x * b * sigmoid(x, b) * (1 - sigmoid(x, b))
    result = extend_result(meta)
    return result


def log_softmax(x):
    a = np.max(x)
    product = x - a - np.log(np.sum(np.exp(x - a)))
    return product


def log_softmax_prime(x):
    probabilities = log_softmax(x)
    # we have all information to evaluate the derivative for each parameter.
    jacobean_matrix = log_softmax_grad(probabilities)
    return jacobean_matrix
    # because jacobean_matrix is symmetrical with respect to the diagonal, we don't have to take the transpose of the
    # result.


def log_softmax_grad(s):
    vector_s = s[0]
    square_dimensions = (vector_s.shape[0], vector_s.shape[0])
    jacobean_matrix = np.zeros(square_dimensions)
    for i in range(len(jacobean_matrix)):
        for j in range(len(jacobean_matrix)):
            if i == j:
                jacobean_matrix[i][j] = 1 - vector_s[i]
            else:
                jacobean_matrix[i][j] = -vector_s[j]
    return jacobean_matrix


def extend_result(meta):
    square_dimensions = (meta.shape[1], meta.shape[1])
    zero_extended = np.zeros(square_dimensions)
    np.fill_diagonal(zero_extended, meta)
    result = zero_extended
    return result
