import numpy as np


def mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))


def mse_prime(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.size


def cross_entropy(y_true, y_pred):
    return -np.log(np.sum(np.multiply(y_true, y_pred)))


def cross_entropy_prime(y_true, y_pred):
    true_class_index = (y_true != 0).argmax()
    denominator = y_pred[0][true_class_index]
    result = -1 / denominator
    derivatives = np.zeros(y_pred.shape)
    derivatives[0][true_class_index] = result
    return derivatives
