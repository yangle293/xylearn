import numpy as np


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


def inverse_sigmoid(x):
    return np.log(x / (1. - x))