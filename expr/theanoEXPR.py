__author__ = 'eric'

import theano.tensor as T


def sigmoid(x):
    return 1. / (1. + T.exp(-x))


def inverse_sigmoid(x):
    return T.log(x / (1. - x))
