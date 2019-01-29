import numpy as np


def prox_l_conj(prox_l):
    def fun(u, y, gamma):
        return u - gamma*prox_l(u/gamma, y, 1/gamma)
    return fun


def prox_G(prox_l_conj):
    def fun(u, y, gamma):
        n = u.shape[0]
        return (1/n)*prox_l_conj(n*u, y, n*gamma)
    return fun


def prox_abs(u, y, gamma):
    diff = u - y

    prox = np.copy(y)
    prox[diff < - gamma] = (u + gamma)[diff < - gamma]
    prox[diff > gamma] = (u - gamma)[diff > gamma]
    return prox


def prox_hinge(u, y, gamma):
    diff = u - 1/y

    prox = np.copy(1/y)
    prox[diff < - gamma*y] = (u + gamma*y)[diff < - gamma*y]
    prox[diff > 0] = u[diff > 0]
    return prox


def prox_abs_conj(u, y, gamma):
    diff = u - y

    prox = gamma * (u - y)
    prox[diff < -1/gamma] = -1
    prox[diff > 1 / gamma] = 1
    return prox