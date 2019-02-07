import numpy as np
import numba


def prox_l_conj(prox_l):
    def fun(u, y, gamma):
        return u - gamma*prox_l(u/gamma, y, 1/gamma)
    return fun


def prox_G(prox_l_conj):
    def fun(u, y, gamma):
        n = u.shape[0]
        return (1/n)*prox_l_conj(n*u, y, n*gamma)
    return fun


def prox_G_numba(u, y, gamma, prox_l):
    n = u.shape[0]
    return u - gamma*prox_l(u/gamma, y, 1/(n*gamma))


# @numba.jit(nopython=True)
def prox_G_abs_numba(u, y, gamma):
    n = u.shape[0]
    prox = u - y*gamma
    return np.clip(prox, a_min=-1/n, a_max=1/n)


#@numba.jit(nopython=True)
def prox_G_abs_v2_numba(u, y, gamma):
    n = u.shape[0]
    return u - gamma*prox_abs(u/gamma, y, 1/(n*gamma))


@numba.jit(nopython=True)
def prox_G_hinge_numba(u, y, gamma):
    n = u.shape[0]
    prox = u - gamma/y
    prox[prox < - y/n] = (-y/n)[prox < - y/n]
    prox[prox > 0] = 0
    return prox


@numba.jit(nopython=True)
def prox_G_hinge_v2_numba(u, y, gamma):
    n = u.shape[0]
    return u - gamma*prox_hinge(u/gamma, y, 1/(n*gamma))


@numba.jit(nopython=True)
def prox_abs(u, y, gamma):
    diff = u - y

    prox = np.copy(y)
    prox[diff < - gamma] = (u + gamma)[diff < - gamma]
    prox[diff > gamma] = (u - gamma)[diff > gamma]

    # due to the nature of the conjugate of the absolute loss, the above two operations can cause problems


    return prox


@numba.jit(nopython=True)
def prox_hinge(u, y, gamma):
    diff = u - 1/y

    prox = np.copy(1/y)
    prox[diff < - gamma*y] = (u + gamma*y)[diff < - gamma*y]
    prox[diff > 0] = u[diff > 0]
    return prox


@numba.jit(nopython=True)
def prox_abs_conj(u, y, gamma):
    diff = u - y

    prox = gamma * (u - y)
    prox[diff < -1/gamma] = -1
    prox[diff > 1 / gamma] = 1
    return prox