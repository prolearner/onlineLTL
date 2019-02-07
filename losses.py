import numpy as np
import prox
import numba


class Loss:
    L = None
    name = None

    @staticmethod
    def get(yhat, y):
        return np.maximum(1 - yhat * y, 0)

    @staticmethod
    def grad(yhat, y):
        raise NotImplementedError

    @staticmethod
    def conj(u, y):
        raise NotImplementedError

    @staticmethod
    def prox(u, y, gamma):
        raise NotImplementedError


class HingeLoss(Loss):
    L = 1
    name = 'Hinge'
    prox_G = prox.prox_G(prox.prox_l_conj(prox.prox_hinge))


    @staticmethod
    # @numba.jit(nopython=True)
    def get(yhat, y):
        return np.maximum(1 - yhat * y, 0)

    @staticmethod
    # @numba.jit(nopython=True)
    def grad(yhat, y):
        if isinstance(y, np.int16):
            return 0 if 1 <= yhat*y else -y
        d = -y
        d[1 <= yhat * y] = 0
        return d

    @staticmethod
    # @numba.jit(nopython=True)
    def conj(u, y):
        n = u.shape[0]
        res = n * u / y
        res[(-1 > res) & (res > 0)] = np.inf
        res = np.mean(res)
        if res == np.inf:
            raise ValueError("infinite value in conjugate loss")
        return res

    @staticmethod
    # @numba.jit(nopython=True)
    def prox(u, y, gamma):
        return prox.prox_G_hinge_numba(u, y, gamma)


class AbsoluteLoss(Loss):
    L = 1
    name = 'Abs'
    prox_G = prox.prox_G(prox.prox_l_conj(prox.prox_abs))
    #prox_G = prox.prox_G(prox.prox_abs_conj)


    @staticmethod
    #@numba.jit(nopython=True)
    def get(yhat, y):
        return np.abs(yhat - y)

    @staticmethod
    @numba.jit(nopython=True)
    def grad(yhat, y):
        return np.sign(yhat - y)

    @staticmethod
    #@numba.jit(nopython=True)
    def conj(u, y):
        n = u.shape[0]
        res = n * u * y
        res[np.abs(n * u) > 1] = np.inf
        res_store = res
        res = np.mean(res)
        if res == np.inf:
            raise ValueError("infinite value in conjugate loss")
        return res

    @staticmethod
    #@numba.jit(nopython=True)
    def prox(u, y, gamma):
        return prox.prox_G_abs_numba(u, y, gamma)


class MSE(Loss):
    L = 1
    name = 'MSE'


    @staticmethod
    #@numba.jit(nopython=True)
    def get(yhat, y):
        return 0.5*((yhat - y)**2)

    @staticmethod
    @numba.jit(nopython=True)
    def grad(yhat, y):
        return yhat - y
