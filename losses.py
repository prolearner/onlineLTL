import numpy as np
import prox


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
    def get(yhat, y):
        return np.maximum(1 - yhat * y, 0)

    @staticmethod
    def grad(yhat, y):
        d = np.copy(-y)
        d[1 <= yhat * y] = 0
        return d

    @staticmethod
    def conj(u, y):
        n = u.shape[0]
        res = n * u / y
        res[(-1 > res) & (res > 0)] = np.inf
        res = np.mean(res)
        return res

    @staticmethod
    def prox(u, y, gamma):
        return HingeLoss.prox_G(u, y, gamma)

    @staticmethod
    def prox2(u, y, gamma):
        n = y.shape[0]
        val = n * y * u

        prox = gamma * (n * u - 1 / y)
        prox[val < 1 - y ** 2 / (n * gamma)] = (-y / (gamma * (n ** 2)))[val < 1 - y ** 2 / (n * gamma)]
        prox[val > 1] = 0
        return prox


class AbsoluteLoss(Loss):
    L = 1
    name = 'Abs'
    prox_G = prox.prox_G(prox.prox_l_conj(prox.prox_abs))
    #prox_G = prox.prox_G(prox.prox_abs_conj)


    @staticmethod
    def get(yhat, y):
        return np.abs(yhat - y)

    @staticmethod
    def grad(yhat, y):
        return np.sign(yhat - y)

    @staticmethod
    def conj(u, y):
        n = u.shape[0]
        res = n * u * y
        res[np.abs(n * u) > 1] = np.inf
        res = np.mean(res)
        return res

    @staticmethod
    def prox2(u, y, gamma):
        n = y.shape[0]
        diff = n * u - y

        prox = gamma * (n * u - y)
        prox[diff < 0] = np.zeros(n)[diff < 0]
        prox[diff > 1 / (n * gamma)] = 1 / n
        return prox

    @staticmethod
    def prox(u, y, gamma):
        return AbsoluteLoss.prox_G(u, y, gamma)
