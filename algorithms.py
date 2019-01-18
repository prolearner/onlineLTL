import numpy as np
import data_generator as gen

from losses import Loss
from utils import PrintLevels

class InnerSolver:
    def __init__(self, lmbd=0.0, h=0.0, loss_class:Loss=None, gamma=None):
        self.lmbd = lmbd
        self.h = h
        self.loss_f = loss_class.get
        self.grad_loss_f = loss_class.grad
        self.gamma = gamma

        self.v = None
        self.v_mean = None
        self.w = h

        # dual variables
        self.prox_f = loss_class.prox
        self.conj_f = loss_class.conj

        self.u = None

    def _init(self, n_iter, dim, h=None, p=None):
        self.h = h if h is not None else self.h
        self.v = np.zeros((n_iter + 1, dim))  # translated weight vector

        if p is not None:
            self.u = np.zeros((n_iter+1, p.shape[0]))
            self.u[0] = p

    def __call__(self, X_n, y_n, h=None, verbose=0, **kwargs):
        raise NotImplementedError

    def outer_gradient(self):
        return - self.lmbd * self.v[-1]

    def evaluate(self, X, y, k=None):
        if k is None:
            return np.mean(self.loss_f(np.dot(X, self.w), y))
        else:
            return np.mean(self.loss_f(np.dot(X, self.v[k] + self.h), y))

    def predict(self, X):
        return np.dot(X, self.w)

    def train_loss(self, X, y, k):
        return np.mean(self.loss_f(np.dot(X, self.v[k] + self.h), y)) + self.lmbd*0.5*np.linalg.norm(self.v[k])

    def init_from_solver(self, other):
        self.h = other.h
        self.lmbd = other.lmbd
        self.v = None
        self.v_mean = None
        self.u = None
        self.w = other.h
        self.loss_f = other.loss_f
        self.grad_loss_f = other.grad_loss_f
        self.conj_f = other.conj_f
        self.prox_f = other.prox_f


class ISTA(InnerSolver):
    def __call__(self, X_n, y_n, h=None, verbose=0, n_iter=1000, rx=1, **kwargs):
        dim = X_n.shape[1]
        n = X_n.shape[0]

        # initialization
        self._init(n_iter, dim, h, p=np.zeros(n))
        if verbose > PrintLevels.inner_train:
            print('--- start inner training')

        gamma = self.lmbd / (n*(rx**2))
        for k in range(n_iter):
            self.u[k+1] = self.prox_f(self.u[k] - gamma*self.grad_smooth_obj_part(self.u[k], X_n), y_n, gamma)
            self.v[k+1] = -(1/self.lmbd)*X_n.T @ self.u[k+1]

            if verbose > PrintLevels.inner_train:
                print('dual train loss iter   %d: %f' % (k, self.train_loss_dual(X_n, y_n, k)))
                print('primal train loss iter %d: %f' % (k, self.train_loss(X_n, y_n, k)))

        self.v_mean = self.v[:-1].mean(axis=0)
        self.w = self.v_mean + self.h
        return self.u

    def smooth_obj_part(self, u, X_n):
        return 0.5*(1/self.lmbd) * ((X_n.T @ u).T @ (X_n.T @ u)) - (X_n @ self.h).T @ u

    def grad_smooth_obj_part(self, u, X_n):
        return (1/self.lmbd) * X_n @ X_n.T @ u - X_n @ self.h

    def evaluate_dual(self, X, y):
        return self.conj_f(np.dot(X, self.w), y)

    def train_loss_dual(self, X, y, k):
        return self.conj_f(self.u[k], y) + self.smooth_obj_part(self.u[k], X)


class FISTA(InnerSolver):
    def __call__(self, X_n, y_n, h=None, verbose=0, n_iter=100, rx=1, **kwargs):
        dim = X_n.shape[1]
        n = X_n.shape[0]

        # initialization
        t = 1
        p = np.zeros(n)
        self._init(n_iter, dim, h, p=p)
        if verbose > PrintLevels.inner_train:
            print('--- start inner training')

        gamma = self.lmbd / (n*(rx**2))
        for k in range(n_iter):
            self.u[k+1] = self.prox_f(p - gamma*self.grad_smooth_obj_part(p, X_n), y_n, gamma)
            self.v[k+1] = -(1/self.lmbd)*X_n.T @ self.u[k+1]
            t_prec = t
            t = 0.5 + 0.5*np.sqrt(1+4*(t_prec**2))
            p = self.u[k+1] + ((t_prec-1)/t)*(self.u[k+1] - self.u[k])

            if verbose > PrintLevels.inner_train:
                print('dual train loss iter   %d: %f' % (k, self.train_loss_dual(X_n, y_n, k)))
                print('primal train loss iter %d: %f' % (k, self.train_loss(X_n, y_n, k)))

        self.v_mean = self.v[:-1].mean(axis=0)
        self.w = self.v_mean + self.h
        return self.u

    def smooth_obj_part(self, u, X_n):
        return 0.5*(1/self.lmbd) * ((X_n.T @ u).T @ (X_n.T @ u)) - (X_n @ self.h).T @ u

    def grad_smooth_obj_part(self, u, X_n):
        return (1/self.lmbd) * X_n @ X_n.T @ u - X_n @ self.h

    def evaluate_dual(self, X, y):
        return self.conj_f(np.dot(X, self.w), y)

    def train_loss_dual(self, X, y, k):
        return self.conj_f(self.u[k], y) + self.smooth_obj_part(self.u[k], X)


class InnerSSubGD(InnerSolver):
    def __call__(self, X_n, y_n, h=None, verbose=0, **kwargs):
        n = X_n.shape[0]
        dim = X_n.shape[1]
        self._init(n, dim, h)

        if verbose > PrintLevels.inner_train:
            print('--- start inner training')

        for k in range(n):
            x, y = X_n[k], y_n[k]
            gamma = 1 / ((k+1) * self.lmbd) if self.gamma is None else self.gamma  # step size
            self.v[k + 1] = self.v[k] - gamma*(self.grad_loss_f(np.dot(x, self.v[k] + self.h), y)*x + self.lmbd*self.v[k])

            if verbose > PrintLevels.inner_train:
                print('train reg loss iter %d: %f' % (k, self.train_loss(X_n, y_n, k)))
                print('train loss iter     %d: %f' % (k, self.evaluate(X_n, y_n, k)))

        self.v_mean = self.v[:-1].mean(axis=0)
        self.w = self.v_mean + self.h
        return self.v


class InnerSubGD(InnerSolver):
    def __call__(self, X_n, y_n, h=None, verbose=0, n_iter=1000, **kwargs):
        dim = X_n.shape[1]
        self._init(n_iter, dim, h)

        if verbose > PrintLevels.inner_train:
            print('--- start inner training')

        for k in range(n_iter):
            gamma = 1 / ((k+1) * self.lmbd) if self.gamma is None else self.gamma  # step size
            self.v[k + 1] = self.v[k] - gamma*(np.mean((self.grad_loss_f(np.dot(X_n, self.v[k] + self.h), y_n).T*
                                                        X_n.T).T, axis=0)
                                               + self.lmbd*self.v[k])

            if verbose > PrintLevels.inner_train:
                print('train reg loss iter %d: %f' % (k, self.train_loss(X_n, y_n, k)))
                print('train loss iter     %d: %f' % (k, self.evaluate(X_n, y_n, k)))

        self.v_mean = self.v[:-1].mean(axis=0)
        self.w = self.v_mean + self.h
        return self.v


def meta_ssgd(alpha, X, y, data_valid, inner_solver: InnerSolver, inner_solver_test: InnerSolver,
              eval_online=True, verbose=0):
    dim = X[0].shape[1]
    n_tasks = len(X)  # T
    n_tasks_val = len(data_valid['X_train'])

    hs = np.zeros((n_tasks+1, dim))
    losses_val = np.zeros((n_tasks+1, n_tasks_val))

    for t in range(n_tasks+1):
        if t < n_tasks:
            inner_solver(X_n=X[t], y_n=y[t], h=hs[t], verbose=verbose)
            hs[t+1] = hs[t] - alpha * inner_solver.outer_gradient()

        if eval_online:
            inner_solver_test.init_from_solver(inner_solver)
            inner_solver_test.h = hs[:t+1].mean(axis=0)

            losses_val[t] = LTL_evaluation(X=data_valid['X_train'], y=data_valid['Y_train'],
                                           X_test=data_valid['X_test'], y_test=data_valid['Y_test'],
                                           inner_solver=inner_solver_test, verbose=verbose)
            if verbose > PrintLevels.outer_eval:
                print(str(t) + '-' + 'loss-val  : ', np.mean(losses_val[t]), np.std(losses_val[t]))

    return hs, losses_val


def lmbd_theory(rx, L, sigma_h, n):
    return (np.sqrt(2) * rx * L / sigma_h) * np.sqrt(2 * (np.log(n) + 1) / n)


def lmbd_theory_meta(rx, L, sigma_bar, n):
    return (2 * np.sqrt(2) * rx * L / sigma_bar) * np.sqrt((np.log(n) + 1) / n)


def alpha_theory(rx, L, w_bar, T, n):
    return np.sqrt(2)*np.linalg.norm(w_bar)/(L*rx) * np.sqrt(1/(T*(1 + 4*(np.log(n) + 1)/n)))


def no_train_evaluation(X_test, y_test, inner_solvers, verbose=0):
    n_tasks = len(X_test)  # T

    losses = np.zeros(n_tasks)
    accs = np.zeros(n_tasks)
    for t in range(n_tasks):
        # Testing
        losses[t] = inner_solvers[t].evaluate(X_test[t], y_test[t])
        accs[t] = np.mean(np.maximum(np.sign(inner_solvers[t].predict(X_test[t])*y_test[t]), 0))

        if verbose > PrintLevels.outer_eval:
            print('loss-test', losses[t])
            print('accs-test', accs[t])

    return losses


def LTL_evaluation(X, y, X_test, y_test, inner_solver, verbose=0):
    n_tasks = len(X)  # T

    losses = np.zeros(n_tasks)
    accs = np.zeros(n_tasks)
    for t in range(n_tasks):
        inner_solver(X_n=X[t], y_n=y[t], verbose=verbose)

        # Testing
        losses[t] = inner_solver.evaluate(X_test[t], y_test[t])
        accs[t] = np.mean(np.maximum(np.sign(inner_solver.predict(X_test[t])*y_test[t]), 0))

        if verbose > PrintLevels.inner_eval:
            print('loss-test', losses[t])
            print('accs-test', accs[t])

    return losses


# Tests

def t_inner_algo(inner_solver_class=FISTA, seed=0, n_iter=1000):
    from data_generator import TasksGenerator
    from losses import AbsoluteLoss, HingeLoss

    n_dims = 30

    tasks_gen = TasksGenerator(seed=seed, val_perc=0.0, n_dims=n_dims, n_train=100, tasks_generation='exp1')

    data_train, oracle_train = tasks_gen(n_tasks=1, n_train=100)

    inner_solver = inner_solver_class(lmbd=0.01, h=np.zeros(n_dims), loss_class=AbsoluteLoss, gamma=None)

    LTL_evaluation(data_train['X_train'], data_train['Y_train'], data_train['X_test'], data_train['Y_test'],
                   inner_solver=inner_solver, verbose=1)


if __name__ == '__main__':
    from eperiments_ICML import exp1

    #exp1(seed=0)
    t_inner_algo(FISTA)