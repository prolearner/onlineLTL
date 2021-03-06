import numpy as np

from losses import Loss, HingeLoss, AbsoluteLoss
from utils import PrintLevels
import numba


class InnerSolver:
    default_n_iter = 2000
    name = None

    def __init__(self, lmbd=0.0, h=0.0, loss_class: Loss=None, gamma=None):
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

    def __str__(self):
        return "%s(%r)" % (self.__class__, self.__dict__)

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
        return np.mean(self.loss_f(X @ (self.v[k] + self.h), y)) + self.lmbd*0.5*(self.v[k].T @ self.v[k])

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


class NoOpt(InnerSolver):
    name = 'noopt'
    def __call__(self, X_n, y_n, h=None, verbose=0, n_iter=InnerSolver.default_n_iter, **kwargs):
        dim = X_n.shape[1]
        n = X_n.shape[0]

        # get rx from data:
        rx = get_rx(X_n)

        # initialization
        self._init(n_iter, dim, h, p=np.zeros(n))
        if verbose > PrintLevels.inner_train:
            print('--- no inner training')

        return self.u


class ISTA(InnerSolver):
    name = 'ista'

    def __call__(self, X_n, y_n, h=None, verbose=0, n_iter=InnerSolver.default_n_iter, **kwargs):
        dim = X_n.shape[1]
        n = X_n.shape[0]

        # get rx from data:
        rx = get_rx(X_n)

        # initialization
        self._init(n_iter, dim, h, p=np.zeros(n))
        if verbose > PrintLevels.inner_train:
            print('--- start inner training')

        gamma = self.lmbd / (n*(rx**2))
        for k in range(n_iter):
            grad = self.grad_smooth_obj_part(self.u[k], X_n)
            self.u[k+1] = self.prox_f(self.u[k] - gamma*grad, y_n, gamma)
            self.v[k+1] = -(1/self.lmbd)*X_n.T @ self.u[k+1]

            if verbose > PrintLevels.inner_train:
                print('primal, dual train loss iter   %d: %f, %f' % (k, self.train_loss_dual(X_n, y_n, k),
                                                                     self.train_loss(X_n, y_n, k)))

            if self.dual_gap(X_n, y_n, k) < 1e-6:
                break

        # print('ISTA iter {}'.format(k))

        self.u = self.u[:k+2]
        self.v = self.v[:k+2]
        self.v_mean = self.v[-1]
        self.w = self.v_mean + self.h
        return self.u

    def dual_gap(self, X_n, y_n, k):
        lp = self.train_loss(X_n, y_n, k)
        ld = self.train_loss_dual(X_n, y_n, k)
        return lp + ld

    def sol_distance(self, k):
        return np.linalg.norm(self.u[k] - self.u[k-1])

    def smooth_obj_part(self, u, X):
        return (0.5/self.lmbd)*(u.T @ X @ X.T @ u) - (u.T @ X @ self.h)

    def grad_smooth_obj_part(self, u, X):
        return (1/self.lmbd) * (X @ X.T @ u) - X @ self.h

    def evaluate_dual(self, y, k):
        return self.conj_f(self.u[k], y)

    def train_loss_dual(self, X, y, k):
        return self.conj_f(self.u[k], y) + self.smooth_obj_part(self.u[k], X)


class FISTA(ISTA):
    """ ERM in the paper (look at Appendix J)"""
    name = 'fista'

    def __call__(self, X_n, y_n, h=None, verbose=0, n_iter=InnerSolver.default_n_iter, **kwargs):
        dim = X_n.shape[1]
        n = X_n.shape[0]

        # get rx from data:
        rx = get_rx(X_n)

        # initialization
        t = 1
        p = np.zeros(n)
        self._init(n_iter, dim, h, p=p)
        if verbose > PrintLevels.inner_train:
            print('--- start inner training')

        gamma = self.lmbd / (n*(rx**2))
        optimal = False
        for k in range(n_iter):
            self.u[k+1] = self.prox_f(p - gamma*self.grad_smooth_obj_part(p, X_n), y_n, gamma)
            self.v[k+1] = -(1/self.lmbd)*X_n.T @ self.u[k+1]
            t_prec = t
            t = 0.5 + 0.5*np.sqrt(1+4*(t_prec**2))
            p = self.u[k+1] + ((t_prec-1)/t)*(self.u[k+1] - self.u[k])

            if verbose > PrintLevels.inner_train:
                print('primal, dual train loss iter   %d: %f, %f' % (k, self.train_loss_dual(X_n, y_n, k),
                                                                     self.train_loss(X_n, y_n, k)))
            if self.dual_gap(X_n, y_n, k+1) < 1e-6:
                break

        # print('ISTA iter {}'.format(k))

        self.u = self.u[:k+2]
        self.v = self.v[:k+2]
        self.v_mean = self.v[-1]
        self.w = self.v_mean + self.h
        return self.u


class InnerSSubGD(InnerSolver):
    """SGD (The proposed method) in the paper"""
    name = 'ssubgd'

    def __call__(self, X_n, y_n, h=None, verbose=0, n_iter=None, **kwargs):
        n = X_n.shape[0]
        n_iter = n if n_iter is None else n_iter
        r_indices = np.random.randint(0, n, size=n_iter)
        dim = X_n.shape[1]
        self._init(n_iter, dim, h)

        if verbose > PrintLevels.inner_train:
            print('--- start inner training')

        for k in range(n_iter):
            x, y = X_n[r_indices[k]], y_n[r_indices[k]]
            gamma = 1 / ((k+1) * self.lmbd) if self.gamma is None else self.gamma  # step size
            self.v[k + 1] = self.v[k] - gamma*(self.grad_loss_f(np.dot(x, self.v[k] + self.h), y)*x + self.lmbd*self.v[k])

            if verbose > PrintLevels.inner_train:
                print('train reg loss iter %d: %f' % (k, self.train_loss(X_n, y_n, k)))
                print('train loss iter     %d: %f' % (k, self.evaluate(X_n, y_n, k)))

        self.v_mean = self.v[:-1].mean(axis=0)
        self.w = self.v_mean + self.h
        return self.v


class InnerSubGD(InnerSolver):
    name = 'subgd'

    def __call__(self, X_n, y_n, h=None, verbose=0, n_iter=InnerSolver.default_n_iter, **kwargs):
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


inner_dict = {InnerSSubGD.name: InnerSSubGD, FISTA.name: FISTA, ISTA.name: ISTA, InnerSubGD.name: InnerSubGD,
              NoOpt.name: NoOpt}


def inner_solver_selector(solver_str):
    return inner_dict[solver_str]


def eval_biases(data_valid, inner_solver_test_list, metric_dict, verbose=0):
    T = len(inner_solver_test_list)  # T
    n_tasks_val = len(data_valid['X_train'])

    metric_results_dict = {'loss': np.zeros((T, n_tasks_val))}
    for metric_name in metric_dict:
        metric_results_dict[metric_name] = np.zeros((T, n_tasks_val))

    for t in range(T):
        mr_dict = LTL_evaluation(X=data_valid['X_train'], y=data_valid['Y_train'],
                                 X_test=data_valid['X_test'], y_test=data_valid['Y_test'],
                                 inner_solver=inner_solver_test_list[t], metric_dict=metric_dict, verbose=verbose)

        for metric_name, res in mr_dict.items():
            metric_results_dict[metric_name][t] = res

        if verbose > PrintLevels.outer_eval:
            for metric_name, res in mr_dict.items():
                print(str(t) + '-' + metric_name + '-val  : ', np.mean(res), np.std(res))

    return metric_results_dict


def meta_ssgd(alpha, X, y, data_valid, inner_solver: InnerSolver, inner_solver_test: InnerSolver, metric_dict={},
              eval_online=True, verbose=0):
    dim = X[0].shape[1]
    n_tasks = len(X)  # T
    n_tasks_val = len(data_valid['X_train'])

    hs = np.zeros((n_tasks+1, dim))
    metric_results_dict = {'loss': np.zeros((n_tasks+1, n_tasks_val))}
    for metric_name in metric_dict:
        metric_results_dict[metric_name] = np.zeros((n_tasks+1, n_tasks_val))

    for t in range(n_tasks+1):
        if t < n_tasks:
            inner_solver(X_n=X[t], y_n=y[t], h=hs[t], verbose=verbose)
            hs[t+1] = hs[t] - alpha * inner_solver.outer_gradient()

        if eval_online:
            inner_solver_test.init_from_solver(inner_solver)
            inner_solver_test.h = hs[:t+1].mean(axis=0)

            mr_dict = LTL_evaluation(X=data_valid['X_train'], y=data_valid['Y_train'],
                                           X_test=data_valid['X_test'], y_test=data_valid['Y_test'],
                                           inner_solver=inner_solver_test, metric_dict=metric_dict, verbose=verbose)

            for metric_name, res in mr_dict.items():
                metric_results_dict[metric_name][t] = res

            if verbose > PrintLevels.outer_eval:
                for metric_name, res in mr_dict.items():
                    print(str(t) + '-' + metric_name + '-val  : ', np.mean(res), np.std(res))

    return hs,  metric_results_dict


def get_rx(X):
    return np.max(np.linalg.norm(X, axis=1))


def lmbd_theory(rx, L, sigma_h, n):
    return (np.sqrt(2) * rx * L / sigma_h) * np.sqrt(2 * (np.log(n) + 1) / n)


def lmbd_theory_meta(rx, L, sigma_bar, n):
    return (2 * np.sqrt(2) * rx * L / sigma_bar) * np.sqrt((np.log(n) + 1) / n)


def alpha_theory(rx, L, w_bar, T, n):
    return np.sqrt(2)*np.linalg.norm(w_bar)/(L*rx) * np.sqrt(1/(T*(1 + 4*(np.log(n) + 1)/n)))


def no_train_evaluation(X_test, y_test, inner_solvers, metric_dict={}, verbose=0):
    n_tasks = len(X_test)  # T

    losses = np.zeros(n_tasks)
    metric_results_dict = {}
    for metric_name in metric_dict:
        metric_results_dict[metric_name] = np.zeros(n_tasks)
    for t in range(n_tasks):
        # Testing
        losses[t] = inner_solvers[t].evaluate(X_test[t], y_test[t])
        for metric_name, metric_f in metric_dict.items():
            metric_results_dict[metric_name][t] = metric_f(y_test[t], inner_solvers[t].predict(X_test[t]))

        if verbose > PrintLevels.outer_eval:
            print('loss-test', losses[t])
            for metric_name, res in metric_results_dict.items():
                print(metric_name + '-test', res[t])

    metric_results_dict['loss'] = losses
    return metric_results_dict


def LTL_evaluation(X, y, X_test, y_test, inner_solver, metric_dict={}, verbose=0):
    n_tasks = len(X)  # T

    losses = np.zeros(n_tasks)
    metric_results_dict = {}
    for metric_name in metric_dict:
        metric_results_dict[metric_name] = np.zeros(n_tasks)

    ws = []
    #accs = np.zeros(n_tasks)
    for t in range(n_tasks):
        inner_solver(X_n=X[t], y_n=y[t], verbose=verbose)

        # Testing
        losses[t] = inner_solver.evaluate(X_test[t], y_test[t])
        for metric_name, metric_f in metric_dict.items():
            metric_results_dict[metric_name][t] = metric_f(y_test[t], inner_solver.predict(X_test[t]))

        # accs[t] = np.mean(np.maximum(np.sign(inner_solver.predict(X_test[t])*y_test[t]), 0))
        ws.append(inner_solver.w)
        if verbose > PrintLevels.inner_eval:
            print('loss-test', losses[t])
            for metric_name, res in metric_results_dict.items():
                print(metric_name + '-test', res[t])


            # print('accs-test', accs[t])
    metric_results_dict['loss'] = losses
    return metric_results_dict


def train_and_evaluate(inner_solvers, data_train, data_val, name='', verbose=0):
    losses_train = []
    for i in range(len(inner_solvers)):
        losses_train.append(LTL_evaluation(data_train['X_train'], data_train['Y_train'],
                                           data_train['X_test'], data_train['Y_test'],
                                           inner_solvers[i], verbose=verbose))

    best_solver_idx = np.argmin(np.mean(np.concatenate([np.expand_dims(l, 0) for l in losses_train]), axis=1))
    print('best ' + name + ': ' + str(inner_solvers[best_solver_idx].lmbd))

    losses_val = LTL_evaluation(data_val['X_train'], data_val['Y_train'],
                                data_val['X_test'], data_val['Y_test'],
                                inner_solvers[best_solver_idx], verbose=verbose)
    return losses_val, inner_solvers[best_solver_idx]


def save_3d_csv(path, arr3d: np.ndarray, hyper_str=None):
    for i in range(arr3d.shape[1]):
        str = path + '-'
        if hyper_str:
            str += hyper_str[i]
        else:
            str += hyper_str[i]
        str += '.csv'

        np.savetxt(str, arr3d[:, i], delimiter=",")


# Tests
def t_inner_algo(inner_solver_class=(InnerSubGD, FISTA, ISTA), seed=2, n_iter=2000):
    from data.data_generator import TasksGenerator
    from losses import AbsoluteLoss

    n_dims = 10
    y_snr = 100000000000000
    n_train = 10
    tasks_gen = TasksGenerator(seed=seed, val_perc=0.0, n_dims=n_dims, n_train=n_train, y_snr=y_snr, tasks_generation='exp1',
                               task_std=0, w_bar=4)

    data_train, oracle_train = tasks_gen(n_tasks=10, n_train=n_train)

    import copy
    w_dict = {}
    losses_dict = {}
    task_n = 2
    X_train, Y_train = data_train['X_train'][task_n], data_train['Y_train'][task_n]
    x_cp, Y_cp = copy.copy(X_train), copy.copy(Y_train)
    X_test, Y_test = data_train['X_test'][task_n], data_train['Y_test'][task_n]
    for isc in inner_solver_class:
        inner_solver = isc(lmbd=0.01, h=np.zeros(n_dims), loss_class=AbsoluteLoss, gamma=None)
        inner_solver(X_train, Y_train, n_iter=n_iter,  verbose=4)
        losses_dict[isc] = inner_solver.train_loss(X_train, Y_train, -1)
        w_dict[isc] = inner_solver.w

    print('ws', w_dict)
    print('losses', losses_dict)
    print('ws distance', np.linalg.norm(w_dict[InnerSubGD]-w_dict[FISTA]))


if __name__ == '__main__':

    #exp1(seed=0)
    t_inner_algo()


