import numpy as np
from sklearn.model_selection import train_test_split


class TasksGenerator:
    def __init__(self, seed=0, n_tasks=120, val_perc=0.5, n_dims=30, n_train=20, n_test=100, y_snr=5, task_std=1,
                 tasks_generation='exp1', w_bar=4):
        self.seed = seed
        self.n_tasks = n_tasks
        self.val_perc = val_perc
        self.n_dims = n_dims
        self.n_train = n_train
        self.n_test = n_test
        self.y_snr = y_snr
        self.task_std = task_std

        self.w_bar = w_bar*np.ones(n_dims) if type(w_bar) == int else w_bar

        if tasks_generation == 'exp1':
            self.rx = 1
            self._task_gen_func = regression_tasks_generator
        elif tasks_generation == 'exp2':
            self.rx = 1
            self._task_gen_func = regression_tasks_generator_v2
        elif tasks_generation == 'expclass':
            self.rx = 1
            self._task_gen_func = classification_tasks_generator

        np.random.seed(seed)

    def sigma_h(self, h):
        # returns sqrt(2)*Var_h in paper notation
        return np.sqrt((self.w_bar - h).T @ (self.w_bar - h) + self.task_std ** 2)

    def gen_tasks(self, n_tasks=None, n_train=None, n_test=None, n_dims=None, y_snr=None, task_std=None):
        if n_tasks is None:
            n_tasks = self.n_tasks
        if n_train is None:
            n_train = self.n_train
        if n_dims is None:
            n_dims = self.n_dims
        if y_snr is None:
            y_snr = self.y_snr
        if task_std is None:
            task_std = self.task_std
        if n_test is None:
            n_test = self.n_test

        return self._task_gen_func(n_tasks, self.val_perc, n_dims, n_train, n_test, y_snr, task_std,
                                   w_bar=self.w_bar)

    def __call__(self, n_tasks=None, n_train=None, n_test=None,  n_dims=None, y_snr=None, task_std=None):
        return self.gen_tasks(n_tasks, n_train, n_test, n_dims, y_snr, task_std)


def regression_tasks_generator(n_tasks=120, val_perc=0.5, n_dims=30, n_train=20, n_test=100, y_snr=5, task_std=1
                               , w_bar=4):

    # Notation similar to NIPS
    w_bar = w_bar * np.ones(n_dims) if type(w_bar) == int else w_bar

    W_true = np.zeros((n_dims, n_tasks))
    X_train, Y_train = [None] * n_tasks, [None] * n_tasks
    X_val, Y_val = [None] * n_tasks, [None] * n_tasks
    X_test, Y_test = [None] * n_tasks, [None] * n_tasks

    for i in range(n_tasks):
        center = 0

        X_n = np.random.randn(n_train + n_test, n_dims)
        X_n = center + X_n / np.linalg.norm(X_n, axis=1, keepdims=True)

        w = w_bar + task_std * np.random.randn(n_dims)

        # eps sampled from a zero-mean gaussian with std chosen to have snr=5
        clean_y_n = X_n @ w  # scalar product
        var_signal = np.var(clean_y_n)
        std_eps = np.sqrt(var_signal / y_snr)
        eps_n = np.random.randn(n_train + n_test) * std_eps

        y_n = clean_y_n + eps_n

        X_train_all, X_test[i], Y_train_all, Y_test[i] = train_test_split(X_n, y_n, test_size=n_test)
        X_train[i], X_val[i], Y_train[i], Y_val[i] = train_test_split(X_train_all, Y_train_all, test_size=val_perc)

        W_true[:, i] = w.ravel()

    data = {'X_train': X_train, 'Y_train': Y_train, 'X_val': X_val, 'Y_val': Y_val, 'X_test': X_test, 'Y_test': Y_test}
    oracle = {'w_bar': w_bar, 'W_true': W_true}
    return data, oracle


def classification_tasks_generator(n_tasks=120, val_perc=0.5, n_dims=30, n_train=20, n_test=100, y_snr=5, task_std=1,
                                   y_dist='sign', w_bar=4):
    w_bar = w_bar * np.ones(n_dims) if type(w_bar) == int else w_bar
    W_true = np.zeros((n_dims, n_tasks))
    X_train, Y_train = [None] * n_tasks, [None] * n_tasks
    X_val, Y_val = [None] * n_tasks, [None] * n_tasks
    X_test, Y_test = [None] * n_tasks, [None] * n_tasks

    for i in range(n_tasks):
        center = 0

        X_n = np.random.randn(n_train + n_test, n_dims)
        X_n = center + X_n / np.linalg.norm(X_n, axis=1, keepdims=True)

        w = w_bar + task_std * np.random.randn(n_dims)

        # eps sampled from a zero-mean gaussian with std chosen to have snr=5
        clean_y_n = X_n @ w  # scalar product
        var_signal = np.var(clean_y_n)
        std_eps = np.sqrt(var_signal / y_snr)
        eps_n = np.random.randn(n_train + n_test) * std_eps

        if y_dist == 'logistic':
            s = 1
            y_n_uniform = np.random.rand(*clean_y_n.shape)
            y_n = np.zeros(clean_y_n.shape)
            p_y_given_x = 1/(1 + np.exp(-clean_y_n/s))
            y_n[y_n_uniform > p_y_given_x] = 1
        elif y_dist == 'sign':
            y_n = np.sign(clean_y_n + eps_n)

        X_train_all, X_test[i], Y_train_all, Y_test[i] = train_test_split(X_n, y_n, test_size=n_test)
        X_train[i], X_val[i], Y_train[i], Y_val[i] = train_test_split(X_train_all, Y_train_all, test_size=val_perc)

        W_true[:, i] = w.ravel()

    data = {'X_train': X_train, 'Y_train': Y_train, 'X_val': X_val, 'Y_val': Y_val, 'X_test': X_test, 'Y_test': Y_test}
    oracle = {'w_bar': w_bar, 'W_true': W_true}
    return data, oracle


def regression_tasks_generator_v2(n_tasks=120, val_perc=0.5, n_dims=30, n_train=20, n_test=100, y_snr=5, task_std=1,
                                  w_bar=4):

    # Notation similar to NIPS

    w_bar = w_bar * np.ones(n_dims) if type(w_bar) == int else w_bar

    W_true = np.zeros((n_dims, n_tasks))
    X_train, Y_train = [None] * n_tasks, [None] * n_tasks
    X_val, Y_val = [None] * n_tasks, [None] * n_tasks
    X_test, Y_test = [None] * n_tasks, [None] * n_tasks

    for i in range(n_tasks):
        center1 = 0
        center2 = 1

        selector = np.random.randint(0, 2, n_train).astype(bool)
        X_n = np.zeros((n_train, n_dims))

        X_n[selector] = center1 + np.random.randn(len(X_n[selector]), n_dims)
        X_n[~selector] = center2 + np.random.randn(len(X_n[~selector]), n_dims)

        X_n = X_n / np.linalg.norm(X_n, axis=1, keepdims=True) # is there projection in this case? (referring to nips)

        w = w_bar + task_std * np.random.randn(n_dims)

        # eps sampled from a zero-mean gaussian with std chosen to have snr=5
        clean_y_n = X_n @ w  # scalar product
        var_signal = np.var(clean_y_n)
        std_eps = np.sqrt(var_signal / y_snr)
        eps_n = np.random.randn(n_train) * std_eps

        y_n = clean_y_n + eps_n

        X_train_all, X_test[i], Y_train_all, Y_test[i] = train_test_split(X_n, y_n, test_size=n_test)
        X_train[i], X_val[i], Y_train[i], Y_val[i] = train_test_split(X_train_all, Y_train_all, test_size=val_perc)

        W_true[:, i] = w.ravel()

    data = {'X_train': X_train, 'Y_train': Y_train, 'X_val': X_val, 'Y_val': Y_val, 'X_test': X_test, 'Y_test': Y_test}
    oracle = {'w_bar': w_bar, 'W_true': W_true}
    return data, oracle


if __name__ == '__main__':
    data, oracle = regression_tasks_generator()
    print(data, oracle)
