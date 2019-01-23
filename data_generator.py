import numpy as np
from sklearn.model_selection import train_test_split
import scipy.io as sio
from numpy.linalg import norm


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

        self.y_dist = None

        self.w_bar = w_bar*np.ones(n_dims) if type(w_bar) == int else w_bar

        if tasks_generation == 'exp1':
            self.rx = 1
            self._task_gen_func = regression_tasks_generator
        elif tasks_generation == 'exp2':
            self.rx = 1
            self._task_gen_func = regression_tasks_generator_v2

        elif tasks_generation == 'expclass':
            self.rx = 1
            self.y_dist = 'nonoisemargin'
            self._task_gen_func = classification_tasks_generator

        np.random.seed(seed)

    def sigma_h(self, h):
        # returns sqrt(2)*Var_h in paper notation
        return np.sqrt((self.w_bar - h).T @ (self.w_bar - h) + self.task_std ** 2)

    def gen_tasks(self, n_tasks=None, n_train=None, n_test=None, y_snr=None, task_std=None, **kwargs):
        if n_tasks is None:
            n_tasks = self.n_tasks
        if n_train is None:
            n_train = self.n_train
        if y_snr is None:
            y_snr = self.y_snr
        if task_std is None:
            task_std = self.task_std
        if n_test is None:
            n_test = self.n_test

        if self.y_dist is None:
            return self._task_gen_func(n_tasks, self.val_perc, self.n_dims, n_train, n_test, y_snr, task_std,
                                       w_bar=self.w_bar)
        else:
            return self._task_gen_func(n_tasks, self.val_perc, self.n_dims, n_train, n_test, y_snr, task_std,
                                       w_bar=self.w_bar, y_dist=self.y_dist)

    def __call__(self, n_tasks=None, n_train=None, n_test=None, y_snr=None, task_std=None, **kwargs):
        return self.gen_tasks(n_tasks, n_train, n_test, y_snr, task_std)


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
                                   y_dist='nonoisemargin', w_bar=4):
    w_bar = w_bar * np.ones(n_dims) if type(w_bar) == int else w_bar
    W_true = np.zeros((n_dims, n_tasks))
    X_train, Y_train = [None] * n_tasks, [None] * n_tasks
    X_val, Y_val = [None] * n_tasks, [None] * n_tasks
    X_test, Y_test = [None] * n_tasks, [None] * n_tasks

    thold = 0.5
    for i in range(n_tasks):
        center = 0

        X_n = np.random.randn(n_train + n_test, n_dims)
        X_n = center + X_n / norm(X_n, axis=1, keepdims=True)

        w = w_bar + task_std * np.random.randn(n_dims)

        # eps sampled from a zero-mean gaussian with std chosen to have snr=5
        clean_y_n = X_n @ w  # scalar product
        var_signal = np.var(clean_y_n)
        std_eps = np.sqrt(var_signal / y_snr)
        eps_n = np.random.randn(n_train + n_test) * std_eps

        if 'margin' in y_dist:
            inside_margin = np.abs(clean_y_n) < thold
            while any(inside_margin):
                inside_margin = np.abs(clean_y_n) < thold
                print('points inside margin {}'.format(np.count_nonzero(inside_margin.astype(int))))
                xtmp = np.random.randn(n_train + n_test, n_dims)
                xtmp = center + xtmp / norm(xtmp, axis=1, keepdims=True)
                clean_y_n[inside_margin] = (xtmp @ w)[inside_margin]
                X_n[inside_margin] = xtmp[inside_margin]

        if y_dist == 'logistic' or y_dist == 'logisticmargin':
            s = 1/10
            y_n_uniform = np.random.rand(*clean_y_n.shape)
            y_n = np.ones(clean_y_n.shape)
            p_y_given_x = 1/(1 + np.exp(-clean_y_n/s))
            y_n[y_n_uniform > p_y_given_x] = -1
        elif y_dist == 'sign':
            y_n = np.sign(clean_y_n + eps_n)
        elif y_dist == 'nonoise' or y_dist == 'nonoisemargin':
            y_n = np.sign(clean_y_n)

        X_train_all, X_test[i], Y_train_all, Y_test[i] = train_test_split(X_n, y_n, test_size=n_test)
        X_train[i], X_val[i], Y_train[i], Y_val[i] = train_test_split(X_train_all, Y_train_all, test_size=val_perc)

        W_true[:, i] = w.ravel() * (1/thold)

        # check hinge loss error
        print('MAE oracle', np.mean(np.maximum(0, 1-y_n*(X_n @ W_true[:, i]))))

    w_bar = w_bar * (1/thold)

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


class RealDatasetGenerator:
    def __init__(self, gen_f, seed=0, n_train_tasks=80, n_val_tasks=20, val_perc=0.5):
        np.random.seed(seed)
        self.seed = seed
        self.val_perc = val_perc

        self.data_train, self.data_val, self.data_test = gen_f(n_train_tasks, n_val_tasks, val_perc)

        self.n_train = None
        self.n_train_tasks = len(self.data_train['Y_train'])
        self.n_val_tasks = len(self.data_val['Y_train'])
        self.n_test_tasks = len(self.data_test['Y_train'])
        self.n_tasks = self.n_test_tasks + self.n_val_tasks + self.n_train_tasks
        self.n_dims = self.data_train['X_train'][0].shape[1]

    def gen_tasks(self, sel='train', n_train=None, **kwargs):
        if sel == 'train':
            data = self.data_train
        if sel == 'val':
            data = self.data_val
        if sel == 'test':
            data = self.data_test

        if n_train is not None:
            data['X_train'] = [t[0:n_train] for t in data['X_train']]
            data['Y_train'] = [t[0:n_train] for t in data['Y_train']]

        return data, None

    def __call__(self, **kwargs):
        return self.gen_tasks(**kwargs)


def threshold_for_classifcation(Y, th):
    Y_bc = np.ones_like(Y)
    Y_bc[Y < th] = -1
    return Y_bc


def computer_data_ge_reg(n_train_tasks=100, n_val_task=40, threshold=5):
    return computer_data_gen(n_train_tasks, n_val_task, cla=False)


def computer_data_gen(n_train_tasks=100, n_val_task=40, threshold=5, cla=True):

    temp = sio.loadmat('lenk_data.mat')
    train_data = temp['Traindata']  # 2880x15  last feature is output (score from 0 to 10) (144 tasks of 20 elements)
    test_data = temp['Testdata']  # 720x15 last feature is y (score from 0 to 10) (26 tasks of 20 elements)

    Y = train_data[:, 14]
    Y_test = test_data[:, 14]

    X = train_data[:, :14]
    X_test = test_data[:, :14]

    print('Y median', np.mean(Y))
    print('Y mean', np.median(Y))

    if cla:
        Y = threshold_for_classifcation(Y, threshold)
        Y_test = threshold_for_classifcation(Y_test, threshold)

    n_tasks = 180
    ne_tr = 16   # numer of elements on train set per task
    ne_test = 4  # numer of elements on test set per task

    def split_tasks(data, number_of_tasks, number_of_elements):
        return [data[i * number_of_elements:(i + 1) * number_of_elements] for i in range(number_of_tasks)]

    X = split_tasks(X, n_tasks, ne_tr)
    Y = split_tasks(Y, n_tasks, ne_tr)

    X_test = split_tasks(X_test, n_tasks, ne_test)
    Y_test = split_tasks(Y_test, n_tasks, ne_test)

    task_shuffled = np.random.permutation(n_tasks)

    task_range_tr = task_shuffled[0:n_train_tasks]
    task_range_val = task_shuffled[n_train_tasks:n_train_tasks+n_val_task]
    task_range_test = task_shuffled[n_train_tasks+n_val_task:]

    data_train = {'X_train': [], 'Y_train': [], 'X_val': [], 'Y_val': [], 'X_test': [], 'Y_test': []}
    data_val = {'X_train': [], 'Y_train': [], 'X_val': [], 'Y_val': [], 'X_test': [], 'Y_test': []}
    data_test = {'X_train': [], 'Y_train': [], 'X_val': [], 'Y_val': [], 'X_test': [], 'Y_test': []}

    def fill_with_tasks(data, task_range, data_m, labels_m, data_test_m, labels_test_m):
        for task_idx in task_range:
            example_shuffled = np.random.permutation(len(labels_m[task_idx]))

            X_train, Y_train = data_m[task_idx][example_shuffled], labels_m[task_idx][example_shuffled]
            X_test, Y_test = data_test_m[task_idx], labels_test_m[task_idx]

            Y_train = Y_train.ravel()
            X_train = X_train
            X_val = []
            Y_val = []

            data['X_train'].append(X_train)
            data['X_val'].append(X_val)
            data['X_test'].append(X_test)
            data['Y_train'].append(Y_train)
            data['Y_val'].append(Y_val)
            data['Y_test'].append(Y_test)

    # make training, validation and test tasks:
    fill_with_tasks(data_train, task_range_tr, X, Y, X_test, Y_test)
    fill_with_tasks(data_val, task_range_val, X, Y,  X_test, Y_test)
    fill_with_tasks(data_test, task_range_test, X, Y, X_test, Y_test)

    return data_train, data_val, data_test


def schools_data_gen(n_train_tasks=80, n_val_tasks=39, val_perc=0.5):

    n_tasks = 139

    task_shuffled = np.random.permutation(n_tasks)

    task_range_tr = task_shuffled[0:n_train_tasks]
    task_range_val = task_shuffled[n_train_tasks:n_train_tasks + n_val_tasks]
    task_range_test = task_shuffled[(n_train_tasks + n_val_tasks):]

    temp = sio.loadmat('schoolData.mat')

    all_data = temp['X'][0]
    all_labels = temp['Y'][0]


    #dataset downsampling to have same number of example for each class:
    min_size = 100  # minsize is 22
    for t in all_labels:
        if min_size > len(t):
            min_size = len(t)

    for i in range(len(all_labels)):
        example_shuffled = np.random.permutation(len(all_labels[i]))
        all_labels[i] = all_labels[i][example_shuffled]
        all_data[i] = all_data[i][:, example_shuffled]

    data_train = {'X_train': [], 'Y_train': [], 'X_val': [], 'Y_val': [], 'X_test': [], 'Y_test': []}
    data_val = {'X_train': [], 'Y_train': [], 'X_val': [], 'Y_val': [], 'X_test': [], 'Y_test': []}
    data_test = {'X_train': [], 'Y_train': [], 'X_val': [], 'Y_val': [], 'X_test': [], 'Y_test': []}

    def normalize_Y(Y, maximum=None, minimum=None):
        miny = minimum if minimum != None else np.min(Y)
        maxy = maximum if maximum != None else np.max(Y)
        return (Y - miny) / (maxy - miny), maxy, miny

    def normalizeX_dimitris(X, *args, **kwargs):
        return X / norm(X, axis=1, keepdims=True), None, None

    def binarize(X, maximum=None, minimum=None):
        minx = minimum if minimum is not None else np.min(X, axis=0)
        maxx = maximum if maximum is not None else np.max(X, axis=0)
        return 2*((X - minx) / ((maxx - minx) +1e-16)) -1, maxx, minx

    normalize_X = binarize

    def fill_with_tasks(data, task_range, test_perc=0.5):

        for task_idx in task_range:
            example_shuffled = np.random.permutation(len(all_labels[task_idx]))
            X, Y = all_data[task_idx].T[example_shuffled], all_labels[task_idx][example_shuffled]
            if test_perc > 0.0:
                X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=val_perc)
            else:
                X_train = X
                Y_train = Y
                X_test = []
                Y_test = []

            X_train, maxx, maxy = normalize_X(X_train)
            Y_train, maxy, miny = normalize_Y(Y_train)
            Y_train = Y_train.ravel()
            if len(X_test) > 0:
                X_test, _, _, = normalize_X(X_train, maxx, maxy)
                Y_test, _, _ = normalize_Y(Y_test, maxy, miny)
                Y_test = Y_test.ravel()

            X_val = []
            Y_val = []

            data['X_train'].append(X_train)
            data['X_val'].append(X_val)
            data['X_test'].append(X_test)
            data['Y_train'].append(Y_train)
            data['Y_val'].append(Y_val)
            data['Y_test'].append(Y_test)

    # make training, validation and test tasks:
    fill_with_tasks(data_train, task_range_tr, test_perc=0.0)
    fill_with_tasks(data_val, task_range_val, test_perc=val_perc)
    fill_with_tasks(data_test, task_range_test, test_perc=val_perc)

    return data_train, data_val, data_test


if __name__ == '__main__':
    #print(schools_data_gen())
    #print(computer_data_gen())
    classification_tasks_generator()
