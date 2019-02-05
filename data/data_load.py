import numpy as np
from numpy.linalg import norm
from scipy import io as sio
from sklearn.model_selection import train_test_split


class RealDatasetGenerator:
    def __init__(self, gen_f, seed=0, n_train_tasks=80, n_val_tasks=20, val_perc=0.5, allow_repick=True):
        np.random.seed(seed)
        self.seed = seed
        self.val_perc = val_perc
        self.allow_repick = allow_repick

        ntt = n_train_tasks if n_train_tasks + n_val_tasks <= 100 else 100 - n_val_tasks

        self.data_train, self.data_val, self.data_test = gen_f(ntt, n_val_tasks)

        self.n_train = None
        self.n_train_tasks = len(self.data_train['Y_train'])
        self.n_val_tasks = len(self.data_val['Y_train'])
        self.n_test_tasks = len(self.data_test['Y_train'])
        self.n_tasks = self.n_test_tasks + self.n_val_tasks + self.n_train_tasks
        self.n_dims = self.data_train['X_train'][0].shape[1]

        self.random_indices = np.random.randint(0, ntt, size=100000)

    def gen_tasks(self, sel='train', n_train=None, n_tasks=None, **kwargs):
        if sel == 'train':
            data = self.data_train
            if self.allow_repick and n_tasks is not None:
                for k in data.keys():
                    data[k] = [data[k][j] for j in self.random_indices[:n_tasks]]
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

    temp = sio.loadmat('data/lenk_data.mat')
    train_data = temp['Traindata']  # 2880x15  last feature is output (score from 0 to 10) (144 tasks of 20 elements)
    test_data = temp['Testdata']  # 720x15 last feature is y (score from 0 to 10) (26 tasks of 20 elements)

    Y = train_data[:, 14]
    Y_test = test_data[:, 14]

    X = train_data[:, :14]
    X_test = test_data[:, :14]

    print('Y median', np.mean(Y))
    print('Y mean', np.median(Y))

    # for i in range(len(X_test) -4):
    #    print(np.max(np.abs(test_data[i, :14] - test_data[i+4, :14])))

    if cla:
        Y = threshold_for_classifcation(Y, threshold)
        Y_test = threshold_for_classifcation(Y_test, threshold)

    n_tasks = 180
    ne_tr = 16   # numer of elements on train set per task
    ne_test = 4  # numer of elements on test set per task

    def split_tasks(data, nt, number_of_elements):
        return [data[i * number_of_elements:(i + 1) * number_of_elements] for i in range(nt)]

    data_m = split_tasks(X, n_tasks, ne_tr)
    labels_m = split_tasks(Y, n_tasks, ne_tr)

    data_test_m = split_tasks(X_test, n_tasks, ne_test)
    labels_test_m = split_tasks(Y_test, n_tasks, ne_test)

    task_shuffled = np.random.permutation(n_tasks)

    task_range_tr = task_shuffled[0:n_train_tasks]
    task_range_val = task_shuffled[n_train_tasks:n_train_tasks+n_val_task]
    task_range_test = task_shuffled[n_train_tasks+n_val_task:]

    data_train = {'X_train': [], 'Y_train': [], 'X_val': [], 'Y_val': [], 'X_test': [], 'Y_test': []}
    data_val = {'X_train': [], 'Y_train': [], 'X_val': [], 'Y_val': [], 'X_test': [], 'Y_test': []}
    data_test = {'X_train': [], 'Y_train': [], 'X_val': [], 'Y_val': [], 'X_test': [], 'Y_test': []}

    def fill_with_tasks(data, task_range, data_m, labels_m, data_test_m, labels_test_m):
        for task_idx in task_range:
            es = np.random.permutation(len(labels_m[task_idx]))

            X_train, Y_train = data_m[task_idx][es], labels_m[task_idx][es]
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
    fill_with_tasks(data_train, task_range_tr, data_m, labels_m, data_test_m, labels_test_m)
    fill_with_tasks(data_val, task_range_val, data_m, labels_m,  data_test_m, labels_test_m)
    fill_with_tasks(data_test, task_range_test, data_m, labels_m, data_test_m, labels_test_m)

    return data_train, data_val, data_test


def schools_data_gen(n_train_tasks=80, n_val_tasks=20, val_perc=0.5, downsample=False, bias=True):

    n_tasks = 139

    task_shuffled = np.random.permutation(n_tasks)

    task_range_tr = task_shuffled[0:n_train_tasks]
    task_range_val = task_shuffled[n_train_tasks:n_train_tasks + n_val_tasks]
    task_range_test = task_shuffled[(n_train_tasks + n_val_tasks):]

    temp = sio.loadmat('data/schoolData.mat')

    if bias:
        all_data = [np.ones((t.shape[0] + 1, t.shape[1])) for t in temp['X'][0]]
        for i in range(len(all_data)):
            all_data[i][1:] = temp['X'][0][i]
    else:
        all_data = temp['X'][0]

    all_labels = temp['Y'][0]

    # dataset downsampling to have same number of example for each class:
    min_size = 100  # minsize is 22
    for t in all_labels:
        if min_size > len(t):
            min_size = len(t)

    for i in range(len(all_labels)):
        example_shuffled = np.random.permutation(len(all_labels[i]))
        all_labels[i] = all_labels[i][example_shuffled]
        all_data[i] = all_data[i][:, example_shuffled]

        if downsample:
            all_labels[i] = all_labels[i][:min_size]
            all_data[i] = all_data[i][:min_size]

    data_train = {'X_train': [], 'Y_train': [], 'X_val': [], 'Y_val': [], 'X_test': [], 'Y_test': []}
    data_val = {'X_train': [], 'Y_train': [], 'X_val': [], 'Y_val': [], 'X_test': [], 'Y_test': []}
    data_test = {'X_train': [], 'Y_train': [], 'X_val': [], 'Y_val': [], 'X_test': [], 'Y_test': []}

    def normalize_Y_new(Y, maximum=None, minimum=None):
        miny = minimum if minimum != None else np.min(Y)
        maxy = maximum if maximum != None else np.max(Y)
        return (Y - miny) / (maxy - miny), maxy, miny

    def do_nothing(X, *args, **kwargs):
        return X, None, None

    def normalizeX_dimitris(X, *args, **kwargs):
        if bias:
            X_sub = np.ones_like(X)
            X_sub[:, 1:] = X[:, 1:] / norm(X[:, 1:], axis=1, keepdims=True)
            return X_sub, None, None
        else:
            return X / norm(X, axis=1, keepdims=True), None, None

    def binarize(X, maximum=None, minimum=None):
        minx = minimum if minimum is not None else np.min(X, axis=0)
        maxx = maximum if maximum is not None else np.max(X, axis=0)
        return 2*((X - minx) / ((maxx - minx) +1e-16)) -1, maxx, minx

    def divide(Y, *args, **kwargs):
        return Y, None, None

    normalize_X = normalizeX_dimitris
    normalize_Y = divide

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
                X_test, _, _, = normalize_X(X_test, maxx, maxy)
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


def schools_data_gen2(n_train_tasks=80, n_val_tasks=39, val_perc=0.5):

    n_tasks = 139

    task_shuffled = np.random.permutation(n_tasks)

    task_range_tr = task_shuffled[0:n_train_tasks]
    task_range_val = task_shuffled[n_train_tasks:n_train_tasks + n_val_tasks]
    task_range_test = task_shuffled[(n_train_tasks + n_val_tasks):]

    temp1 = sio.loadmat('school2/school_b.mat')
    temp2 = sio.loadmat('school2/school_1_indexes.mat')

    0

if __name__ == '__main__':
    #print(computer_data_gen())
    print(schools_data_gen())