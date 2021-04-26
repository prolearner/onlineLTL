import numpy as np
from numpy.linalg import norm
from scipy import io as sio
from scipy.sparse import  issparse
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import pickle
import os
from sklearn.decomposition import PCA
import math


class RealDatasetGenerator:
    def __init__(self, gen_f, seed=0, n_train_tasks=80, n_val_tasks=20, val_perc=0.5, allow_repick=False):
        np.random.seed(seed)
        self.seed = seed
        self.val_perc = val_perc
        self.allow_repick = allow_repick

        ntt = n_train_tasks

        self.data_train, self.data_val, self.data_test, self.desc = gen_f(n_train_tasks=ntt, n_val_tasks=n_val_tasks)

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
            cap_lengths = [min(n_train, t.shape[0]) for t in data['Y_train']]
            data['X_train'] = [data['X_train'][t][0:cap_lengths[t]] for t in range(len(data['X_train']))]
            data['Y_train'] = [data['Y_train'][t][0:cap_lengths[t]] for t in range(len(data['Y_train']))]

        return data, None

    def __call__(self, **kwargs):
        return self.gen_tasks(**kwargs)


def threshold_for_classifcation(Y, th):
    Y_bc = np.ones_like(Y)
    Y_bc[Y < th] = -1
    return Y_bc


def computer_data_ge_reg(n_train_tasks=100, n_val_task=40, threshold=5):
    return computer_data_gen(n_train_tasks, n_val_task, cla=False)


def computer_data_gen(n_train_tasks=100, n_val_task=40, threshold=5, cla=True, balanced=False):

    temp = sio.loadmat('lenk_data.mat')
    
    # train_data and test_data will contain the train and test elements of the tasks respectively.
    train_data = temp['Traindata']  # 2880x15  last feature is output (score from 0 to 10) (180 tasks (people) of 16 elements (computers))
    test_data = temp['Testdata']  # 720x15 last feature is y (score from 0 to 10) (180 tasks (people) of 4 elements (computers))
    

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
            es = list(range(len(labels_m[task_idx])))

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

    def print_quality(data, n_train=16, name='', t_hold=1.0):
        count_train_one_class = 0
        count_test_one_class = 0
        for i, Y in enumerate(data['Y_train']):
            if np.abs(np.mean(Y[:n_train])) >= t_hold:
                count_train_one_class += 1
        for i, Y in enumerate(data['Y_test']):
            if np.abs(np.mean(Y)) >= t_hold:
                count_test_one_class += 1
        print(name+' train/test tasks(n{}) with mean(Y) >= {} = {}/{}'.format(n_train, t_hold, count_train_one_class, count_test_one_class))
        return count_train_one_class, count_test_one_class

    def print_tot(n_train=16, t_hold=1.0):
        ctr_train, cts_train = print_quality(data_train, n_train=n_train, t_hold=t_hold, name='train')
        ctr_val, cts_val = print_quality(data_val, n_train=n_train, t_hold=t_hold, name='val')
        ctr_test, cts_test = print_quality(data_test, n_train=n_train, t_hold=t_hold,  name='test')

        ctr_tot = ctr_test + ctr_train + ctr_val
        cts_tot = cts_test + cts_train + cts_val
        print('total train/test tasks(n{}) with mean(Y) >= {} = {}/{}'.format(n_train, t_hold, ctr_tot, cts_tot))

    print_tot(8, 0.5)
    print_tot(8, 1)

    return data_train, data_val, data_test, None


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

    return data_train, data_val, data_test, None


#### MULAN DATASETS


def load_mulan_data(name, path, n_labels=983, sparse=False, features='nominal', test_set=True, test_size=0.5,
                    label_location='end', min_y_cardinality=0):
    from skmultilearn.dataset import load_from_arff

    if features == 'nominal':
        input_feature_type = 'int'
    elif features == 'numeric':
        input_feature_type = 'double'

    def inner_load(path):
        return load_from_arff(path,
                              # number of labels
                              label_count=n_labels,
                              # MULAN format, labels at the end of rows in arff data
                              label_location=label_location,
                              # bag of words
                              input_feature_type=input_feature_type, encode_nominal=False,
                              # sometimes the sparse ARFF loader is borked, like in delicious,
                              # scikit-multilearn converts the loaded data to sparse representations,
                              # so disabling the liac-arff sparse loader
                              load_sparse=sparse,
                              # this decides whether to return attribute names or not, usually
                              # you don't need this
                              return_attribute_definitions=False)

    if test_set:
        X_train, y_train = inner_load(os.path.join(path, name+"-train.arff"))
        X_test, y_test = inner_load(os.path.join(path, name+"-test.arff"))
    else:
        X, Y = inner_load(os.path.join(path, name+".arff"))

        # remove examples with low number of labels
        print('y_shape:', Y.shape)
        print('y_cardinality :', np.sum(Y, axis=1))

        if min_y_cardinality > 0:
            X = X[np.sum(Y, axis=1) >= min_y_cardinality]
            Y = Y[np.sum(Y, axis=1) >= min_y_cardinality]

        print('X shape :', X.shape)

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size)

    data = {'X_train': X_train, 'y_train': y_train, 'X_test': X_test, 'y_test': y_test}

    if issparse(data['X_train']):
        return data['X_train'].toarray(), data['y_train'].toarray(), data['X_test'].toarray(), data['y_test'].toarray()
    else:
        return data['X_train'], data['y_train'], data['X_test'], data['y_test']


mulan_dict = {'Corel5k':   {'n_labels': 374, 'features': 'nominal', 'test_set': True,  'test_size': 0.5},
              'CAL500':    {'n_labels': 174, 'features': 'numeric', 'test_set': False, 'test_size': 0.5},
              'delicious': {'n_labels': 983, 'features': 'nominal', 'test_set': True,  'test_size': 0.5},
              'bookmarks': {'n_labels': 208, 'features': 'nominal', 'test_set': False, 'test_size': 0.5},
              'bibtex':    {'n_labels': 159, 'features': 'nominal', 'test_set': True,  'test_size': 0.5}
              }

mulan_settings = {'Corel5k': {'balanced':True, 'add_bias':True, 'multi_transform': 'onevsmaxn_rand',
                              'normalization': 'meanstd', 'pca_comp': None},
                  'CAL500': {'balanced':True, 'add_bias':True, 'multi_transform': 'onevsmaxn_rand',
                             'normalization': 'l2', 'pca_comp': None},
                  'delicious': {'balanced':True, 'add_bias':True, 'multi_transform': 'onevsmaxn_rand',
                                'normalization': 'l2', 'pca_comp': None},
                  'bookmarks': {'balanced':True, 'add_bias':True, 'multi_transform': 'onevsmaxn_rand',
                                'normalization': 'meanstd', 'pca_comp': None},
                  'bibtex': {'balanced': True, 'add_bias': True, 'multi_transform': 'onevsmaxn_rand',
                                'normalization': 'none', 'pca_comp': None, 'perc': 1}
                  }


def get_mulan_loader(data_name, **kwargs):
    def mulan_loader(**kwargs):
        return mulan(data_name=data_name, **mulan_dict[data_name], **mulan_settings[data_name], **kwargs)
    return mulan_loader


def mulan(data_name, n_labels=374, features='nominal', test_set=True, test_size=0.5,
          n_train_tasks=200, n_val_tasks=73, parent_path="data", balanced=True, add_bias=True,
          multi_transform='onevsmaxn_rand', normalization='meanstd', pca_comp=None, min_y_cardinality=0,
          min_n_per_task=1, perc=1, data_analisys=False):

    desc = 'ts'+str(test_size)+'bal'+str(balanced)+'bias'+str(add_bias)+'mt'+multi_transform+'norm' \
            +normalization+'pca_c'+str(pca_comp)+'miny'+str(min_y_cardinality)+'mnpt'+str(min_n_per_task) \
            +'perc'+str(perc)

    path = os.path.join(parent_path, data_name)
    dataset_file_path = os.path.join(parent_path, data_name, data_name + desc + "_multitask.p")

    if os.path.isfile(dataset_file_path):
        print('load already made meta-dataset')
        dataset_dict = pickle.load(open(dataset_file_path, "rb"))
        train, test = dataset_dict['train'], dataset_dict['test']
        data_m, labels_m , data_test_m, labels_test_m = train['x'], train['y'], test['x'], test['y']

    else:
        print('create meta-dataset from scratch')
        X_train, y_train, X_test, y_test = load_mulan_data(data_name, path, n_labels=n_labels,
                                                           sparse=False, features=features,
                                                           test_set=test_set, test_size=test_size,
                                                           min_y_cardinality=min_y_cardinality)

        # data normalization:
        if normalization == 'meanstd':
            scaler = preprocessing.StandardScaler()
        elif normalization == 'to[-1,1]':
            scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
        elif normalization == 'l2':
            scaler = preprocessing.Normalizer()
        elif normalization == 'none':
            scaler = None
        else:
            raise ValueError('normalization :', normalization, ' not defined!')

        if scaler:
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        if pca_comp:
            pca = PCA(n_components=pca_comp)
            X_train = pca.fit_transform(X_train)
            X_test = pca.transform(X_test)

            print('explained variance', pca.explained_variance_ratio_)

        # task with most examples has index 112
        n_tasks = y_train.shape[1]
        most_task = np.argmax(np.array([np.sum(y_train[:, i] > 0) for i in range(n_tasks)]))
        most_task_n = np.max(np.array([np.sum(y_train[:, i] > 0) for i in range(n_tasks)]))
        print('the task {} has the max number of examples: {}'.format(most_task, most_task_n))

        def extract_task(i, X, y, bal=balanced, add_b=add_bias, is_train=True):
            Xi_pos = X[y[:, i] > 0]
            yi_pos = np.ones(Xi_pos.shape[0])

            if 'onevsmaxn' in multi_transform:
                if most_task == i:
                    return None, None
                else:
                    Xi_neg = X[np.logical_and(y[:, most_task] > 0, y[:, i] < 1)]
            elif 'onevsall' in multi_transform:
                Xi_neg = X[y[:, i] < 1]  # one against all the other lables
            elif 'onevsrandom' in multi_transform:
                # one against a random label
                ol = np.random.randint(0, n_tasks)
                while ol == i:
                   ol = np.random.randint(0, n_tasks)
                Xi_neg = X[y[:, ol] > 0]
            else:
                raise ValueError('multitransform', multi_transform, 'not defined!')

            yi_neg = - np.ones(Xi_neg.shape[0])
            pos_len, neg_len = Xi_pos.shape[0], Xi_neg.shape[0]

            if '_rand' in multi_transform:
                perm = np.random.permutation(max(neg_len, pos_len))
            else:
                perm = np.array(list(range(max(neg_len, pos_len))))

            # reduce test set to make less duplicates:
            n_ret = min(neg_len, pos_len) if is_train else round(min(neg_len, pos_len)*perc)

            if bal:
                if pos_len < neg_len:
                    Xi_neg = Xi_neg[perm][:n_ret]
                    yi_neg = yi_neg[perm][:n_ret]
                else:
                    perm = np.random.permutation(pos_len)
                    Xi_pos = Xi_pos[perm][:n_ret]
                    yi_pos = yi_pos[perm][:n_ret]

            perm_idx = np.random.permutation(len(yi_neg) + len(yi_pos))
            Xi = np.concatenate((Xi_pos, Xi_neg), axis=0)[perm_idx]
            yi = np.concatenate((yi_pos, yi_neg), axis=0)[perm_idx]

            if add_b:
                n_xi = np.ones((Xi.shape[0], Xi.shape[1] + 1))
                n_xi[:, 1:] = Xi
                Xi = n_xi

            return Xi, yi

        # Create the tasks: from multi-labeling to multi-task
        data_m, labels_m, labels_test_m, data_test_m = [], [], [], []
        min_size = 1000000
        max_size = 0
        for i in range(n_tasks):
            Xi_train, yi_train = extract_task(i, X_train, y_train)
            Xi_test, yi_test = extract_task(i, X_test, y_test, is_train=False)

            if Xi_train is not None and len(yi_train) >= min_n_per_task and len(yi_test) != 0:
                data_m.append(Xi_train)
                labels_m.append(yi_train)

                data_test_m.append(Xi_test)
                labels_test_m.append(yi_test)
                if len(yi_train) < min_size:
                    min_size = len(yi_train)
                if len(yi_train) > max_size:
                    max_size = len(yi_train)

                # task statistics:
                print('task {}, n={}, n_test={},'
                      ' balance=tr: {:.3f}, ts: {:.3f}'.format(i, len(yi_train), len(yi_test), np.mean(yi_train),
                                                               np.mean(yi_test)))
        print('tasks min/max size', min_size, max_size)
        dataset_dict = {'train': {'x': data_m, 'y':labels_m}, 'test': {'x': data_test_m, 'y':labels_test_m}}
        pickle.dump(dataset_dict, open(dataset_file_path, "wb"))

    n_tasks = len(labels_m)
    print('Number of tasks:', len(labels_m), len(labels_test_m))

    if data_analisys:
        print('X_train values')

        nd_count=0
        pd_count=0
        example_count = 0
        unique_count = 0
        unique = []
        overlap = 0
        for i in range(n_tasks):
            print('task', i, 'mean y train, test, n examples train, test ',
                  np.mean(labels_m[i]), np.mean(labels_test_m[i]), len(labels_m[i]), len(labels_test_m[i]))

            # negative duplicates:

            for e, y in zip(data_test_m[i], labels_test_m[i]):
                is_present = False
                example_count+=1
                for ec, yc in unique:
                    if np.all(e == ec) and y == yc:
                        is_present=True
                        if y > 0:
                            pd_count+=1
                        else:
                            nd_count+=1
                if not is_present:
                    unique_count+=1
                    unique.append([e, y])

            overlap = unique_count/example_count
            print('nd, pd, ec, uc, rt', nd_count, pd_count, example_count, unique_count, overlap)
        desc = desc+'ov'+str(overlap)

        for i in range(len(data_m[0][0])):
            ith_feat = np.concatenate([data_m[t][:, i] for t in range(n_tasks)], axis=0)
            print('feature {}: mean {:.3f}, std {:.3f}, max {:.3f},'
                  ' min {:.3f}'.format(i, np.mean(ith_feat), np.std(ith_feat), np.max(ith_feat), np.min(ith_feat)))

    task_shuffled = np.random.permutation(n_tasks)
    task_range_tr = task_shuffled[0:n_train_tasks]
    task_range_val = task_shuffled[n_train_tasks:n_train_tasks+n_val_tasks]
    task_range_test = task_shuffled[n_train_tasks+n_val_tasks:]

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

    return data_train, data_val, data_test, desc


if __name__ == '__main__':
    data_name = 'bibtex'
    computer_data_ge_reg()
    #get_mulan_loader(data_name)(parent_path='')
