import numpy as np
# import optimisation
import matplotlib.pylab as plt
import numpy.random as random
from numpy.linalg import norm
from sklearn.model_selection import train_test_split
from numpy.linalg import pinv
from numpy import identity as eye
from numpy.linalg import svd
from numpy.linalg import lstsq
from numpy.linalg import solve
from numpy.linalg import matrix_power

import scipy as sp
import scipy.io as sio
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score
from numpy.linalg import lstsq

from sklearn.preprocessing import MinMaxScaler

import os
import sys
import pickle
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import scale


def synthetic_data_gen(data_settings):
    n_tasks = data_settings['n_tasks']

    data_settings['train_perc'] = 0.5
    data_settings['val_perc'] = 0.5

    n_dims = data_settings['n_dims']
    n_points = data_settings['n_points']
    train_perc= data_settings['train_perc']
    val_perc = data_settings['val_perc']
    noise = data_settings['y_noise']

    data = {}

    task_std = data_settings['w_std']
    w_0 = 4 * np.ones(n_dims)
    w_1 = 4 * np.ones(n_dims)
    w_2 = 2 * np.ones(n_dims)

    # w_1 = random.randn(n_dims)
    # w_1 = w_1/norm(w_1)
    # w_1 = w_0 + 2*w_1

    # s1 = 20
    # temp = np.random.random(size=(n_dims, s1))
    # U1, _ = np.linalg.qr(temp)
    # U1 = np.hstack([U1, 0.000001 * np.random.randn(n_dims, n_dims - s1)])
    #
    # s2 = 20
    # temp = np.random.random(size=(n_dims, s2))
    # U2, _ = np.linalg.qr(temp)
    # U2 = np.hstack([U2, 0.000001 * np.random.randn(n_dims, n_dims - s2)])




    # import scipy
    # q2 = scipy.linalg.orth(a)
    #
    # from scipy.stats import special_ortho_group
    # x = special_ortho_group.rvs(3)

    W_true = np.zeros((n_dims, n_tasks))
    X_train, Y_train = [None] * n_tasks, [None] * n_tasks
    X_val, Y_val = [None] * n_tasks, [None] * n_tasks
    X_test, Y_test = [None] * n_tasks, [None] * n_tasks

    for task_idx in range(n_tasks):
        # generating and normalizing the data
        # features = random.randn(n_points, n_dims)
        # features = features / norm(features, axis=1, keepdims=True)

        tmp_selector = random.randint(0,2)

        # if tmp_selector == 0:
        #     temp = random.randn(n_points, n_dims)
        #     temp = temp / norm(temp, axis=1, keepdims=True)
        #     features = temp @ U1.T
        #
        #     weight_vector = w_2 + 0.3 * task_std * random.randn(n_dims)
        # elif tmp_selector == 1:
        #     temp = random.randn(n_points, n_dims)
        #     temp = temp / norm(temp, axis=1, keepdims=True)
        #     features = temp @ U2.T
        #
        #     weight_vector = w_1 + task_std * random.randn(n_dims)


        # center1 = 2
        # center2 = 0.5 * center1
        center1 = 0
        center2 = 0 * center1
        if tmp_selector == 0:
            features = random.randn(n_points, n_dims)
            features = center1 + features / norm(features, axis=1, keepdims=True)

            # weight_vector = w_2 + task_std * random.randn(n_dims)
            weight_vector = w_0 + task_std * random.randn(n_dims)
        elif tmp_selector == 1:
            features = random.randn(n_points, n_dims)
            features = center2 * features / norm(features, axis=1, keepdims=True)

            # weight_vector = w_1 + task_std * random.randn(n_dims)
            weight_vector = w_0 + task_std * random.randn(n_dims)



        # weight_vector = (1 - tmp_selector)*w_0 + tmp_selector*w_1 + task_std * random.randn(n_dims)
        # weight_vector = w_0 + task_std * random.randn(n_dims)

        # features = features / norm(features, axis=1, keepdims=True)

        clean_labels = features @ weight_vector
        # clean_labels = clean_labels / 100

        SNR = noise  # shitty variable name
        if SNR == 0:
            labels = clean_labels
        else:
            var_signal = np.var(clean_labels)
            std_noise = np.sqrt(var_signal / SNR)
            added_noise = 1 * random.randn(n_points)
            labels = clean_labels + added_noise




    # plt.imshow(W_true[:, :100])
    # plt.colorbar()
    # plt.pause(0.1)

    data['X_train'] = X_train
    data['Y_train'] = Y_train
    data['X_val'] = X_val
    data['Y_val'] = Y_val
    data['X_test'] = X_test
    data['Y_test'] = Y_test
    data['W_true'] = W_true
    # data['w_0'] = w_0
    return data


def schools_data_gen(data_settings):
    # n_tasks = data_settings['schools_train_tasks']
    # n_train_tasks = int(round(n_tasks * 0.9))
    # n_val_tasks = n_tasks - n_train_tasks
    # n_test_tasks = 139 - n_train_tasks - n_val_tasks
    # n_tasks = 139
    #
    # task_shuffled = np.random.permutation(n_tasks)
    #
    # data_settings['task_range_tr'] = task_shuffled[0:n_train_tasks]
    # data_settings['task_range_val'] = task_shuffled[n_train_tasks:n_train_tasks + n_val_tasks]
    # data_settings['task_range_test'] = task_shuffled[n_train_tasks + n_val_tasks:]
    #
    # data_settings['task_range'] = task_shuffled
    # data_settings['n_tasks'] = n_tasks

    temp = sio.loadmat('schoolData.mat')

    all_data = temp['X'][0]
    all_labels = temp['Y'][0]

    n_tasks = len(all_data)
    task_range_tr = data_settings['task_range_tr']
    task_range_val = data_settings['task_range_val']
    task_range_test = data_settings['task_range_test']


    X_train, Y_train = [None] * n_tasks, [None] * n_tasks
    X_val, Y_val = [None] * n_tasks, [None] * n_tasks
    X_test, Y_test = [None] * n_tasks, [None] * n_tasks

    # training tasks:
    max_score = 0
    for task_idx in task_range_tr:
        X_train[task_idx] = all_data[task_idx].T
        Y_train[task_idx] = all_labels[task_idx]

        # X_train[task_idx], _, Y_train[task_idx], _ = train_test_split(all_data[task_idx].T,
        #                                                               all_labels[task_idx],
        #                                                               test_size=0.5)

        X_train[task_idx] = X_train[task_idx] / norm(X_train[task_idx], axis=1, keepdims=True)

        Y_train[task_idx] = Y_train[task_idx].ravel()

        # commented out
        if max(Y_train[task_idx]) > max_score:
            max_score = max(Y_train[task_idx])


        # miny = min(min of whatever)
        # maxy = max(max of whather)
        #
        # y in [ 0 1]
        # (y - miny) / (maxy - miny)

        X_val[task_idx] = []
        Y_val[task_idx] = []

        X_test[task_idx] = []
        Y_test[task_idx] = []
        # print('training tasks | n_points: %3d' % len(Y_train[task_idx]))

    # commented out
    for task_idx in task_range_tr:
        # Y_train[task_idx] = Y_train[task_idx] / max_score
        # Y_train[task_idx] = minmax_scale(Y_train[task_idx])
        pass

    # max_score = 0
    for task_idx in task_range_val:
        X_train[task_idx], X_val[task_idx], Y_train[task_idx], Y_val[task_idx] = train_test_split(all_data[task_idx].T,
                                                                                                  all_labels[task_idx],
                                                                                                  test_size=0.5)

        Y_train[task_idx] = Y_train[task_idx].ravel()
        Y_val[task_idx] = Y_val[task_idx].ravel()

        # commented out
        # if max(Y_train[task_idx]) > max_score:
        #     max_score = max(Y_train[task_idx])

        X_train[task_idx] = X_train[task_idx] / norm(X_train[task_idx], axis=1, keepdims=True)
        X_val[task_idx] = X_val[task_idx] / norm(X_val[task_idx], axis=1, keepdims=True)

        X_test[task_idx] = []
        Y_test[task_idx] = []
        # print('validation tasks | n_points training: %3d | n_points validation: %3d' % (len(Y_train[task_idx]), len(Y_val[task_idx])))

    # commented out
    for task_idx in task_range_val:
        # Y_train[task_idx] = Y_train[task_idx] / max_score
        # Y_val[task_idx] = Y_val[task_idx] / max_score

        # Y_train[task_idx] = minmax_scale(Y_train[task_idx])
        # Y_val[task_idx] = minmax_scale(Y_val[task_idx])
        pass

    # max_score = 0
    for task_idx in task_range_test:
        X_train_all, X_test[task_idx], Y_train_all, Y_test[task_idx] = train_test_split(all_data[task_idx].T,
                                                                                        all_labels[task_idx],
                                                                                        test_size=0.25)
        X_train[task_idx], X_val[task_idx], Y_train[task_idx], Y_val[task_idx] = train_test_split(X_train_all,
                                                                                                  Y_train_all,
                                                                                                  test_size=0.5)
        Y_train[task_idx] = Y_train[task_idx].ravel()
        Y_val[task_idx] = Y_val[task_idx].ravel()
        Y_test[task_idx] = Y_test[task_idx].ravel()

        # commented out
        # if max(Y_train[task_idx]) > max_score:
        #     max_score = max(Y_train[task_idx])

        X_train[task_idx] = X_train[task_idx] / norm(X_train[task_idx], axis=1, keepdims=True)
        X_val[task_idx] = X_val[task_idx] / norm(X_val[task_idx], axis=1, keepdims=True)
        X_test[task_idx] = X_test[task_idx] / norm(X_test[task_idx], axis=1, keepdims=True)

        # print('test tasks | n_points training: %3d | n_points validation: %3d | n_points test: %3d' % (
        # len(Y_train[task_idx]), len(Y_val[task_idx]), len(Y_test[task_idx])))

    # commented out
    for task_idx in task_range_test:
        # Y_train[task_idx] = Y_train[task_idx] / max_score
        # Y_val[task_idx] = Y_val[task_idx] / max_score
        # Y_test[task_idx] = Y_test[task_idx] / max_score

        # Y_train[task_idx] = minmax_scale(Y_train[task_idx])
        # Y_val[task_idx] = minmax_scale(Y_val[task_idx])
        # Y_test[task_idx] = minmax_scale(Y_test[task_idx])
        pass

    data_settings['n_dims'] = X_train[0].shape[1]

    data = {}
    data['X_train'] = X_train
    data['Y_train'] = Y_train
    data['X_val'] = X_val
    data['Y_val'] = Y_val
    data['X_test'] = X_test
    data['Y_test'] = Y_test
    return data, data_settings


def mean_squared_error(data_settings, X, true, W, W_true, task_indeces):
    # n_tasks = len(task_indeces)
    # mse = 0
    # for _, task_idx in enumerate(task_indeces):
    #     n_points = len(true[task_idx])
    #     pred = W[:, task_idx]
    #
    #     mse = mse + norm(W_true[:, task_idx].ravel() - pred) ** 2
    #
    # performance = mse / n_tasks

    if data_settings['dataset'] == 'synthetic_regression':
        n_tasks = len(task_indeces)
        mse = 0
        mse_spam = []
        for _, task_idx in enumerate(task_indeces):
            n_points = len(true[task_idx])
            pred = X[task_idx] @ W[:, task_idx]

            if task_idx == 49:
                k = 1
            mse_temp = norm(true[task_idx].ravel() - pred) ** 2 / n_points
            # mse_spam.append(mse_temp)
            mse = mse +  mse_temp

            from sklearn.metrics import r2_score

            # mse = mse + r2_score(true[task_idx].ravel(), pred)

        performance = mse / n_tasks
        mse = mse / n_tasks
    elif data_settings['dataset'] == 'schools':
        n_tasks = len(task_indeces)
        explained_variance = 0
        for _, task_idx in enumerate(task_indeces):
            n_points = len(true[task_idx])
            pred = X[task_idx] @ W[:, task_idx]

            # mse = norm(true[task_idx].ravel() - pred)**2 / n_points
            # explained_variance = explained_variance +  (1 - mse/np.var(true[task_idx]))
            try:
                explained_variance = explained_variance + explained_variance_score(true[task_idx].ravel(), pred)
            except:
                k=1

        performance = 100 * explained_variance / n_tasks
    return performance


def solve_wrt_w_online(h, X, Y, n_tasks, data, W_pred, param1, task_range):
    for _, task_idx in enumerate(task_range):
        n_points = len(Y[task_idx])

        curr_w_pred = (X[task_idx].T @ pinv(X[task_idx] @ X[task_idx].T + param1 * n_points * eye(n_points)) @ \
                      (X[task_idx] @ h - Y[task_idx]) + h).ravel()
        W_pred[:, task_idx] = curr_w_pred
    return W_pred


def solve_wrt_w(h, X, Y, n_tasks, data, W_pred, param1, task_range):
    for _, task_idx in enumerate(task_range):
        n_points = len(Y[task_idx])
        n_dims = X[task_idx].shape[1]

        curr_w_pred = (pinv(X[task_idx].T @ X[task_idx] + param1 * n_points * eye(n_dims)) @
                       (n_points * param1 * h + X[task_idx].T @ Y[task_idx])).ravel()

        W_pred[:, task_idx] = curr_w_pred
    return W_pred


def solve_wrt_w_cooked(h, XtXinv, XtY, X, W_pred, param1, task_range):
    for _, task_idx in enumerate(task_range):
        n_points = X[task_idx].shape[0]

        curr_w_pred = (XtXinv[task_idx] @ (n_points * param1 * h + XtY[task_idx])).ravel()

        # curr_w_pred = (pinv(X[task_idx].T @ X[task_idx] + param1 * n_points * eye(n_dims)) @
        #                (n_points * param1 * h + X[task_idx].T @ Y[task_idx])).ravel()

        W_pred[:, task_idx] = curr_w_pred
    return W_pred


def solve_wrt_h_stochastic_v1(h, training_settings, data, X_train, Y_train, n_points, task_range, param1, c_iter):
    obj = lambda h: sum([param1**2 * n_points[i] * norm(pinv(X_train[i] @ X_train[i].T + param1 * n_points[i] * eye(n_points[i]))
                                                     @ (X_train[i] @ h - Y_train[i])) ** 2 for i in task_range])
    var_grad = lambda h: sum([2 * param1**2 * n_points[i] * X_train[i].T @ matrix_power(pinv(X_train[i] @ X_train[i].T +
                            param1 * n_points[i] * eye(n_points[i])), 2) @ ((X_train[i] @ h).ravel() - Y_train[i]) for i in task_range])

    alpha_value = training_settings['alpha_value']

    curr_obj = obj(h)

    objectives = []

    n_points_for_step = np.array(n_points).astype('float')
    n_points_for_step[n_points_for_step == 0] = np.nan
    n_iter = 10

    curr_tol = 10 ** 10
    conv_tol = 10 ** -6
    inner_iter = 0

    t = time.time()
    while (inner_iter < n_iter) and (curr_tol > conv_tol):
        inner_iter = inner_iter + 1
        prev_h = h
        prev_obj = curr_obj

        c_iter = c_iter + inner_iter
        # step_size = np.sqrt(2) * alpha_value / ((alpha_value + 1) * np.sqrt(c_iter))
        step_size = alpha_value / (2*np.sqrt(2)*(alpha_value + 1)*np.sqrt(c_iter))
        h = prev_h - step_size * var_grad(prev_h)

        curr_obj = obj(h)
        objectives.append(curr_obj)

        curr_tol = abs(curr_obj - prev_obj) / prev_obj

        if curr_obj > 1.001 * prev_obj:
            k=1

        if (time.time() - t > 5):
            t = time.time()
            print("iter: %5d | obj: %20.15f | tol: %20.18f" % (inner_iter, curr_obj, curr_tol))

        if norm(h) > alpha_value:
            h = alpha_value * h / norm(h)

        # TODO return mean of h and D (from the UAI paper)

    return h, c_iter


def solve_wrt_h_stochastic_v2(h, training_settings, data, X_tr, Y_tr, X_ts, Y_ts, n_points_tr, n_points_ts, task_range, param1, c_iter):
    obj = lambda h: sum([1/n_points_ts[i] * norm(X_ts[i] @ pinv(X_tr[i].T @ X_tr[i] + n_points_tr[i] * param1 * eye(X_tr[i].shape[1])) @ (n_points_tr[i] * param1 * h + X_tr[i].T @ Y_tr[i]) - Y_ts[i]) ** 2 for i in task_range])

    def var_grad(h, X_tr, Y_tr, X_ts, Y_ts, n_points_tr, n_points_ts, task_range, param1):
        for _, i in enumerate(task_range):
            n_tr = n_points_tr[i]
            n_ts = n_points_ts[i]
            A_hat = n_tr * param1 * X_ts[i] @ pinv(X_tr[i].T @ X_tr[i] + n_tr * param1 * eye(X_tr[i].shape[1]))
            b_hat = Y_ts[i] - X_ts[i] @ pinv(X_tr[i].T @ X_tr[i] + n_tr * param1 * eye(X_tr[i].shape[1])) @ X_tr[i].T @ Y_tr[i]

            grad = 2/n_ts * A_hat.T @ (A_hat @ h - b_hat)
        return grad

    alpha_value = training_settings['alpha_value']

    curr_obj = obj(h)

    objectives = []

    n_iter = 1

    curr_tol = 10 ** 10
    conv_tol = 10 ** -6
    inner_iter = 0

    t = time.time()
    while (inner_iter < n_iter) and (curr_tol > conv_tol):
        inner_iter = inner_iter + 1
        prev_h = h
        prev_obj = curr_obj

        c_iter = c_iter + inner_iter
        step_size = alpha_value / (2*np.sqrt(2)*(alpha_value + 1 +  1/param1)*np.sqrt(c_iter))
        h = prev_h - step_size * var_grad(prev_h, X_tr, Y_tr, X_ts, Y_ts, n_points_tr, n_points_ts, task_range, param1)

        curr_obj = obj(h)
        objectives.append(curr_obj)

        curr_tol = abs(curr_obj - prev_obj) / prev_obj

        if curr_obj > 1.001 * prev_obj:
            k=1

        if (time.time() - t > 5):
            t = time.time()
            print("iter: %5d | obj: %20.15f | tol: %20.18f" % (inner_iter, curr_obj, curr_tol))

        if norm(h) > alpha_value:
            h = alpha_value * h / norm(h)

    return h, c_iter


def solve_wrt_h_stochastic_generic(h, obj, grad, step_gamma, c_iter):
    curr_obj = obj(h)

    objectives = []

    n_iter = 1

    curr_tol = 10 ** 10
    conv_tol = 10 ** -6
    inner_iter = 0

    t = time.time()
    while (inner_iter < n_iter) and (curr_tol > conv_tol):
        inner_iter = inner_iter + 1
        prev_h = h
        prev_obj = curr_obj

        c_iter = c_iter + inner_iter
        # step_size = step_gamma / c_iter
        step_size = step_gamma / 1
        h = prev_h - step_size * grad(prev_h)

        curr_obj = obj(h)
        objectives.append(curr_obj)

        curr_tol = abs(curr_obj - prev_obj) / prev_obj

        if curr_obj > 1.001 * prev_obj:
            k=1

        if (time.time() - t > 5):
            t = time.time()
            print("iter: %5d | obj: %20.15f | tol: %20.18f" % (inner_iter, curr_obj, curr_tol))

    return h, c_iter, curr_obj


def save_results(results, data_settings, training_settings):
    foldername = training_settings['foldername']
    filename = training_settings['filename']
    param1_range = training_settings['param1_range']

    if not os.path.exists(foldername):
        os.makedirs(foldername)
    f = open(foldername + '/' + filename + ".pckl", 'wb')
    pickle.dump(results, f)
    pickle.dump(data_settings, f)
    pickle.dump(training_settings, f)
    f.close()
