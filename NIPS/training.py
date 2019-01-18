from NIPS.utilities import *

from numpy.linalg import pinv
from numpy import identity as eye
import time

from NIPS.batch_bar_LTL import batch_bar_LTL
from NIPS.online_with_split_no_constraint import online_with_split_no_constraint


def training(data, data_settings, training_settings):
    method = training_settings['method']

    if method == 'online_var_LTL_v1':
        online_var_LTL_v1(data, data_settings, training_settings)
    elif method == 'online_var_LTL_v2':
        online_var_LTL_v2(data, data_settings, training_settings)
    elif method == 'var_ITL':
        var_ITL(data, data_settings, training_settings)
    elif method == 'var_ITL_oracle':
        var_ITL_oracle(data, data_settings, training_settings)
    elif method == 'batch_bar_LTL':
        batch_bar_LTL(data, data_settings, training_settings)
    elif method == 'online_with_split_no_constraint':
        online_with_split_no_constraint(data, data_settings, training_settings)

    print('done')
    return


def online_var_LTL_v1(data, data_settings, training_settings):
    foldername = training_settings['foldername']
    filename = training_settings['filename']
    param1_range = training_settings['param1_range']
    n_tasks = data_settings['n_tasks']
    n_dims = data_settings['n_dims']
    task_range = data_settings['task_range']
    task_range_tr = data_settings['task_range_tr']
    task_range_val = data_settings['task_range_val']
    task_range_test = data_settings['task_range_test']
    W_true = data['W_true']

    T = len(task_range_tr)

    time_lapsed = [None] * T

    best_h = random.randn(n_dims)

    if data_settings['dataset'] == 'synthetic_regression':
        best_val_performance = 10 ** 8
    else:
        best_val_performance = -10 ** 8


    validation_curve = [[] for i in range(len(param1_range))]
    for param1_idx, param1 in enumerate(param1_range):
        all_train_perf, all_val_perf, all_test_perf = [[] for i in range(T)], [[] for i in range(T)], [[] for i in range(len(param1_range))]

        W_pred = np.zeros((n_dims, n_tasks))
        all_h = [None] * T

        h = best_h

        c_iter = 0
        t = time.time()
        for pure_task_idx, curr_task_range_tr in enumerate(task_range_tr):
            #####################################################
            # OPTIMISATION
            X_train, Y_train = [None] * n_tasks, [None] * n_tasks
            n_points = [0] * n_tasks
            for _, task_idx in enumerate([curr_task_range_tr]):
                X_train[task_idx] = data['X_train'][task_idx]
                Y_train[task_idx] = data['Y_train'][task_idx]
                n_points[task_idx] = len(Y_train[task_idx])

            h, c_iter = solve_wrt_h_stochastic_v1(h, training_settings, data, X_train, Y_train, n_points,
                                                  [curr_task_range_tr], param1, c_iter)

            # print(h)
            all_h[pure_task_idx] = h


            time_lapsed[pure_task_idx] = time.time() - t

        # h = np.average(all_h[:pure_task_idx], axis=0)

            #####################################################
            # TEST
            X_train, Y_train = [None] * n_tasks, [None] * n_tasks
            for _, task_idx in enumerate(task_range_test):
                X_train[task_idx] = np.concatenate((data['X_val'][task_idx], data['X_train'][task_idx]))
                Y_train[task_idx] = np.concatenate((data['Y_val'][task_idx], data['Y_train'][task_idx]))

            W_pred = solve_wrt_w(h, X_train, Y_train, n_tasks, data, W_pred, param1, task_range_test)

            test_perf = mean_squared_error(data_settings, data['X_test'], data['Y_test'], W_pred, W_true, task_range_test)
            all_test_perf[param1_idx].append(test_perf)

        # W_pred = solve_wrt_w(h, data['X_train'], data['Y_train'], n_tasks, data, W_pred, param1, task_range_tr)
        # train_perf = mean_squared_error(data_settings, data['X_train'], data['Y_train'], W_pred, task_range_tr)
        # all_train_perf[pure_task_idx].append(train_perf)
        all_train_perf = [None]

        #####################################################
        # VALIDATION
        W_pred = solve_wrt_w(h, data['X_train'], data['Y_train'], n_tasks, data, W_pred, param1, task_range_val)

        val_perf = mean_squared_error(data_settings, data['X_val'], data['Y_val'], W_pred, W_true, task_range_val)
        all_val_perf[pure_task_idx].append(val_perf)

        print('lambda: %6e | val MSE: %7.5f | test MSE: %7.5f | norm D: %4.2f' %
              (param1, val_perf, test_perf, norm(best_h)))

        validation_curve[param1_idx] = val_perf


            # print(W_pred[:, 501])
            # print(data['W_true'][:, 501])

        best_h = np.average(all_h[:pure_task_idx], axis=0)

        if data_settings['dataset'] == 'synthetic_regression':
            if val_perf < best_val_performance:
                validation_criterion = True
            else:
                validation_criterion = False
        else:
            if val_perf > best_val_performance:
                validation_criterion = True
            else:
                validation_criterion = False

        if validation_criterion == True:
            best_val_performance = val_perf

            best_param1 = param1
            best_param1_idx = param1_idx
            # print(best_h)


            best_test = test_perf
            best_train_perf = all_train_perf
            best_val_perf = all_val_perf
            best_test_perf = all_test_perf[param1_idx]


    # plt.figure()
    # plt.plot(best_test_perf)
    # plt.pause(0.01)

    plt.figure()
    plt.plot(np.log10(validation_curve))
    plt.pause(0.01)

    results = {}
    results['best_param'] = best_param1
    # results['D'] = best_D
    results['best_train_perf'] = best_train_perf
    results['best_val_perf'] = best_val_perf
    results['best_test_perf'] = best_test_perf
    results['time_lapsed'] = time_lapsed

    save_results(results, data_settings, training_settings, filename, foldername)


def online_var_LTL_v2(data, data_settings, training_settings):
    foldername = training_settings['foldername']
    filename = training_settings['filename']
    param1_range = training_settings['param1_range']
    n_tasks = data_settings['n_tasks']
    n_dims = data_settings['n_dims']
    task_range = data_settings['task_range']
    task_range_tr = data_settings['task_range_tr']
    task_range_val = data_settings['task_range_val']
    task_range_test = data_settings['task_range_test']
    W_true = data['W_true']

    T = len(task_range_tr)

    time_lapsed = [None] * T

    best_h = random.randn(n_dims)

    if data_settings['dataset'] == 'synthetic_regression':
        best_val_performance = 10 ** 8
    else:
        best_val_performance = -10 ** 8


    validation_curve = [[] for i in range(len(param1_range))]
    for param1_idx, param1 in enumerate(param1_range):
        all_train_perf, all_val_perf, all_test_perf = [[] for i in range(T)], [[] for i in range(T)], [[] for i in range(len(param1_range))]

        W_pred = np.zeros((n_dims, n_tasks))
        all_h = [None] * T

        h = best_h

        c_iter = 0
        t = time.time()
        for pure_task_idx, curr_task_range_tr in enumerate(task_range_tr):
            #####################################################
            # OPTIMISATION
            X_tr, Y_tr, X_ts, Y_ts = [None] * n_tasks, [None] * n_tasks, [None] * n_tasks, [None] * n_tasks
            n_points_tr = [0] * n_tasks
            n_points_ts = [0] * n_tasks
            for _, task_idx in enumerate([curr_task_range_tr]):
                n_total_tr = len(data['Y_train'][task_idx])

                n_tr = int(np.floor(0.05 * n_total_tr))
                n_ts = int(n_total_tr - n_tr)

                X_tr[task_idx] = data['X_train'][task_idx][0:n_tr]
                Y_tr[task_idx] = data['Y_train'][task_idx][0:n_tr]

                X_ts[task_idx] = data['X_train'][task_idx][n_tr:]
                Y_ts[task_idx] = data['Y_train'][task_idx][n_tr:]

                n_points_tr[task_idx] = n_tr
                n_points_ts[task_idx] = n_ts

            h, c_iter = solve_wrt_h_stochastic_v2(h, training_settings, data, X_tr, Y_tr, X_ts, Y_ts, n_points_tr, n_points_ts,
                                                  [curr_task_range_tr], param1, c_iter)

            # print(h)
            all_h[pure_task_idx] = h


            time_lapsed[pure_task_idx] = time.time() - t

        # h = np.average(all_h[:pure_task_idx], axis=0)

            #####################################################
            # TEST
            X_train, Y_train = [None] * n_tasks, [None] * n_tasks
            for _, task_idx in enumerate(task_range_test):
                X_train[task_idx] = np.concatenate((data['X_val'][task_idx], data['X_train'][task_idx]))
                Y_train[task_idx] = np.concatenate((data['Y_val'][task_idx], data['Y_train'][task_idx]))

            W_pred = solve_wrt_w(h, X_train, Y_train, n_tasks, data, W_pred, param1, task_range_test)

            test_perf = mean_squared_error(data_settings, data['X_test'], data['Y_test'], W_pred, W_true, task_range_test)
            all_test_perf[param1_idx].append(test_perf)

        # W_pred = solve_wrt_w(h, data['X_train'], data['Y_train'], n_tasks, data, W_pred, param1, task_range_tr)
        # train_perf = mean_squared_error(data_settings, data['X_train'], data['Y_train'], W_pred, task_range_tr)
        # all_train_perf[pure_task_idx].append(train_perf)
        all_train_perf = [None]

        #####################################################
        # VALIDATION
        W_pred = solve_wrt_w(h, data['X_train'], data['Y_train'], n_tasks, data, W_pred, param1, task_range_val)

        val_perf = mean_squared_error(data_settings, data['X_val'], data['Y_val'], W_pred, W_true, task_range_val)
        all_val_perf[pure_task_idx].append(val_perf)

        print('lambda: %6e | val MSE: %7.5f | test MSE: %7.5f | norm D: %4.2f' %
              (param1, val_perf, test_perf, norm(best_h)))

        validation_curve[param1_idx] = val_perf


            # print(W_pred[:, 501])
            # print(data['W_true'][:, 501])

        best_h = np.average(all_h[:pure_task_idx], axis=0)

        if data_settings['dataset'] == 'synthetic_regression':
            if val_perf < best_val_performance:
                validation_criterion = True
            else:
                validation_criterion = False
        else:
            if val_perf > best_val_performance:
                validation_criterion = True
            else:
                validation_criterion = False

        if validation_criterion == True:
            best_val_performance = val_perf

            best_param1 = param1
            best_param1_idx = param1_idx
            # print(best_h)


            best_test = test_perf
            best_train_perf = all_train_perf
            best_val_perf = all_val_perf
            best_test_perf = all_test_perf[param1_idx]


    # plt.figure()
    # plt.plot(best_test_perf)
    # plt.pause(0.01)

    plt.figure()
    plt.plot(np.log10(validation_curve))
    plt.pause(0.01)

    results = {}
    results['best_param'] = best_param1
    # results['D'] = best_D
    results['best_train_perf'] = best_train_perf
    results['best_val_perf'] = best_val_perf
    results['best_test_perf'] = best_test_perf
    results['time_lapsed'] = time_lapsed

    save_results(results, data_settings, training_settings, filename, foldername)


def var_ITL(data, data_settings, training_settings):
    foldername = training_settings['foldername']
    filename = training_settings['filename']
    param1_range = training_settings['param1_range']
    n_tasks = data_settings['n_tasks']
    n_dims = data_settings['n_dims']
    task_range = data_settings['task_range']
    task_range_tr = data_settings['task_range_tr']
    task_range_val = data_settings['task_range_val']
    task_range_test = data_settings['task_range_test']
    # W_true = data['W_true']
    W_true = np.nan

    W_pred = np.zeros((n_dims, n_tasks))
    best_W_pred = np.zeros((n_dims, n_tasks))

    for pure_task_idx, curr_task_range_test in enumerate(task_range_test):
        if data_settings['dataset'] == 'synthetic_regression':
            best_val_performance = 10 ** 8
        else:
            best_val_performance = -10 ** 8

        validation_curve = [[] for i in range(len(param1_range))]
        for param1_idx, param1 in enumerate(param1_range):
            #####################################################
            # OPTIMISATION
            X_train = data['X_train'][curr_task_range_test]
            Y_train = data['Y_train'][curr_task_range_test]
            n_points = len(Y_train)

            curr_w = pinv(X_train.T @ X_train + param1 * eye(n_dims)) @ X_train.T @ Y_train

            W_pred[:, curr_task_range_test] = curr_w

            train_perf = mean_squared_error(data_settings, data['X_train'], data['Y_train'], W_pred, W_true, [curr_task_range_test])
            #####################################################
            # VALIDATION
            val_perf = mean_squared_error(data_settings, data['X_val'], data['Y_val'], W_pred, W_true, [curr_task_range_test])

            if data_settings['dataset'] == 'synthetic_regression':
                if val_perf < best_val_performance:
                    validation_criterion = True
                else:
                    validation_criterion = False
            else:
                if val_perf > best_val_performance:
                    validation_criterion = True
                else:
                    validation_criterion = False

            if pure_task_idx == 18:
                # print(val_perf)
                pass
                if val_perf == 0.0:
                    k=1

            if validation_criterion == True:
                best_val_performance = val_perf

                best_W_pred[:, curr_task_range_test] = curr_w
                best_w = curr_w

                validation_curve[param1_idx] = val_perf

                best_param1 = param1
                best_train_perf = train_perf
                best_val_perf = val_perf
                # if curr_task_range_test == 49:
                #     print(param1, val_perf)
        k = 1


        # plt.figure()
        # plt.plot(np.log10(validation_curve))
        # plt.pause(0.01)

        print('T: %3d | lambda: %6e | val MSE: %12.5f | norm D: %4.2f' %
              (pure_task_idx, best_param1, best_val_perf, norm(best_w)))
        print("")
    #####################################################
    # TEST
    test_perf = mean_squared_error(data_settings, data['X_test'], data['Y_test'], best_W_pred, W_true, task_range_test)
    best_test_perf = test_perf


    # plt.figure()
    # plt.imshow(W_pred)
    # W_true = data['W_true']
    # plt.figure()
    # plt.imshow(W_true)
    # plt.pause(0.01)

    results = {}
    results['best_param'] = best_param1
    results['best_train_perf'] = best_train_perf
    results['best_val_perf'] = best_val_perf
    results['best_test_perf'] = best_test_perf

    save_results(results, data_settings, training_settings)
    k=1
    return


def var_ITL_oracle(data, data_settings, training_settings):
    foldername = training_settings['foldername']
    filename = training_settings['filename']
    param1_range = training_settings['param1_range']
    n_tasks = data_settings['n_tasks']
    n_dims = data_settings['n_dims']
    task_range = data_settings['task_range']
    task_range_tr = data_settings['task_range_tr']
    task_range_val = data_settings['task_range_val']
    task_range_test = data_settings['task_range_test']
    w_0 = data['w_0']
    W_true = data['W_true']


    W_pred = np.zeros((n_dims, n_tasks))
    for pure_task_idx, curr_task_range_test in enumerate(task_range_test):
        if data_settings['dataset'] == 'synthetic_regression':
            best_val_performance = 10 ** 8
        else:
            best_val_performance = -10 ** 8

        validation_curve = [[] for i in range(len(param1_range))]
        for param1_idx, param1 in enumerate(param1_range):
            #####################################################
            # OPTIMISATION
            X_train = data['X_train'][curr_task_range_test]
            Y_train = data['Y_train'][curr_task_range_test]
            n_points = len(Y_train)

            # curr_w = pinv(X_train.T @ X_train + param1 * eye(n_dims)) @ (X_train.T @ Y_train + param1 * w_0)
            curr_w = (pinv(X_train.T @ X_train + param1 * n_points * eye(n_dims)) @
                       (n_points * param1 * w_0 + X_train.T @ Y_train)).ravel()


            train_perf = mean_squared_error(data_settings, data['X_train'], data['Y_train'], W_pred, W_true, [curr_task_range_test])
            #####################################################
            # VALIDATION
            val_perf = mean_squared_error(data_settings, data['X_val'], data['Y_val'], W_pred, W_true, [curr_task_range_test])

            #####################################################
            # TEST
            test_perf = mean_squared_error(data_settings, data['X_test'], data['Y_test'], W_pred, W_true, task_range_test)

            validation_curve[param1_idx] = val_perf

            if data_settings['dataset'] == 'synthetic_regression':
                if val_perf < best_val_performance:
                    validation_criterion = True
                else:
                    validation_criterion = False
            else:
                if val_perf > best_val_performance:
                    validation_criterion = True
                else:
                    validation_criterion = False

            if validation_criterion == True:
                best_val_performance = val_perf

                W_pred[:, curr_task_range_test] = curr_w
                best_w = curr_w

                best_param1 = param1
                best_train_perf = train_perf
                best_val_perf = val_perf
                best_test_perf = test_perf

        # plt.figure()
        # plt.plot(validation_curve)
        # plt.pause(0.01)

        print('T: %3d | lambda: %6e | val MSE: %7.5f | test MSE: %7.5f | norm D: %4.2f' %
              (pure_task_idx, best_param1, best_val_perf, best_test_perf, norm(best_w)))
        print("")

    # plt.figure()
    # plt.imshow(W_pred)
    # W_true = data['W_true']
    # plt.figure()
    # plt.imshow(W_true)
    # plt.pause(0.01)

    results = {}
    results['best_param'] = best_param1
    results['best_train_perf'] = best_train_perf
    results['best_val_perf'] = best_val_perf
    results['best_test_perf'] = best_test_perf

    save_results(results, data_settings, training_settings, filename, foldername)


def batch_bar_LTL_old(data, data_settings, training_settings):
    foldername = training_settings['foldername']
    filename = training_settings['filename']
    param1_range = training_settings['param1_range']
    n_tasks = data_settings['n_tasks']
    n_dims = data_settings['n_dims']
    task_range = data_settings['task_range']
    task_range_tr = data_settings['task_range_tr']
    task_range_val = data_settings['task_range_val']
    task_range_test = data_settings['task_range_test']
    alpha_value = training_settings['alpha_value']
    W_true = data['W_true']


    T = len(task_range_tr)

    # time_lapsed = [None] * T

    if data_settings['dataset'] == 'synthetic_regression':
        best_val_performance = 10 ** 8
    else:
        best_val_performance = -10 ** 8


    validation_curve = [[] for i in range(len(param1_range))]
    for param1_idx, param1 in enumerate(param1_range):
        all_train_perf, all_val_perf, all_test_perf = [[] for i in range(T)], [[] for i in range(T)], [[] for i in range(T)]

        W_pred = np.zeros((n_dims, n_tasks))

        c_iter = 0
        t = time.time()
        #####################################################
        # OPTIMISATION
        X_train, Y_train = [None] * n_tasks, [None] * n_tasks
        n_points = [0] * n_tasks
        for _, task_idx in enumerate(task_range_tr):
            X_train[task_idx] = data['X_train'][task_idx]
            Y_train[task_idx] = data['Y_train'][task_idx]
            n_points[task_idx] = len(Y_train[task_idx])


        A = np.zeros((n_dims, n_dims))
        b = np.zeros(n_dims)
        for _, task_idx in enumerate(task_range_tr):
            G = pinv(X_train[task_idx] @ X_train[task_idx].T + n_points[task_idx] * param1 * eye(n_points[task_idx]))

            A = A + X_train[task_idx].T @ G @ G @ X_train[task_idx]
            b = b + X_train[task_idx].T @ G @ Y_train[task_idx]
        A = 1 / T * A
        b = 1 / T * b
        h = pinv(A) @ b

        # if norm(h) > alpha_value:
        #     h = alpha_value * h / norm(h)
        # else:
        #     k = 1

        # time_lapsed[pure_task_idx] = time.time() - t

        # Check performance on ALL training tasks for this D
        W_pred = solve_wrt_w(h, data['X_train'], data['Y_train'], n_tasks, data, W_pred, param1, task_range_tr)
        train_perf = mean_squared_error(data_settings, data['X_train'], data['Y_train'], W_pred, W_true, task_range_tr)

        #####################################################
        # VALIDATION
        W_pred = solve_wrt_w(h, data['X_train'], data['Y_train'], n_tasks, data, W_pred, param1, task_range_val)
        val_perf = mean_squared_error(data_settings, data['X_val'], data['Y_val'], W_pred, W_true, task_range_val)

        #####################################################
        # TEST
        X_train, Y_train = [None] * n_tasks, [None] * n_tasks
        for _, task_idx in enumerate(task_range_test):
            X_train[task_idx] = np.concatenate((data['X_val'][task_idx], data['X_train'][task_idx]))
            Y_train[task_idx] = np.concatenate((data['Y_val'][task_idx], data['Y_train'][task_idx]))

        W_pred = solve_wrt_w(h, X_train, Y_train, n_tasks, data, W_pred, param1, task_range_test)
        test_perf = mean_squared_error(data_settings, data['X_test'], data['Y_test'], W_pred, W_true, task_range_test)

        print('T: ALL | lambda: %6e | val MSE: %7.5f | test MSE: %7.5f | norm h: %4.2f' %
              (param1, val_perf, test_perf, norm(h)))

        validation_curve[param1_idx] = val_perf

        if data_settings['dataset'] == 'synthetic_regression':
            if val_perf < best_val_performance:
                validation_criterion = True
            else:
                validation_criterion = False
        else:
            if val_perf > best_val_performance:
                validation_criterion = True
            else:
                validation_criterion = False


        # print(W_pred[:, 201])
        # print(data['W_true'][:, 201])

        if validation_criterion == True:
            best_val_performance = val_perf

            best_param1 = param1
            best_h = h
            best_W = W_pred
            # print(best_h)
            # print(best_W[:, 201])
            # print(data['W_true'][:, 201])
            best_train_perf = train_perf
            best_val_perf = val_perf
            best_test_perf = test_perf

            # print("best validation stats:")
            # print('param1: %8e | val MSE: %7.5f | test MSE: %7.5f | norm h: %4.2f' %
            #       (param1, val_perf, test_perf, norm(best_h)))
            # print("")

    plt.figure()
    plt.plot(validation_curve)
    plt.pause(0.01)

    plt.figure();
    plt.imshow(best_W)
    plt.figure();
    plt.imshow(data['W_true'])
    plt.pause(0.01)

    results = {}
    results['best_param'] = best_param1
    # results['D'] = best_D
    results['best_train_perf'] = best_train_perf
    results['best_val_perf'] = best_val_perf
    results['best_test_perf'] = best_test_perf

    save_results(results, data_settings, training_settings, filename, foldername)