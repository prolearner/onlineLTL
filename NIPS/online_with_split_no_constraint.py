# import optimisation

from NIPS.utilities import *

from numpy.linalg import pinv
from numpy import identity as eye


def online_with_split_no_constraint(data, data_settings, training_settings):
    param1_range = training_settings['param1_range']
    n_tasks = data_settings['n_tasks']
    n_dims = data_settings['n_dims']
    task_range_tr = data_settings['task_range_tr']
    task_range_val = data_settings['task_range_val']
    task_range_test = data_settings['task_range_test']
    alpha_range = training_settings['alpha_value']
    # r_split = training_settings['r_split']
    # W_true = data['W_true']
    W_true = np.nan
    step_range = training_settings['step_range']

    temp = []
    for idx in task_range_tr:
        temp.append(len(data['Y_train'][idx]))
    r_range = range(0, np.min(temp))
    alt_plot = [None] * len(r_range)

    if data_settings['dataset'] == 'schools':
        np.random.shuffle(task_range_tr)

    T = len(task_range_tr)
    task_range_tr_untouched = task_range_tr

    all_vals = np.zeros((len(r_range), len(param1_range), len(step_range), len(task_range_tr_untouched)))

    for r_split_idx, r_split in enumerate(r_range):
        print("r: %3d" % r_split)

        best_h = random.randn(n_dims)

        if data_settings['dataset'] == 'synthetic_regression':
            best_val_performance = 10 ** 8
        else:
            best_val_performance = -10 ** 8

        validation_curve = [[] for i in range(len(param1_range))]

        for param1_idx, param1 in enumerate(param1_range):
            #####################################################
            # precompute components
            X_tr, Y_tr, X_ts, Y_ts = [None] * n_tasks, [None] * n_tasks, [None] * n_tasks, [None] * n_tasks
            XtX_bar, XtY_bar = [None] * n_tasks, [None] * n_tasks
            X_bar, Y_bar = [None] * n_tasks, [None] * n_tasks
            n_tr = [0] * n_tasks
            n_ts = [0] * n_tasks
            for _, task_idx in enumerate(task_range_tr_untouched):
                n_total_tr = len(data['Y_train'][task_idx])

                n_tr_temp = r_split
                n_ts_temp = int(n_total_tr - n_tr_temp)

                X_tr[task_idx] = data['X_train'][task_idx][0:n_tr_temp]
                Y_tr[task_idx] = data['Y_train'][task_idx][0:n_tr_temp]

                X_ts[task_idx] = data['X_train'][task_idx][n_tr_temp:]
                Y_ts[task_idx] = data['Y_train'][task_idx][n_tr_temp:]

                n_tr[task_idx] = n_tr
                n_ts[task_idx] = n_ts_temp

                C_r = X_tr[task_idx].T @ X_tr[task_idx] + n_total_tr * param1 * eye(n_dims)

                w_0 = pinv(C_r) @ X_tr[task_idx].T @ Y_tr[task_idx]
                X_bar_temp = n_total_tr * param1 * X_ts[task_idx] @ pinv(C_r)
                Y_bar_temp = Y_ts[task_idx] - X_ts[task_idx] @ w_0

                X_bar[task_idx] = X_bar_temp
                Y_bar[task_idx] = Y_bar_temp

                XtX_bar[task_idx] = X_bar_temp.T @ X_bar_temp
                XtY_bar[task_idx] = X_bar_temp.T @ Y_bar_temp

            XtXinv_val, XtY_val = [None] * n_tasks, [None] * n_tasks
            for _, task_idx in enumerate(task_range_val):
                X = data['X_train'][task_idx]
                Y = data['Y_train'][task_idx]
                n_points = len(Y)

                XtXinv_val[task_idx] = pinv(X.T @ X + param1 * n_points * eye(n_dims))
                XtY_val[task_idx] = X.T @ Y

            XtXinv_test, XtY_test = [None] * n_tasks, [None] * n_tasks
            for _, task_idx in enumerate(task_range_test):
                X = np.concatenate((data['X_val'][task_idx], data['X_train'][task_idx]))
                Y = np.concatenate((data['Y_val'][task_idx], data['Y_train'][task_idx]))
                n_points = len(Y)

                XtXinv_test[task_idx] = pinv(X.T @ X + param1 * n_points * eye(n_dims))
                XtY_test[task_idx] = X.T @ Y
            #####################################################



            # step_gamma = 0.01
            # while step_gamma < 1000:



            for step_gamma_idx, step_gamma in enumerate(step_range):
                all_test_perf = [None] * T
                all_val_perf = [None] * T

                W_pred = np.zeros((n_dims, n_tasks))
                all_h = [None] * T
                h = best_h

                ###
                delete_objectives = []

                c_iter = 0
                for pure_task_idx, curr_task in enumerate(task_range_tr):
                    currX_bar = X_bar[curr_task]
                    currY_bar = Y_bar[curr_task]
                    currN_ts = n_ts[curr_task]

                    obj = lambda h: 0.5/currN_ts * norm(currX_bar @ h - currY_bar)**2
                    grad = lambda h: 1/currN_ts * currX_bar.T @ (currX_bar @ h - currY_bar)

                    h, c_iter, delete_new_obj = solve_wrt_h_stochastic_generic(h, obj, grad, step_gamma, c_iter)
                    delete_objectives.append(delete_new_obj)

                    all_h[pure_task_idx] = h
                    if pure_task_idx > 0:
                        h = np.average(all_h[:pure_task_idx], axis=0)

                    #####################################################
                    # VALIDATION
                    W_pred = solve_wrt_w_cooked(h, XtXinv_val, XtY_val, data['X_train'], W_pred, param1, task_range_val)

                    val_perf = mean_squared_error(data_settings, data['X_val'], data['Y_val'], W_pred, W_true, task_range_val)
                    all_val_perf[pure_task_idx] = val_perf



                    # step_gamma = 1.25 * step_gamma

                    # all_vals[r_split_idx, param1_idx, step_gamma_idx, pure_task_idx] = val_perf

                #####################################################
                # TEST
                X_train, Y_train = [None] * n_tasks, [None] * n_tasks
                for _, task_idx in enumerate(task_range_test):
                    X_train[task_idx] = np.concatenate((data['X_val'][task_idx], data['X_train'][task_idx]))
                    Y_train[task_idx] = np.concatenate((data['Y_val'][task_idx], data['Y_train'][task_idx]))

                W_pred = solve_wrt_w_cooked(h, XtXinv_test, XtY_test, X_train, W_pred, param1, task_range_test)

                test_perf = mean_squared_error(data_settings, data['X_test'], data['Y_test'], W_pred, W_true, task_range_test)
                all_test_perf[pure_task_idx] = test_perf

                # print('lambda: %6e | val: %8.5f | test: %8.5f | g: %7.6f' % (param1, val_perf, test_perf, step_gamma))

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
                    best_h = h

                    print('lambda: %6e | val: %8.5f | test: %8.5f | g: %7.6f' % (param1, val_perf, test_perf, step_gamma))

                    best_param1 = param1
                    best_param1_idx = param1_idx

                    best_test_per_vec = all_test_perf
                    best_val_per_vec = all_val_perf
                    alt_plot[r_split_idx] = all_val_perf[-1]
            print("")
        print("")
        # good
        plt.figure(1)
        best_val_per_vec_plot = np.array(best_val_per_vec).astype(np.double)
        s1mask = np.isfinite(best_val_per_vec_plot)
        task_range_tr_plot = np.array(range(T))
        plt.plot(task_range_tr_plot[s1mask], best_val_per_vec_plot[s1mask])

        if r_split_idx == 0:
            ymin, ymax = plt.ylim()
            ylim_val = min(best_val_per_vec_plot[s1mask]) + 0.5 * min(best_val_per_vec_plot[s1mask])
            # ylim_val = max(0.1 * max(best_val_per_vec_plot[s1mask]), ymax)

        # plt.ylim(ymax=ylim_val)
        plt.pause(0.01)

        plt.figure(2)
        alt_plot_plot = np.array(alt_plot).astype(np.double)
        s1mask = np.isfinite(alt_plot_plot)

        from itertools import compress
        s1mask_range = list(compress(r_range, s1mask))

        plt.plot(s1mask_range, alt_plot_plot[s1mask], 'b')
        plt.pause(0.01)

    # good
    plt.figure(1)
    legend = [str(i) for i in list(r_range)]
    plt.legend(legend)

    k = 1

    # plt.figure()
    # plt.plot(best_test_perf)
    # plt.pause(0.01)
    #
    # plt.figure()
    # plt.plot(validation_curve)
    # plt.pause(0.01)

    results = {}
    results['best_param'] = best_param1
    results['best_param_idx'] = best_param1_idx
    results['best_val_perf'] = best_val_per_vec
    results['best_test_perf'] = best_test_per_vec
    results['val_error_vs_tr_points'] = alt_plot
    # results['all_vals'] = all_vals


    save_results(results, data_settings, training_settings)

    return