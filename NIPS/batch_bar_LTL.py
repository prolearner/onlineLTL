# import optimisation

from NIPS.utilities import *
from numpy.linalg import pinv
from numpy import identity as eye

def batch_bar_LTL(data, data_settings, training_settings):
    param1_range = training_settings['param1_range']
    n_tasks = data_settings['n_tasks']
    n_dims = data_settings['n_dims']
    task_range_tr = data_settings['task_range_tr']
    task_range_val = data_settings['task_range_val']
    task_range_test = data_settings['task_range_test']
    alpha_range = training_settings['alpha_value']
    r_split = training_settings['r_split']
    W_true = data['W_true']




    r_range = range(0, len(data['Y_train'][0]))
    alt_plot = [None] * len(r_range)

    T = len(task_range_tr)
    # skipping tasks for speed improvement
    n_tasks_step = int(round(0.08 * T))
    task_range_tr_indeces = np.arange(0, T - 1, n_tasks_step)
    if T - 1 not in task_range_tr_indeces:
        task_range_tr_indeces = np.append(task_range_tr_indeces, T - 1)
    task_range_tr_untouched = task_range_tr
    task_range_tr = [task_range_tr[i] for i in task_range_tr_indeces]

    for r_split_idx, r_split in enumerate(r_range):
        print("r: %3d" % r_split)

        if data_settings['dataset'] == 'synthetic_regression':
            best_val_performance = 10 ** 8
        else:
            best_val_performance = -10 ** 8

        validation_curve = [[] for i in range(len(param1_range))]
        validation_plain = np.zeros((len(param1_range), len(alpha_range)))
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

                # n_tr_temp = int(np.floor(0.05 * n_total_tr))
                n_tr_temp = r_split
                n_ts_temp = int(n_total_tr - n_tr_temp)

                X_tr[task_idx] = data['X_train'][task_idx][0:n_tr_temp]
                Y_tr[task_idx] = data['Y_train'][task_idx][0:n_tr_temp]

                X_ts[task_idx] = data['X_train'][task_idx][n_tr_temp:]
                Y_ts[task_idx] = data['Y_train'][task_idx][n_tr_temp:]

                n_tr[task_idx] = n_tr
                n_ts[task_idx] = n_ts_temp

                C_r = X_tr[task_idx].T @ X_tr[task_idx] + n_total_tr * param1 * eye(n_dims)
                # C_r = X_tr[task_idx].T @ X_tr[task_idx] + n_tr_temp * param1 * eye(n_dims)

                w_0 = pinv(C_r) @ X_tr[task_idx].T @ Y_tr[task_idx]
                X_bar_temp = n_total_tr * param1 * X_ts[task_idx] @ pinv(C_r)
                # X_bar_temp = n_tr_temp * param1 * X_ts[task_idx] @ pinv(C_r)

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
            for alpha_idx, alpha_value in enumerate(alpha_range):
                all_test_perf = [None] * T
                all_val_perf = [None] * T
                W_pred = np.zeros((n_dims, n_tasks))

                curr_task_range_tr = []  ##################
                for pure_task_idx, _ in enumerate(task_range_tr):  ##################

                    pure_task_idx = task_range_tr_indeces[pure_task_idx]
                    curr_task_range_tr = task_range_tr_untouched[:pure_task_idx+1]
                    # print(len(curr_task_range_tr))

                    # curr_task_range_tr.append(new_tr_task)  ##################


                    if r_split_idx == 1:
                        k = 1


                    #####################################################
                    # OPTIMISATION
                    # obj = lambda h: sum([0.5/n_ts[i] * norm(X_bar[i] @ h - Y_bar[i])**2 for i in curr_task_range_tr])
                    # grad = lambda h: sum([1/n_ts[i] * XtX_bar[i] @ h - XtY_bar[i] for i in curr_task_range_tr])

                    curr_X_bar = X_bar[curr_task_range_tr[0]]
                    curr_Y_bar = Y_bar[curr_task_range_tr[0]]
                    if len(curr_task_range_tr) > 1:
                        for _, task_idx in enumerate(curr_task_range_tr[1:]):
                            curr_X_bar = np.vstack((curr_X_bar, X_bar[task_idx]))
                            curr_Y_bar = np.hstack((curr_Y_bar, Y_bar[task_idx]))

                    h = pinv(curr_X_bar.T @ curr_X_bar + alpha_value*eye(n_dims)) @ curr_X_bar.T @ curr_Y_bar
                    # print(h)

                    # if norm(h) > alpha_value:
                    #     h = alpha_value * h / norm(h)
                    # else:
                    #     k = 1

                    #####################################################
                    # VALIDATION
                    W_pred = solve_wrt_w_cooked(h, XtXinv_val, XtY_val, data['X_train'], W_pred, param1, task_range_val)
                    val_perf = mean_squared_error(data_settings, data['X_val'], data['Y_val'], W_pred, W_true, task_range_val)
                    all_val_perf[pure_task_idx] = val_perf
                    # if r_split_idx > 0:
                    #     print(all_val_perf)
                    # print(T)

                    #####################################################
                # TEST
                X_train, Y_train = [None] * n_tasks, [None] * n_tasks
                for _, task_idx in enumerate(task_range_test):
                    X_train[task_idx] = np.concatenate((data['X_val'][task_idx], data['X_train'][task_idx]))
                    Y_train[task_idx] = np.concatenate((data['Y_val'][task_idx], data['Y_train'][task_idx]))

                W_pred = solve_wrt_w_cooked(h, XtXinv_test, XtY_test, X_train, W_pred, param1, task_range_test)
                test_perf = mean_squared_error(data_settings, data['X_test'], data['Y_test'], W_pred, W_true, task_range_test)
                all_test_perf[pure_task_idx] = test_perf


                validation_curve[param1_idx] = val_perf
                validation_plain[param1_idx, alpha_idx] = val_perf

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
                    best_alpha = alpha_value
                    best_W = W_pred
                    best_val_perf = val_perf
                    best_test_perf = test_perf

                    best_test_per_vec = all_test_perf
                    best_val_per_vec = all_val_perf
                    alt_plot[r_split_idx] = all_val_perf[-1]


                print('lambda: %6e | a: %6e | val MSE: %7.5f | test MSE: %7.5f | ||h||: %4.2f' %
                      (param1, alpha_value, val_perf, test_perf, norm(h)))
        print("")

# good
#         plt.figure(1)
#         best_val_per_vec_plot = np.array(best_val_per_vec).astype(np.double)
#         s1mask = np.isfinite(best_val_per_vec_plot)
#         task_range_tr_plot = np.array(range(T))
#         plt.plot(task_range_tr_plot[s1mask], best_val_per_vec_plot[s1mask])
#
#         if r_split_idx == 0:
#             ymin, ymax = plt.ylim()
#             ylim_val = min(best_val_per_vec_plot[s1mask]) + 0.5 * min(best_val_per_vec_plot[s1mask])
#             # ylim_val = max(0.1 * max(best_val_per_vec_plot[s1mask]), ymax)
#
#             # ylim_val = max(ylim_val, ymax)
#
#         # plt.ylim(ymax=ylim_val)
#         plt.pause(0.01)
#
#         plt.figure(2)
#         alt_plot_plot = np.array(alt_plot).astype(np.double)
#         s1mask = np.isfinite(alt_plot_plot)
#
#         from itertools import compress
#         s1mask_range = list(compress(r_range, s1mask))
#
#         plt.plot(s1mask_range, alt_plot_plot[s1mask], 'b')
#         plt.pause(0.01)




        # plt.figure()
        # plt.imshow(validation_plain)
        # plt.colorbar()
        # plt.pause(0.01)


# good
#     plt.figure(1)
#     legend = [str(i) for i in list(r_range)]
#     plt.legend(legend)



    # plt.figure()
    # plt.plot(r_range, alt_plot)
    # plt.pause(0.01)


    # plt.figure()
    # plt.plot(best_test_per_vec)
    # plt.plot(best_val_per_vec, '-.')
    # plt.pause(0.01)
    #
    # plt.figure()
    # plt.plot(validation_curve)
    # plt.pause(0.01)




    # plt.figure()
    # plt.imshow(best_W)
    # plt.figure()
    # plt.imshow(data['W_true'])
    # plt.pause(0.01)

    results = {}
    results['best_param'] = best_param1
    results['best_alpha'] = best_alpha
    results['best_val_perf'] = best_val_per_vec
    results['best_test_perf'] = best_test_per_vec
    results['val_error_vs_tr_points'] = alt_plot


    save_results(results, data_settings, training_settings)

    k = 1
    return