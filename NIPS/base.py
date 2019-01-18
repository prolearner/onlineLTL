from NIPS.training import *

def main(data_settings, training_settings):

    training(data, data_settings, training_settings)

    return


if __name__ == "__main__":

    def split(a, n):
        k, m = divmod(len(a), n)
        return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

    if len(sys.argv) > 1:
        seed = int(sys.argv[1])
        n_points = int(sys.argv[2])
        n_tasks = int(sys.argv[3])
        n_dims = int(sys.argv[4])
        method_IDX = int(sys.argv[5]) # 0 batch, 1 online
        dataset_IDX = int(sys.argv[6])
        if method_IDX == 1:
            # alpha_value = int(sys.argv[7])
            # alpha_value = [10 ** float(i) for i in np.linspace(-6, 1, 10)]
            alpha_value = [0]
        else:
            # lambda_idx = int(sys.argv[7])
            # alpha_value = np.nan
            lambda_idx = 0
            # alpha_value = [10 ** float(i) for i in np.linspace(-6, 1, 10)]
            alpha_value = [0]
    else:
        seed = 1
        n_points = 120 # 530
        n_tasks = 120
        n_dims = 30 # 50
        # 3 ITL 4 oracle 5 batch 9 online_with_split
        method_IDX = 9 ###########################################
        dataset_IDX = 0
        # step_range = [10 ** float(i) for i in range(-6, 0)]
        if (method_IDX == 1) or (method_IDX == 2) or (method_IDX == 5) or (method_IDX == 9):
            # alpha_value = [10 ** float(i) for i in np.linspace(-6, 0, 6)]
            alpha_value = [0]
        else:
            alpha_value = np.nan


    r_split = 2 # spam

    step_range = [10 ** float(i) for i in np.linspace(-3, 3, 16)]
    lambda_range = [10 ** float(i) for i in np.linspace(-6, 3, 40)]
    # step_range = [0.0001]
    # lambda_range = [0.0001]

    # step_range = [10 ** float(i) for i in np.linspace(2, 3, 8)]
    # step_range.insert(0, 0.001)
    # lambda_range = [10 ** float(i) for i in np.linspace(-5, -3, 10)]

    data_settings = {}

    data_settings['seed'] = 2 # seed

    data_settings['n_points'] = n_points
    data_settings['n_dims'] = n_dims

    if dataset_IDX == 0:
        n_test_tasks = 100 #####################
        data_settings['n_tasks'] = n_tasks + n_test_tasks
    elif dataset_IDX == 1:
        data_settings['n_tasks'] = 139
        data_settings['schools_train_tasks'] = n_tasks

    # data_settings['y_noise'] = 2 # snr
    # data_settings['w_std'] = 2
    # data_settings['y_noise'] = 5 # snr
    # data_settings['w_std'] = 0.5
    data_settings['y_noise'] = 5  # snr
    data_settings['w_std'] = 1

    np.random.seed(data_settings['seed'])

    training_settings = {}
    # setting for step size on online LTL
    training_settings['alpha_value'] = alpha_value
    training_settings['conv_tol'] = 10 ** -5 # for batch
    training_settings['param1_range'] = lambda_range
    training_settings['r_split'] = r_split
    training_settings['step_range'] = step_range

    ########################################
    ########################################
    # data generation
    if dataset_IDX == 0:
        data = synthetic_data_gen(data_settings)
        data_settings['dataset'] = 'synthetic_regression'
    elif dataset_IDX == 1:
        n_tasks_og = n_tasks
        n_train_tasks = int(round(n_tasks * 0.5))
        # n_train_tasks = 80
        n_val_tasks = n_tasks - n_train_tasks
        n_test_tasks = 139 - n_train_tasks - n_val_tasks
        n_tasks = 139

        task_shuffled = np.random.permutation(n_tasks)

        data_settings['task_range_tr'] = task_shuffled[0:n_train_tasks]
        data_settings['task_range_val'] = task_shuffled[n_train_tasks:n_train_tasks + n_val_tasks]
        data_settings['task_range_test'] = task_shuffled[n_train_tasks + n_val_tasks:]

        data_settings['task_range'] = task_shuffled
        data_settings['n_tasks'] = n_tasks
        # pass

    temp_seed = seed
    np.random.seed(temp_seed)

    # move to datagen
    if dataset_IDX == 0:
        n_train_tasks = round(n_tasks * 0.5)
        n_val_tasks = n_tasks - n_train_tasks

        shuffled_tasks = np.random.permutation(n_tasks+n_test_tasks)

        data_settings['task_range_tr'] = list(shuffled_tasks[:n_train_tasks])
        data_settings['task_range_val'] = list(shuffled_tasks[n_train_tasks:n_tasks])
        data_settings['task_range_test'] = list(shuffled_tasks[n_tasks:n_tasks + n_test_tasks])

        n_tasks = n_train_tasks + n_val_tasks + n_test_tasks
        data_settings['n_tasks'] = n_tasks

        data_settings['task_range'] = np.nan # list(np.arange(0, n_tasks))
    elif dataset_IDX == 1:
        data_settings['dataset'] = 'schools'
        data, data_settings = schools_data_gen(data_settings)
        # pass
    ########################################
    ########################################

    # print(data['X_train'][0][1,:])
    # print(data_settings['task_range_tr'])
    t = time.time()

    if (method_IDX == 1) or (method_IDX == 2) or (method_IDX == 5) or (method_IDX == 9):
        if method_IDX == 5:
            training_settings['method'] = 'batch_bar_LTL'
        elif method_IDX == 9:
            training_settings['method'] = 'online_with_split_no_constraint'
        else:
            training_settings['method'] = 'online_var_LTL_v' + str(method_IDX)
        training_settings['filename'] = "seed_" + str(seed)

        if dataset_IDX == 0:
            training_settings['foldername'] = 'results/' + data_settings['dataset'] + '-T_' + \
                                              str(n_tasks) + '-n_' + str(n_points) + '/' \
                                              + training_settings['method']
        elif dataset_IDX == 1:
            training_settings['foldername'] = 'results/' + data_settings['dataset'] + '-T_' + \
                                              str(n_tasks_og) + '/' + training_settings['method']

        main(data_settings, training_settings)
    elif (method_IDX == 3) or (method_IDX == 4):
        if method_IDX == 3:
            training_settings['method'] = 'var_ITL'
        elif method_IDX == 4:
            training_settings['method'] = 'var_ITL_oracle'

        training_settings['filename'] = "seed_" + str(seed)

        if dataset_IDX == 0:
            training_settings['foldername'] = 'results/' + data_settings['dataset'] + '-T_' + \
                                              str(n_tasks) + '-n_' + str(n_points) + '/' \
                                              + training_settings['method']
        elif dataset_IDX == 1:
            training_settings['foldername'] = 'results/' + data_settings['dataset'] + '-T_' + \
                                              str(n_tasks_og) + '/' + training_settings['method']

        main(data_settings, training_settings)
    print(time.time() - t)
    print(time.time() - t)
    print(time.time() - t)
    print(time.time() - t)



    print("done")
