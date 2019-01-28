import numpy as np
from data import data_generator as gen
from least_mean_squares import least_mean_squares, LTL_evaluation, least_mean_squares_train_only
from plots import plot
from utils import print_metric_mean_and_std
import datetime
import os


def make_exp_dir(experiment_name):
    now = datetime.datetime.now()
    experiment_path = experiment_name + '-' + str(now)
    os.makedirs(experiment_path)
    return experiment_path


def save_exp_parameters(exp_parameters, exp_dir_path):
    import json
    with open(os.path.join(exp_dir_path, 'parameters.txt'), 'w') as file:
        file.write(json.dumps(exp_parameters))  # use `json.loads` to do the reverse


def exp1(seed=0, y_snr=5, task_std=1, use_valid=False, show_plot=False):
    return exp(seed, task_std, y_snr, 'exp1', use_valid, show_plot)


def exp2(seed=0, y_snr=5, task_std=1, use_valid=False, show_plot=False):
    return exp(seed, task_std, y_snr, 'exp2', use_valid, show_plot)


def exp1_multirun(n_seeds=30, y_snr=5, task_std=1, use_valid=False, show_plot=False):
    results = []
    for i in range(n_seeds):
        results.append(exp1(i, y_snr, task_std, use_valid, show_plot))

    exp_name = 'exp_1'

    exp_parameters = results[0]['params']
    use_valid_str = ''

    mses_ltl = np.mean(np.concatenate([np.expand_dims(s['mses_ltl'], 0) for s in results], axis=0), axis=0)
    mses_itl = np.mean(np.concatenate([np.expand_dims(s['mses_itl'], 0) for s in results], axis=0), axis=0)
    mses_oracle = np.mean(np.concatenate([np.expand_dims(s['mses_oracle'], 0) for s in results], axis=0), axis=0)
    hs = np.mean(np.concatenate([np.expand_dims(s['hs'], 0) for s in results], axis=0), axis=0)

    exp_str = exp_name + 'n_seeds' + str(n_seeds) + 'taskstd' + str(task_std) + 'y_snr' + str(y_snr)

    exp_dir_path = make_exp_dir(exp_str)
    save_exp_parameters(exp_parameters, exp_dir_path)
    plot(mses_ltl, mses_itl, mses_oracle, use_valid_str, y_label='test MSE (mean and std over test tasks)',
         save_dir_path=exp_dir_path, show_plot=show_plot)
    np.savetxt(os.path.join(exp_dir_path, "ltl.csv"), mses_ltl, delimiter=",")
    np.savetxt(os.path.join(exp_dir_path, "itl.csv"), mses_itl, delimiter=",")
    np.savetxt(os.path.join(exp_dir_path, "oracle.csv"), mses_itl, delimiter=",")
    np.savetxt(os.path.join(exp_dir_path, "hs.csv"), hs, delimiter=",")

    return {'mses_itl': mses_itl, 'mses_oracle': mses_oracle, 'mses_ltl': mses_ltl, 'hs': hs, 'params': exp_parameters}


def exp(seed, task_std, y_snr, exp_name, use_valid, show_plot):
    # outer-parameters
    gamma = 10 # outer learning rate
    alpha = 0.00  # mean regularization parameter

    # inner-parameters
    lmbd = 0.1  # inner regularization parameter

    # data settings
    n_tasks = 100
    n_tasks_test = 200
    n_points = 110
    n_dims = 30

    # set experiment name with some hyperparameters that you want to change
    exp_str = exp_name + 'taskstd' + str(task_std) + 'y_snr' + str(y_snr)

    val_perc = 0.0
    use_valid_str = ''
    if use_valid:
        use_valid_str = 'with valid'
        val_perc = 0.5

    exp_parameters = locals()
    print('parameters ' + exp_name, exp_parameters)

    tasks_gen = gen.TasksGenerator(seed=seed, n_tasks=n_tasks, val_perc=val_perc, n_dims=n_dims, n_points=n_points,
                                   y_snr=y_snr, task_std=task_std, tasks_generation=exp_name)

    data_train, oracle_train = tasks_gen.gen_tasks(n_tasks)
    data_valid, oracle_valid = tasks_gen.gen_tasks(n_tasks_test)

    if use_valid:
        hs, mses_ltl = least_mean_squares(gamma, alpha, lmbd, data_train['X_train'], data_train['Y_train'],
                                          data_train['X_val'], data_train['Y_val'], data_valid)
    else:
        hs, mses_ltl = least_mean_squares_train_only(gamma, alpha, lmbd, data_train['X_train'], data_train['Y_train'],
                                                     data_valid)

    print('hs :', hs)

    mses_itl = LTL_evaluation(lmbd, np.zeros(n_dims), data_valid['X_train'], data_valid['Y_train'],
                              data_valid['X_test'],
                              data_valid['Y_test'], verbose=0)

    mses_oracle = LTL_evaluation(lmbd, oracle_valid['w_bar'], data_valid['X_train'], data_valid['Y_train'],
                                 data_valid['X_test'], data_valid['Y_test'], verbose=0)

    print_metric_mean_and_std(mses_itl, name="ITL")
    print_metric_mean_and_std(mses_ltl[-1], name="LTL")
    print_metric_mean_and_std(mses_oracle, name="Oracle")

    exp_dir_path = make_exp_dir(exp_str)
    save_exp_parameters(exp_parameters, exp_dir_path)
    plot(mses_ltl, mses_itl, mses_oracle, use_valid_str, ylabel='test MSE (mean and std over test tasks)',
         save_dir_path=exp_dir_path, show_plot=show_plot)
    np.savetxt(os.path.join(exp_dir_path, "ltl.csv"), mses_ltl, delimiter=",")
    np.savetxt(os.path.join(exp_dir_path, "itl.csv"), mses_itl, delimiter=",")
    np.savetxt(os.path.join(exp_dir_path, "oracle.csv"), mses_itl, delimiter=",")
    np.savetxt(os.path.join(exp_dir_path, "hs.csv"), hs, delimiter=",")

    return {'mses_itl': mses_itl, 'mses_oracle': mses_oracle, 'mses_ltl': mses_ltl, 'hs': hs, 'params': exp_parameters}


def exp1_varying_taskstd():
    for task_std in np.linspace(0, 5, 11):
        exp1(task_std=task_std)


def exp1_varying_seeds():
    for seed in range(10):
        exp1(seed=seed, task_std=5, show_plot=True)


if __name__ == '__main__':
    exp1_multirun(task_std=1, show_plot=True)


