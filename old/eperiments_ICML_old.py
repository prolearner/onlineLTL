import numpy as np
import data_generator as gen
from algorithms import meta_ssgd, LTL_evaluation, InnerSSubGD, InnerSubGD, no_train_evaluation, \
    FISTA
from plots import plot
from utils import print_metric_mean_and_std, is_jsonable
import datetime
from losses import hinge_loss, grad_hinge_loss, absolute_loss, grad_absolute_loss
import os


def make_exp_dir(experiment_name):
    now = datetime.datetime.now()
    experiment_path = experiment_name + '-' + str(now)
    os.makedirs(experiment_path)
    return experiment_path


def save_exp_parameters(exp_parameters, exp_dir_path):
    param_serializable = []
    for ep in exp_parameters:
        if is_jsonable(ep):
            param_serializable.append(str(ep))
        else:
            param_serializable.append(ep)

    import json
    with open(os.path.join(exp_dir_path, 'parameters.txt'), 'w') as file:
        file.write(json.dumps(param_serializable))  # use `json.loads` to do the reverse


def exp1(seed=0, y_snr=100, task_std=1, n_tasks=100, n_train=100, n_dims=30, alpha=10, lmbd=0.01, gamma=None,
         n_tasks_test=200, n_test=100, val_perc=0.0, exp_dir='', inner_solver_str=('ssubgd', 'subgd'),
         inner_solver_test_str='ssubgd', show_plot=False):
    # regression experiment with the absolute loss |y - yhat|

    tasks_gen = gen.TasksGenerator(seed=seed, task_std=task_std, y_snr=y_snr, val_perc=val_perc,
                                   tasks_generation='exp1')

    exp_name = 'expICMLreg1' + 'taskstd' + str(task_std) + 'y_snr' + str(y_snr) + 'is'
    return exp(exp_name=exp_name, seed=seed, tasks_gen=tasks_gen, loss_f=absolute_loss, grad_loss_f=grad_absolute_loss,
               alpha=alpha, lmbd=lmbd, gamma=gamma, exp_dir=exp_dir,
               n_tasks=n_tasks, n_train=n_train, n_dims=n_dims, inner_solver_str=inner_solver_str,
               inner_solver_test_str= inner_solver_test_str,
               n_tasks_test=n_tasks_test, n_test=n_test, val_perc=val_perc,
               show_plot=show_plot, y_label='test AE (mean and std over test tasks)',
               metric_name='AE loss', eval_online=True)


def exp2(seed=0, y_snr=100, task_std=2, n_tasks=100, n_train=100, n_dims=30, alpha=10, lmbd=0.01, gamma=None,
         n_tasks_test=200, n_test=100, val_perc=0.0, exp_dir='', inner_solver_str=('ssubgd', 'subgd'),
         inner_solver_test_str='ssubgd', show_plot=False):
    # classification experiment with the hinge loss: max(1 - yhat*y, 0)

    tasks_gen = gen.TasksGenerator(seed=seed, task_std=task_std, y_snr=y_snr, val_perc=val_perc,
                                   tasks_generation='expclass')

    exp_name = 'expICMLclass1' + 'taskstd' + str(task_std) + 'y_snr' + str(y_snr)
    return exp(exp_name=exp_name, seed=seed, tasks_gen=tasks_gen, loss_f=hinge_loss, grad_loss_f=grad_hinge_loss,
               alpha=alpha, lmbd=lmbd, gamma=gamma, n_tasks=n_tasks, n_train=n_train, n_dims=n_dims, exp_dir=exp_dir,
               n_tasks_test=n_tasks_test, n_test=n_test, val_perc=val_perc, inner_solver_str=inner_solver_str,
               inner_solver_test_str=inner_solver_test_str,
               show_plot=show_plot, y_label='test Hinge loss (mean and std over test tasks)',
               metric_name='H loss', eval_online=True)


def exp(exp_name, seed, tasks_gen, loss_f, grad_loss_f, alpha=0.1, lmbd=(0.01, 0.1), gamma=None,
        n_tasks=100, n_train=5, n_dims=30, n_tasks_test=200, n_test=100, val_perc=0.0, show_plot=False,
        inner_solver_str=('ssubgd', 'subgd'), inner_solver_test_str='ssubgd', exp_dir='', y_label='', metric_name='loss',
        verbose=1, eval_online=True):
    exp_str = exp_name + 'is' + str(inner_solver_str) + 'ist' + inner_solver_test_str +\
              'alpha' + str(alpha) + 'lmbd' + str(lmbd) + 'T' + str(n_tasks) + \
              'n' + str(n_train) + 'val_perc' + str(val_perc) + 'dim' + str(n_dims)

    exp_parameters = locals()
    print('parameters ' + exp_name, exp_parameters)

    data_train, oracle_train = tasks_gen(n_tasks=n_tasks, n_train=n_train, n_test=n_test, n_dims=n_dims)
    data_valid, oracle_valid = tasks_gen(n_tasks=n_tasks_test, n_train=n_train, n_test=n_test, n_dims=n_dims)

    inner_solver_test_class = inner_solver_selector(inner_solver_test_str)

    def get_solvers(h_list, lmbd=0.0, gamma=0.0):
        return [inner_solver_test_class(lmbd, h, loss_f, grad_loss_f, gamma=gamma) for h in h_list]

    # Get eval loss for w = 0, w = w_\mu, w = \bar{w}
    inner_solvers = get_solvers([np.zeros(n_dims) for _ in range(n_tasks_test)])
    loss_inner_initial = no_train_evaluation(data_valid['X_test'], data_valid['Y_test'], inner_solvers, verbose=1)

    inner_solvers = get_solvers([oracle_valid['W_true'][:, i] for i in range(n_tasks_test)])
    loss_inner_oracle = no_train_evaluation(data_valid['X_test'], data_valid['Y_test'], inner_solvers, verbose=1)

    inner_solvers = get_solvers([oracle_valid['w_bar'] for _ in range(n_tasks_test)])
    loss_wbar = no_train_evaluation(data_valid['X_test'], data_valid['Y_test'], inner_solvers, verbose=1)

    # Evaluate losses for the oracle meta model h = \bar{w}
    inner_solver = inner_solver_test_class(lmbd, oracle_valid['w_bar'], loss_f, grad_loss_f, gamma=gamma)
    losses_oracle = LTL_evaluation(data_valid['X_train'], data_valid['Y_train'],
                                   data_valid['X_test'], data_valid['Y_test'], inner_solver, verbose=1)

    # Meta train and evaluation
    hs_dict = {}
    losses_ltl_dict = {}
    for is_str in inner_solver_str:
        inner_solver = inner_solver_selector(is_str)(lmbd, np.zeros(n_dims), loss_f, grad_loss_f, gamma=gamma)
        inner_solver_test = inner_solver_test_class(lmbd, np.zeros(n_dims), loss_f, grad_loss_f, gamma=gamma)
        hs, losses_ltl = meta_ssgd(alpha, data_train['X_train'], data_train['Y_train'], data_valid,
                                   inner_solver, inner_solver_test, eval_online=eval_online)
        hs_dict[is_str] = hs
        losses_ltl_dict[is_str] = losses_ltl

    # Evaluate losses for the itl case: starting from h = 0
    inner_solver = inner_solver_test_class(lmbd, np.zeros(n_dims), loss_f, grad_loss_f, gamma=gamma)
    losses_itl = LTL_evaluation(data_valid['X_train'], data_valid['Y_train'],
                                data_valid['X_test'], data_valid['Y_test'], inner_solver, verbose=1)
    print('hs :', hs)
    for i, h in enumerate(hs):
        print('h-%d dot w_bar:  %f' % (i, h @ oracle_valid['w_bar']))

    print_metric_mean_and_std(losses_itl, name=metric_name + " ITL")
    for is_str, losses_ltl in losses_ltl_dict.items():
        print_metric_mean_and_std(losses_ltl[-1], name=metric_name + is_str + " LTL")
    print_metric_mean_and_std(losses_oracle, name=metric_name + " Oracle")

    exp_dir_path = make_exp_dir(os.path.join(exp_dir, exp_str))
    plot_2fig(losses_ltl_dict, losses_itl, losses_oracle, loss_inner_initial, loss_inner_oracle,
              loss_wbar, '', y_label=y_label, title=exp_str,
              save_dir_path=exp_dir_path, show_plot=show_plot)

    save_exp_parameters(exp_parameters, exp_dir_path)
    np.savetxt(os.path.join(exp_dir_path, "wbar-oracle.csv"), loss_wbar, delimiter=",")
    np.savetxt(os.path.join(exp_dir_path, "inner-oracle.csv"), loss_inner_oracle, delimiter=",")
    np.savetxt(os.path.join(exp_dir_path, "zero-losses.csv"), loss_inner_initial, delimiter=",")
    for is_str, losses_ltl in losses_ltl_dict.items():
        np.savetxt(os.path.join(exp_dir_path, "ltl-"+is_str+".csv"), losses_ltl, delimiter=",")
    for is_str, hs in hs_dict.items():
        np.savetxt(os.path.join(exp_dir_path, "hs-"+is_str+".csv"), hs, delimiter=",")
    np.savetxt(os.path.join(exp_dir_path, "itl.csv"), losses_itl, delimiter=",")
    np.savetxt(os.path.join(exp_dir_path, "oracle.csv"), losses_itl, delimiter=",")

    return {'losses_itl': losses_itl, 'losses_oracle': losses_oracle, 'losses_ltl_dict': losses_ltl_dict,
            'hs_dict': hs_dict, 'wbar-oracle': loss_wbar, 'inner-oracle': loss_inner_oracle,
            'zero-losses': loss_inner_initial}


def train_and_evaluate(inner_solvers, data_train, data_val, name='', verbose=0):
    losses_train = []
    for i in range(len(inner_solvers)):
        losses_train.append(LTL_evaluation(data_train['X_train'], data_train['Y_train'],
                                           data_train['X_test'], data_train['Y_test'],
                                           inner_solvers[i], verbose=verbose))

    best_solver_idx = np.argmin(np.mean(np.concatenate([np.expand_dims(l, 0) for l in losses_train]), axis=1))
    print('best ' + name + ': ' + str(inner_solvers[best_solver_idx].lmbd))

    losses_val = LTL_evaluation(data_val['X_train'], data_val['Y_train'],
                                data_val['X_test'], data_val['Y_test'],
                                inner_solvers[best_solver_idx], verbose=verbose)
    return losses_val, inner_solvers[best_solver_idx]


def save_3d_csv(path, arr3d: np.ndarray, hyper_str=None):
    for i in range(arr3d.shape[1]):
        str = path + '-'
        if hyper_str:
            str += hyper_str[i]
        else:
            str += hyper_str[i]
        str += '.csv'

        np.savetxt(str, arr3d[:, i], delimiter=",")


def exp_selector(exp_str):
    if exp_str == 'exp1':
        return exp1, absolute_loss, grad_absolute_loss, 'AL', 'exp1'
    elif exp_str == 'exp2':
        return exp2, hinge_loss, grad_hinge_loss, 'Hinge', 'expclass'
    else:
        raise NotImplementedError('exp: {} not implemented'.format(exp_str))


def inner_solver_selector(solver_str):
    if solver_str == 'subgd':
        return InnerSubGD
    elif solver_str == 'ssubgd':
        return InnerSSubGD
    elif solver_str == 'fista':
        return FISTA
    else:
        raise NotImplementedError('inner solver {} not found'.format(solver_str))


def exp_gid_search(exp_str='exp1', seed=0, lambdas=np.logspace(-6, 3, num=10), alphas=(0.001, 0.01, 0.1, 1, 10, 100),
                   n_processes=30, y_snr=100, task_std=1, n_tasks=100, n_train=100, n_dims=30,
                   n_tasks_test=200, n_test=100, val_perc=0.0, inner_solver_str=('ssubgd', 'subgd'),
                   inner_solver_test_str='ssubgd', show_plot=True):
    from grid_search import HyperList, par_grid_search, find_best_config

    exp_parameters = locals()
    print('parameters ' + exp_str, exp_parameters)

    exp_f, loss_f, grad_loss_f, loss_name, task_gen_str = exp_selector(exp_str)
    inner_solver_test_class = inner_solver_selector(inner_solver_test_str)

    exp_str = 'grid_search' + exp_str + 'taskstd' + str(task_std) + 'y_snr' + str(y_snr) + 'is' + str(inner_solver_str) \
              + 'ist' + inner_solver_test_str + 'n' + str(n_train) + 'val_perc' + str(val_perc) + 'dim' + str(n_dims)

    exp_parameters = locals()
    print('parameters ' + exp_str, exp_parameters)

    exp_dir_path = make_exp_dir(exp_str)

    # hyperparameters for the grid search
    lambdas = HyperList(lambdas)
    alphas = HyperList(alphas)

    params = {'seed': seed, 'y_snr': y_snr, 'task_std': task_std, 'n_tasks': n_tasks, 'n_train': n_train,
              'n_dims': n_dims, 'alpha': alphas, 'lmbd': lambdas, 'gamma': None, 'n_tasks_test': n_tasks_test,
              'n_test': n_test, 'val_perc': val_perc, 'inner_solver_str': inner_solver_str,
              'inner_solver_test_str': inner_solver_test_str, 'show_plot': False, 'exp_dir': exp_dir_path}

    results = par_grid_search(params, exp_f, n_processes=n_processes)

    tasks_gen = gen.TasksGenerator(seed=seed + 1, task_std=task_std, y_snr=y_snr, val_perc=val_perc,
                                   tasks_generation=task_gen_str)
    data_test, oracle_test = tasks_gen(n_tasks=n_tasks_test, n_train=n_train, n_test=n_test, n_dims=n_dims)

    def itl_metric(res):
        return np.mean(res['losses_itl'])

    def oracle_metric(res):
        return np.mean(res['losses_oracle'])

    # Evaluate losses for the itl case: starting from h = 0
    best_itl, _ = find_best_config(itl_metric, results)
    inner_solver = inner_solver_test_class(best_itl['params']['lmbd'], np.zeros(n_dims), loss_f, grad_loss_f,
                                      gamma=best_itl['params']['gamma'])
    losses_itl = LTL_evaluation(data_test['X_train'], data_test['Y_train'],
                                data_test['X_test'], data_test['Y_test'], inner_solver, verbose=1)

    # Evaluate losses for the itl case: starting from h = 0
    best_oracle, _ = find_best_config(oracle_metric, results)
    inner_solver = inner_solver_test_class(best_oracle['params']['lmbd'], oracle_test['w_bar'], loss_f, grad_loss_f,
                                      gamma=best_oracle['params']['gamma'])
    losses_oracle = LTL_evaluation(data_test['X_train'], data_test['Y_train'],
                                   data_test['X_test'], data_test['Y_test'], inner_solver, verbose=1)


    hs_dict = {}
    losses_ltl_dict = {}
    for is_name in inner_solver_str:
        losses_ltl = np.zeros((n_tasks + 1, n_tasks_test))
        hs = np.zeros((n_tasks + 1, n_dims))
        for t in range(n_tasks + 1):
            def ltl_metric(res):
                return np.mean(res['losses_ltl_dict'][is_name][t])

            best_ltl, _ = find_best_config(ltl_metric, results)
            hs[t] = best_ltl['out']['hs_dict'][is_name][:t+1].mean(axis=0)
            inner_solver = inner_solver_test_class(best_ltl['params']['lmbd'], hs[t], loss_f, grad_loss_f,
                                              gamma=best_ltl['params']['gamma'])

            losses_ltl[t] = LTL_evaluation(X=data_test['X_train'], y=data_test['Y_train'],
                                           X_test=data_test['X_test'], y_test=data_test['Y_test'],
                                           inner_solver=inner_solver, verbose=1)

            print(str(t) + '-' + 'loss-test  : ', np.mean(losses_ltl[t]), np.std(losses_ltl[t]))

        losses_ltl_dict[is_name] = losses_ltl
        hs_dict[is_name] = hs

    # Get eval loss for w = 0, w = w_\mu, w = \bar{w}
    def get_solvers(h_list, lmbd=0.0, gamma=0.0):
        return [inner_solver_test_class(lmbd, h, loss_f, grad_loss_f, gamma=gamma) for h in h_list]

    inner_solvers = get_solvers([np.zeros(n_dims) for _ in range(n_tasks_test)])
    loss_inner_initial = no_train_evaluation(data_test['X_test'], data_test['Y_test'], inner_solvers, verbose=1)

    inner_solvers = get_solvers([oracle_test['W_true'][:, i] for i in range(n_tasks_test)])
    loss_inner_oracle = no_train_evaluation(data_test['X_test'], data_test['Y_test'], inner_solvers, verbose=1)

    inner_solvers = get_solvers([oracle_test['w_bar'] for _ in range(n_tasks_test)])
    loss_wbar = no_train_evaluation(data_test['X_test'], data_test['Y_test'], inner_solvers, verbose=1)

    ltl_hyper_str = '_'.join([h + str(best_ltl['params'][h]) for h in ['lmbd', 'alpha']])
    itl_hyper_str = '_'.join([h + str(best_itl['params'][h]) for h in ['lmbd']])
    oracle_hyper_str = '_'.join([h + str(best_oracle['params'][h]) for h in ['lmbd']])

    print_metric_mean_and_std(losses_itl, name=loss_name + itl_hyper_str + " ITL")
    for is_str, losses_ltl in losses_ltl_dict.items():
        print_metric_mean_and_std(losses_ltl[-1], name=loss_name + is_str + "-" + ltl_hyper_str + " LTL")
    print_metric_mean_and_std(losses_oracle, name=loss_name + oracle_hyper_str + " Oracle")

    plot_2fig(losses_ltl_dict, losses_itl, losses_oracle, loss_inner_initial, loss_inner_oracle, loss_wbar,
              '', y_label='test ' + loss_name + ' (mean and std over test tasks)',
              title='', save_dir_path=exp_dir_path, show_plot=show_plot)

    save_exp_parameters(exp_parameters, exp_dir_path)
    for is_str, losses_ltl in losses_ltl_dict.items():
        np.savetxt(os.path.join(exp_dir_path, "ltl-"+is_str+"-"+ltl_hyper_str+".csv"), losses_ltl, delimiter=",")
    for is_str, hs in hs_dict.items():
        np.savetxt(os.path.join(exp_dir_path, "hs-"+is_str+"-"+ltl_hyper_str+".csv"), hs, delimiter=",")
    np.savetxt(os.path.join(exp_dir_path, "itl-" + itl_hyper_str + ".csv"), losses_itl, delimiter=",")
    np.savetxt(os.path.join(exp_dir_path, "oracle-" + oracle_hyper_str + ".csv"), losses_itl, delimiter=",")
    np.savetxt(os.path.join(exp_dir_path, "wbar-oracle.csv"), loss_wbar, delimiter=",")
    np.savetxt(os.path.join(exp_dir_path, "inner-oracle.csv"), loss_inner_oracle, delimiter=",")
    np.savetxt(os.path.join(exp_dir_path, "zero-losses.csv"), loss_inner_initial, delimiter=",")

    return {'losses_itl': losses_itl, 'losses_oracle': losses_oracle, 'losses_ltl': losses_ltl, 'hs': hs,
            'wbar-oracle': loss_wbar, 'inner-oracle': loss_inner_oracle, 'zero-losses': loss_inner_initial}


def plot_2fig(metric_ltl, metric_itl, metric_oracle, metric_inner_initial=None, metric_inner_oracle=None,
              metric_wbar=None,
              use_valid_str='', y_label='', title='', save_dir_path=None, show_plot=True):
    plot(metric_ltl, metric_itl, metric_oracle, None, None, None,
         use_valid_str, y_label, title, save_dir_path, show_plot, 'lossT1000.png')

    plot(metric_ltl, metric_itl, metric_oracle, metric_inner_initial, metric_inner_oracle, metric_wbar,
         use_valid_str, y_label, title, save_dir_path, show_plot, 'loss_plus.png')


def grid_search_several_trials(exp_str='exp1', n_processes=10):
    for n_train in [10, 100, 200, 1000]:
        for tasks_std in [1, 2, 4]:
            for y_snr in [5, 10, 100]:
                exp_gid_search(exp_str=exp_str, n_train=n_train, task_std=tasks_std, y_snr=y_snr,
                               n_processes=n_processes)


if __name__ == '__main__':
    exp_gid_search('exp2', lambdas=[0.01, 0.1], alphas=[100, 10], y_snr=10, task_std=3, n_train=10, n_tasks=100)
    #grid_search_several_trials(exp_str='exp1', n_processes=30)
    #grid_search_several_trials(exp_str='exp2', n_processes=30)
    #exp1(seed=0, y_snr=100, task_std=2, n_tasks=100, n_train=10, n_dims=30, alpha=50,
    #     lmbd=0.5, gamma=None, n_tasks_test=200, n_test=100, val_perc=0.0, inner_solver_str=['ssubgd', 'subgd'],
    #     inner_solver_test_str='ssubgd', show_plot=True)
