import argparse

import numpy as np
from numpy.linalg import  norm
import data_generator as gen
from algorithms import meta_ssgd, LTL_evaluation, no_train_evaluation, \
    lmbd_theory, lmbd_theory_meta, alpha_theory, inner_solver_selector
from plots import plot_2fig
from utils import print_metric_mean_and_std, save_nparray, make_exp_dir, save_exp_parameters
from losses import HingeLoss, AbsoluteLoss, Loss
from sklearn.metrics import explained_variance_score, accuracy_score
import copy
import os


parser = argparse.ArgumentParser(description='LTL online numpy experiments')
parser.add_argument('--seed', type=int, default=0, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--n-processes', type=int, default=30, metavar='N',
                    help='n processes for grid search (default: 30)')


EXP_FOLDER = 'exps'


def main():
    args = parser.parse_args()

    #exp_meta_val('exp2', y_snr=10, task_std=2, n_train=10, n_tasks=10, val_perc=0.5, n_processes=40,
    #             lambdas=np.logspace(-3, 3, num=10), alphas=np.logspace(-3, 3, num=10),
    #             inner_solver_str=['ssubgd'], w_bar=16, verbose=2, use_hyper_bounds=True)
    # grid_search_variance(exp_str='exp1', n_processes=30)
    # grid_search_several_trials(exp_str='exp2', n_processes=30)
    # exp2(seed=0, y_snr=10, task_std=2, n_tasks=100, n_train=50, n_dims=30, alpha=100, w_bar=16,
    #     lmbd=0.05, gamma=None, n_tasks_test=200, n_test=100, val_perc=0.0, inner_solver_str=['ssubgd'],
    #     use_hyper_bounds=False,
    #    inner_solver_test_str='ssubgd', show_plot=True)

    #exp_meta_val('exp1', seed=args.seed, n_processes=args.n_processes, alphas=K100, lambdas=0.01,
    #             n_tasks=10, n_train=10)
    #grid_search_variance('exp2', seed=args.seed, n_processes=args.n_processes)
    #school_meta_val(seed=args.seed, n_processes=args.n_processes, inner_solver_test_str='ssubgd', alphas=[0.01, 0.1],
    #                lambdas=[10, 1])
    exp1_school()


def exp_blank_ms():
    exp_multi_seed('exp1', n_train=10, n_tasks=10, w_bar=4, y_snr=1, task_std=1,
                   use_hyper_bounds=True, inner_solver_str=['ssubgd'])

def lenk_one():
    lenk_meta_val(lambdas=0.01, alphas=10)

def school_one():
    school_meta_val(lambdas=0.01, alphas=10)

def exp1_school():
    school_multi_seed(seeds=range(30))

def exp1_len():
    lenk_multi_seed()


def exp_reg_explore():
    exp_multi_seed('exp1', n_train=10, n_tasks=10, w_bar=4, y_snr=1, task_std=1,
                   use_hyper_bounds=True, inner_solver_str=['ssubgd'])


def exp_class():
    for tasks_std in [0.5, 1, 2, 4]:
        for n_train in [10, 50, 100]:
            exp_multi_seed('exp2', n_train=n_train, n_tasks=300, w_bar=4, y_snr=1, task_std=tasks_std,
                            use_hyper_bounds=True, inner_solver_str=['ssubgd'], search_oracle=True)


def exp_reg():
    for tasks_std in [0.5, 1, 2, 4]:
        for n_train in [10, 50, 100]:
            for n_tasks in [200, 1000]:
                exp_multi_seed('exp1', n_train=n_train, n_tasks=n_tasks, w_bar=4, y_snr=1, task_std=tasks_std,
                               use_hyper_bounds=True, inner_solver_str=['ssubgd'])


def exp_75():
    exp_multi_seed('exp1', n_train=10, n_tasks=1000, w_bar=4, y_snr=1, task_std=1,
                   use_hyper_bounds=True, inner_solver_str=['ssubgd'])
    exp_multi_seed('exp1', n_train=20, n_tasks=1000, w_bar=4, y_snr=1, task_std=1,
                   use_hyper_bounds=True, inner_solver_str=['ssubgd'])
    exp_multi_seed('exp1', n_train=50, n_tasks=1000, w_bar=4, y_snr=1, task_std=1,
                   use_hyper_bounds=True, inner_solver_str=['ssubgd'])


def select_exp(exp_str, seed=0, task_std=1, y_snr=10, val_perc=0.5, w_bar=4, n_dims=30,
               n_train_tasks=0, n_val_tasks=0):
    if exp_str == 'exp1':
        tasks_gen = gen.TasksGenerator(seed=seed, task_std=task_std, y_snr=y_snr, val_perc=val_perc, n_dims=n_dims,
                                       tasks_generation='exp1', w_bar=w_bar)
        exp_name = exp_str + 'w_bar' + str(w_bar) + 'taskstd' + str(task_std) + 'y_snr' + str(y_snr) + \
                   'dim' + str(n_dims)
        loss = AbsoluteLoss
        metric_dict = {}
        val_metric = 'loss'
    elif exp_str == 'exp2':
        tasks_gen = gen.TasksGenerator(seed=seed, task_std=task_std, y_snr=y_snr, val_perc=val_perc, n_dims=n_dims,
                                       tasks_generation='expclass', w_bar=w_bar)
        exp_name = exp_str + 'w_bar' + str(w_bar) + 'taskstd' + str(task_std) + 'y_snr' + str(y_snr) + \
                   'dim' + str(n_dims) + 'y_dist' + str(tasks_gen.y_dist)
        loss = HingeLoss
        val_metric = 'loss'
        metric_dict = {}
    elif exp_str == 'school':
        tasks_gen = gen.RealDatasetGenerator(gen_f=gen.schools_data_gen, seed=seed, n_train_tasks=n_train_tasks,
                                             n_val_tasks=n_val_tasks,
                                             val_perc=val_perc)
        exp_name = 'expSchool' + 'n_tasks_val' + str(n_val_tasks) + 'n_tasks' + str(tasks_gen.n_tasks) \
                   + 'dim' + str(tasks_gen.n_dims)
        loss = AbsoluteLoss
        val_metric = 'neg exp variance'
        metric_dict = {'neg exp variance': ne_exp_var}
    elif exp_str == 'lenk':
        tasks_gen = gen.RealDatasetGenerator(gen_f=gen.computer_data_gen,
                                             seed=seed, n_train_tasks=n_train_tasks,
                                             n_val_tasks=n_val_tasks,
                                             val_perc=val_perc)
        exp_name = 'expLenk' + 'n_tasks_train' + str(n_train_tasks) + 'n_tasks' + str(tasks_gen.n_tasks) \
                   + 'dim' + str(tasks_gen.n_dims)
        loss = HingeLoss
        val_metric = 'loss'
        metric_dict = {}
    else:
        raise NotImplementedError('exp: {} not implemented'.format(exp_str))

    return loss, tasks_gen, exp_name, metric_dict, val_metric


def exp(exp_name, tasks_gen, loss_class: Loss, alpha=0.1, lmbd=(0.01, 0.1), gamma=None,
        n_tasks=100, n_train=5, n_tasks_test=200, n_test=100, val_perc=0.0, search_oracle=None, show_plot=False,
        inner_solver_str=('ssubgd', 'subgd'), inner_solver_test_str='ssubgd', metric_dict={}, exp_dir='',
        use_hyper_bounds=False, verbose=1, eval_online=True, save_res=False):

    if use_hyper_bounds and hasattr(tasks_gen, 'rx') and hasattr(tasks_gen, 'w_bar'):
        lmbd = lmbd_theory_meta(rx=tasks_gen.rx, L=loss_class.L, sigma_bar=tasks_gen.sigma_h(tasks_gen.w_bar),
                                n=n_train)
        alpha = alpha_theory(rx=tasks_gen.rx, L=loss_class.L, w_bar=tasks_gen.w_bar, T=n_tasks, n=n_train)

        lmbd_oracle = lmbd_theory(rx=tasks_gen.rx, L=loss_class.L, sigma_h=tasks_gen.sigma_h(tasks_gen.w_bar),
                                  n=n_train)
        lmbd_itl = lmbd_theory(rx=tasks_gen.rx, L=loss_class.L, sigma_h=tasks_gen.sigma_h(np.zeros(tasks_gen.n_dims)),
                               n=n_train)
    else:
        lmbd_itl, lmbd_oracle = lmbd, lmbd

    exp_str = exp_name + 'is' + str(inner_solver_str) + 'ist' + inner_solver_test_str + \
              'hb' + str(use_hyper_bounds) + \
              'alpha' + str(alpha) + 'lmbd' + str(lmbd) + 'lmbditl' + str(lmbd_itl) + 'lmbdor' + str(lmbd_oracle) + \
              'T' + str(n_tasks) + 'n' + str(n_train) + 'val_perc' + str(val_perc) + 'dim' + str(tasks_gen.n_dims)

    exp_parameters = locals()
    print('parameters ' + exp_name, exp_parameters)

    data_train, oracle_train = tasks_gen(n_tasks=n_tasks, n_train=n_train, n_test=n_test, sel='train')
    data_valid, oracle_valid = tasks_gen(n_tasks=n_tasks_test, n_train=n_train, n_test=n_test, sel='val')

    inner_solver_test_class = inner_solver_selector(inner_solver_test_str)

    def get_solvers(h_list, lmbd=0.0, gamma=0.0):
        return [inner_solver_test_class(lmbd, h, loss_class, gamma=gamma) for h in h_list]

    # Get eval loss for w = 0, w = w_\mu, w = \bar{w}
    inner_solvers = get_solvers([np.zeros(tasks_gen.n_dims) for _ in range(len(data_valid['Y_test']))])
    m_inner_initial = no_train_evaluation(data_valid['X_test'],
                                          data_valid['Y_test'], inner_solvers,
                                          metric_dict=metric_dict, verbose=verbose)
    w_bar_mult = None
    if oracle_valid is not None:
        # Evaluate losses for the oracle meta model h = \bar{w}
        max_iter_search = 1 if not search_oracle else 10
        w_bar_mult = 1
        step = 10
        best_loss = np.Inf
        best_m_oracle = None
        best_w_mult = w_bar_mult
        for i in range(max_iter_search):
            inner_solver = inner_solver_test_class(lmbd_oracle, oracle_valid['w_bar']*w_bar_mult, loss_class, gamma=gamma)
            m_oracle = LTL_evaluation(data_valid['X_train'], data_valid['Y_train'],
                                      data_valid['X_test'], data_valid['Y_test'], inner_solver,
                                      metric_dict=metric_dict, verbose=verbose)
            current_loss = np.mean(m_oracle['loss'])
            if current_loss < best_loss:
                best_loss = current_loss
                best_m_oracle = m_oracle
                best_w_mult = w_bar_mult
                w_bar_mult += step
            else:
                step = step/2
                w_bar_mult -= step
                step = step/2

        m_oracle = best_m_oracle
        w_bar_mult = best_w_mult
        print('w_bar_mult', w_bar_mult)

        inner_solvers = get_solvers([oracle_valid['W_true'][:, i]*w_bar_mult for i in range(len(data_valid['Y_test']))])
        m_inner_oracle = no_train_evaluation(data_valid['X_test'], data_valid['Y_test'], inner_solvers,
                                             metric_dict=metric_dict, verbose=verbose)

        inner_solvers = get_solvers([oracle_valid['w_bar']*w_bar_mult for _ in range(len(data_valid['Y_test']))])
        m_wbar = no_train_evaluation(data_valid['X_test'], data_valid['Y_test'], inner_solvers,
                                     metric_dict=metric_dict, verbose=verbose)

    else:
        m_inner_oracle, m_oracle, m_wbar = None, None, None

    # Meta train and evaluation
    hs_dict = {}
    m_ltl_dict = {'loss': {}}
    for mn, _ in metric_dict.items():
        m_ltl_dict[mn] = {}
    for is_str in inner_solver_str:
        inner_solver = inner_solver_selector(is_str)(lmbd, np.zeros(tasks_gen.n_dims), loss_class, gamma=gamma)
        inner_solver_test = inner_solver_test_class(lmbd, np.zeros(tasks_gen.n_dims), loss_class, gamma=gamma)
        hs, m_ltl = meta_ssgd(alpha, data_train['X_train'], data_train['Y_train'], data_valid,
                                inner_solver, inner_solver_test, metric_dict=metric_dict, eval_online=eval_online)
        hs_dict[is_str] = hs
        for mn, res in m_ltl.items():
            m_ltl_dict[mn][is_str] = res

    # Evaluate losses for the itl case: starting from h = 0
    inner_solver = inner_solver_test_class(lmbd_itl, np.zeros(tasks_gen.n_dims), loss_class, gamma=gamma)
    m_itl = LTL_evaluation(data_valid['X_train'], data_valid['Y_train'],
                                data_valid['X_test'], data_valid['Y_test'], inner_solver,
                                       metric_dict=metric_dict, verbose=verbose)
    print('hs :', hs)
    if oracle_valid is not None:
        for i, h in enumerate(hs):
            print_similarities(h, oracle_valid['w_bar'], i)

    exp_dir_path = make_exp_dir(os.path.join(exp_dir, exp_str)) if save_res else None
    print_results(m_ltl_dict, m_itl, m_oracle, m_inner_initial)
    plot_results(exp_dir_path, m_ltl_dict, m_itl, m_oracle, m_inner_oracle, m_inner_initial, m_wbar,
                 show_plot=show_plot)
    save_results(exp_dir_path, exp_parameters, m_ltl_dict, m_itl, m_oracle, m_inner_oracle,
                 m_inner_initial, m_wbar, hs_dict)

    return {'hs_dict': hs_dict, 'alpha': alpha, 'lmbd': lmbd, 'lmbd_itl': lmbd_itl,
            'lmbd_oracle': lmbd_oracle, 'm_itl': m_itl, 'm_oracle': m_oracle, 'm_ltl_dict': m_ltl_dict,
            'm-wbar-oracle': m_wbar, 'm-inner-oracle': m_inner_oracle,
            'm-zero-losses': m_inner_initial, 'w_bar_mult': w_bar_mult}


def exp_meta_val(exp_str='exp1', seed=0, lambdas=np.logspace(-6, 3, num=10), alphas=np.logspace(-6, 3, num=10),
                 gamma=None, n_processes=30, w_bar=4, y_snr=100, task_std=1, n_tasks=100, n_train=100, n_dims=30,
                 n_tasks_test=200, n_test=100, val_perc=0.0, w_bar_mult=None, search_oracle=False,
                 exp_dir=EXP_FOLDER, inner_solver_str=('ssubgd', 'subgd'),
                 use_hyper_bounds=False, inner_solver_test_str='ssubgd', show_plot=True, save_res=True, verbose=1):

    loss_class, tasks_gen, inner_exp_name,\
        metric_dict, val_metric = select_exp(exp_str, seed=seed, task_std=task_std, y_snr=y_snr,
                                             n_train_tasks=n_tasks, n_val_tasks=n_tasks_test, n_dims=n_dims,
                                             val_perc=val_perc, w_bar=w_bar)

    params = {'exp_name': inner_exp_name, 'tasks_gen': tasks_gen, 'loss_class': loss_class, 'n_tasks': n_tasks,
              'n_train': n_train, 'alpha': alphas, 'lmbd': lambdas, 'gamma': gamma, 'n_tasks_test': n_tasks_test,
              'n_test': n_test, 'val_perc': val_perc, 'inner_solver_str': inner_solver_str,
              'search_oracle': search_oracle,
              'inner_solver_test_str': inner_solver_test_str, 'show_plot': show_plot,
              'metric_dict': metric_dict, 'exp_dir': exp_dir, 'verbose': verbose, 'save_res': save_res}

    # execute single experiment if lambdas and alphas are not tuple:
    if (not hasattr(alphas, '__iter__')) and (not hasattr(lambdas, '__iter__')):
        return exp(**params)

    from grid_search import HyperList, par_grid_search, find_best_config

    exp_name = 'grid_search' + inner_exp_name + 'seed' + str(seed) + 'is' \
               + str(inner_solver_str) + 'ist' + inner_solver_test_str + 'n' + str(n_train) + 'val_perc' + str(val_perc)

    exp_parameters = locals()
    print('parameters ' + exp_str, exp_parameters)

    inner_solver_test_class = inner_solver_selector(inner_solver_test_str)

    exp_dir_path = make_exp_dir(os.path.join(exp_dir, exp_name))

    # hyperparameters for the grid search
    params['lmbd'] = HyperList(lambdas)
    params['alpha'] = HyperList(alphas)

    # grid search over hyperparameters
    params['exp_dir'] = exp_dir_path
    params['show_plot'] = False
    params['save_res'] = False
    results = par_grid_search(params, exp, n_processes=n_processes)

    data_test, oracle_test = tasks_gen(n_tasks=n_tasks_test, n_train=n_train, n_test=n_test, sel='test')

    def itl_metric(res):
        return np.mean(res['m_itl'][val_metric])

    def oracle_metric(res):
        return np.mean(res['m_oracle'][val_metric])

    # Get eval loss for w = 0, w = w_\mu, w = \bar{w}
    def get_solvers(h_list, lmbd=0.0, gamma=0.0):
        return [inner_solver_test_class(lmbd, h, loss_class, gamma=gamma) for h in h_list]
    if oracle_test is not None:
        best_oracle, _ = find_best_config(oracle_metric, results)
        w_mult = best_oracle['out']['w_bar_mult']

        inner_solver = inner_solver_test_class(best_oracle['params']['lmbd'], oracle_test['w_bar']*w_mult, loss_class,
                                               gamma=best_oracle['params']['gamma'])
        m_oracle = LTL_evaluation(data_test['X_train'], data_test['Y_train'],
                                       data_test['X_test'], data_test['Y_test'], inner_solver,
                                                 metric_dict=metric_dict,
                                                 verbose=verbose)
        inner_solvers = get_solvers([oracle_test['W_true'][:, i]*w_mult for i in range(len(data_test['Y_test']))])
        m_inner_oracle = no_train_evaluation(data_test['X_test'], data_test['Y_test'], inner_solvers,
                                                   metric_dict=metric_dict,
                                                   verbose=verbose)

        inner_solvers = get_solvers([oracle_test['w_bar']*w_mult for _ in range(len(data_test['Y_test']))])
        m_wbar = no_train_evaluation(data_test['X_test'], data_test['Y_test'], inner_solvers,
                                           metric_dict=metric_dict,
                                           verbose=verbose)
    else:
        m_oracle, m_wbar, m_inner_oracle, best_oracle = None, None, None, None

    inner_solvers = get_solvers([np.zeros(tasks_gen.n_dims) for _ in range(len(data_test['Y_test']))])
    m_inner_initial = no_train_evaluation(data_test['X_test'], data_test['Y_test'],
                                                              inner_solvers,
                                                              metric_dict=metric_dict,
                                                              verbose=verbose)

    hs_dict = {}
    m_ltl_dict = {}
    for mn in list(metric_dict) + ['loss']:
        m_ltl_dict[mn] = {}
        for is_name in inner_solver_str:
            m_ltl_dict[mn][is_name] = np.zeros((n_tasks + 1, len(data_test['X_train'])))

    for is_name in inner_solver_str:
        hs = np.zeros((n_tasks + 1, tasks_gen.n_dims))
        for t in range(n_tasks + 1):
            def ltl_metric(res):
                return np.mean(res['m_ltl_dict'][val_metric][is_name][t])

            best_ltl, _ = find_best_config(ltl_metric, results)
            hs[t] = best_ltl['out']['hs_dict'][is_name][:t + 1].mean(axis=0)
            inner_solver = inner_solver_test_class(best_ltl['params']['lmbd'], hs[t], loss_class,
                                                   gamma=best_ltl['params']['gamma'])

            m_ltl = LTL_evaluation(X=data_test['X_train'], y=data_test['Y_train'],
                                           X_test=data_test['X_test'], y_test=data_test['Y_test'],
                                           metric_dict=metric_dict,
                                           inner_solver=inner_solver, verbose=verbose)
            for mn, res in m_ltl.items():
                m_ltl_dict[mn][is_name][t] = res
                print(str(t) + '-' + mn+'-test  : ', np.mean(res), np.std(res))

            if oracle_test is not None:
                print_similarities(hs[t], oracle_test['w_bar'], t)

        hs_dict[is_name] = hs

    # Evaluate losses for the itl case: starting from h = 0
    best_itl, _ = find_best_config(itl_metric, results)
    inner_solver = inner_solver_test_class(best_itl['params']['lmbd'], np.zeros(tasks_gen.n_dims), loss_class,
                                           gamma=best_itl['params']['gamma'])
    m_itl = LTL_evaluation(data_test['X_train'], data_test['Y_train'],
                                       data_test['X_test'], data_test['Y_test'], inner_solver,
                                       metric_dict=metric_dict,
                                       verbose=verbose)

    # ltl_hyper_str = '_'.join([h + str(best_ltl['params'][h]) for h in ['lmbd', 'alpha']])
    # itl_hyper_str = '_'.join([h + str(best_itl['params'][h]) for h in ['lmbd']])
    exp_parameters['best_ltl'] = best_ltl['params']
    exp_parameters['best_itl'] = best_itl['params']

    if best_oracle is not None:
        # oracle_hyper_str = '_'.join([h + str(best_oracle['params'][h]) for h in ['lmbd']])
        m_oracle_dict = {}
        for mn in m_itl:
            m_oracle_dict[mn] = {'': m_oracle[mn]}
        exp_parameters['best_oracle'] = best_oracle['params']
    else:
        m_oracle_dict = None

    m_itl_dict = {}
    for mn in m_itl:
        m_itl_dict[mn] = {'': m_itl[mn]}

    plot_results(exp_dir_path, m_ltl_dict, m_itl_dict, m_oracle, m_inner_oracle, m_inner_initial, m_wbar,
                 show_plot=show_plot)

    # get theory hyperparams results
    if use_hyper_bounds and oracle_test is not None:
        params['use_hyper_bounds'] = True
        theory_result = exp(**params)

        ltl_hyper_str_theory = '_'.join([h + str(theory_result[h]) for h in ['lmbd', 'alpha']])
        itl_hyper_str_theory = '_'.join([h + str(theory_result[h]) for h in ['lmbd_itl']])
        oracle_hyper_str_theory = '_'.join([h + str(theory_result[h]) for h in ['lmbd_oracle']])

        print(ltl_hyper_str_theory, itl_hyper_str_theory, oracle_hyper_str_theory)

        theory_str = 'th'
        for mn in m_itl:
            m_oracle_dict[mn][theory_str] = theory_result['m_oracle'][mn]
            m_itl_dict[mn][theory_str] = theory_result['m_itl'][mn]
            for is_name in inner_solver_str:
                m_ltl_dict[mn][is_name + '-' + theory_str] = theory_result['m_ltl_dict'][mn][is_name]
                m_ltl_dict[mn][is_name] = m_ltl_dict[mn][is_name]

        plot_results(exp_dir_path, m_ltl_dict, m_itl_dict, m_oracle, m_inner_oracle, m_inner_initial, m_wbar,
                     name='th', show_plot=show_plot)

    print_results(m_ltl_dict, m_itl, m_oracle, m_inner_initial)
    save_results(exp_dir_path, exp_parameters, m_ltl_dict, m_itl, m_oracle, m_inner_oracle,
                 m_inner_initial, m_wbar, hs_dict)

    return {'m_itl_dict': m_itl_dict, 'm_oracle_dict': m_oracle_dict,
            'm_ltl_dict': m_ltl_dict, 'hs_dict': hs_dict,
            'm_wbar': m_wbar, 'm_inner_oracle': m_inner_oracle, 'm_inner_initial': m_inner_initial}


def exp_multi_seed(exp_str='exp1', seeds=list(range(10)), lambdas=np.logspace(-6, 3, num=10), alphas=np.logspace(-6, 3, num=10),
                 gamma=None, n_processes=30, w_bar=4, y_snr=100, task_std=1, n_tasks=100, n_train=100, n_dims=30,
                 n_tasks_test=200, n_test=100, val_perc=0.0, search_oracle=False,
                 exp_dir=EXP_FOLDER, inner_solver_str=('ssubgd', 'subgd'),
                 use_hyper_bounds=False, inner_solver_test_str='ssubgd', show_plot=True, save_res=True, verbose=1):

    loss_class, tasks_gen, inner_exp_name,\
        metric_dict, val_metric = select_exp(exp_str, seed=seeds[0], task_std=task_std, y_snr=y_snr,
                                             n_train_tasks=n_tasks, n_val_tasks=n_tasks_test, n_dims=n_dims,
                                             val_perc=val_perc, w_bar=w_bar)

    exp_name = 'grid_search' + inner_exp_name + 'over' + str(len(seeds)) + 'seeds' + 'is' \
               + str(inner_solver_str) + 'ist' + inner_solver_test_str + 'n' + str(n_train) + 'val_perc' + str(val_perc)

    exp_parameters = locals()
    print('parameters ' + exp_str, exp_parameters)
    exp_dir_path = make_exp_dir(os.path.join(exp_dir, exp_name))

    metrics = {'m_itl_dict': [], 'm_oracle_dict': [], 'm_ltl_dict': [], 'm_inner_oracle': [], 'm_wbar': [],
               'm_inner_initial': [], 'hs_dict': []}

    def concat(array_of_np_arrays):
        return np.concatenate(array_of_np_arrays)

    def means(array_of_np_arrays):
        new_obj = []
        for a in array_of_np_arrays:
            new_obj.append(np.mean(a))
        if len(new_obj) > 1:
            return np.concatenate([np.expand_dims(o, 0) for o in new_obj])
        else:
            return new_obj[0]

    comb_dict = {'sm': means, 's': concat}

    def get_combination(array_of_obj, comb_f=means):
        new_obj = None
        obj = array_of_obj[0]
        if isinstance(obj, dict):
            new_obj = {}
            for k, v in obj.items():
                if v is not None:
                    new_obj[k] = get_combination([a[k] for a in array_of_obj], comb_f)
                else:
                    new_obj[k] = None

        elif isinstance(obj, list):
            new_obj = []
            for i, v in enumerate(obj):
                if v is not None:
                    new_obj[i] = get_combination([a[i] for a in array_of_obj], comb_f)
                else:
                    new_obj[i] = None
        elif isinstance(obj, np.ndarray) and len(obj.shape) > 1:
            new_obj = []
            for i in range(obj.shape[0]):
                new_obj.append(get_combination([a[i] for a in array_of_obj], comb_f))
            new_obj = np.concatenate([np.expand_dims(o, 0) for o in new_obj])
        elif isinstance(obj, np.ndarray) and len(obj.shape) == 1:
            new_obj = comb_f(array_of_obj)

        return new_obj

    for i, s in enumerate(seeds):
        r = exp_meta_val(exp_str=exp_str, seed=s, lambdas=lambdas, alphas=alphas, gamma=gamma, n_processes=n_processes,
                         w_bar=w_bar, y_snr=y_snr, task_std=task_std, n_tasks=n_tasks, n_train=n_train, n_dims=n_dims,
                         n_tasks_test=n_tasks_test, n_test=n_test, val_perc=val_perc, exp_dir=exp_dir_path,
                         inner_solver_str=inner_solver_str, use_hyper_bounds=use_hyper_bounds, inner_solver_test_str=
                         inner_solver_test_str, search_oracle=search_oracle, show_plot=show_plot, save_res=save_res, verbose=verbose)

        metrics['m_itl_dict'].append(r['m_itl_dict'])
        metrics['m_oracle_dict'].append(r['m_oracle_dict'])
        metrics['m_ltl_dict'].append(r['m_ltl_dict'])
        metrics['m_inner_oracle'].append(r['m_inner_oracle'])
        metrics['m_wbar'].append(r['m_wbar'])
        metrics['m_inner_initial'].append(r['m_inner_initial'])
        metrics['hs_dict'].append(r['hs_dict'])

        # compute average
        if i > 0:
            for comb_key, comb_f in comb_dict.items():
                avg_metrics = {}
                for k, m in metrics.items():
                        avg_metrics[k] = get_combination(m, comb_f=comb_f)

                print(avg_metrics)

                m_itl_dict = avg_metrics['m_itl_dict']
                m_oracle_dict = avg_metrics['m_oracle_dict']
                m_ltl_dict = avg_metrics['m_ltl_dict']
                m_inner_oracle = avg_metrics['m_inner_oracle']
                m_wbar = avg_metrics['m_wbar']
                hs_dict = avg_metrics['hs_dict']
                m_inner_initial = avg_metrics['m_inner_initial']

                theory_str = 'th' if m_oracle_dict is not None else ''
                y_label_add = '(mean and std over {} run)'.format(len(metrics['m_itl_dict']))
                print_results(m_ltl_dict, m_itl_dict, m_oracle_dict, m_inner_initial)
                plot_results(exp_dir_path, m_ltl_dict, m_itl_dict, m_oracle_dict, m_inner_oracle, m_inner_initial, m_wbar,
                             name=comb_key+theory_str, show_plot=show_plot, y_label_add=y_label_add)
                save_results(exp_dir_path, exp_parameters, m_ltl_dict, m_itl_dict, m_oracle_dict, m_inner_oracle,
                             m_inner_initial, m_wbar, hs_dict, add_str=comb_key)

                if m_oracle_dict is not None:
                    for mn in avg_metrics['m_itl_dict']:
                        m_oracle_dict[mn].pop(theory_str, None)
                        m_itl_dict[mn].pop(theory_str, None)
                        for is_name in inner_solver_str:
                            m_ltl_dict[mn].pop(is_name + '-' + theory_str, None)

                    plot_results(exp_dir_path, m_ltl_dict, m_itl_dict, m_oracle_dict, m_inner_oracle, m_inner_initial, m_wbar,
                                 name=comb_key, show_plot=show_plot, y_label_add=y_label_add)


def school_meta_val(seed=0, lambdas=np.logspace(-3, 3, num=100), alphas=np.logspace(-1, 6, num=10),
                 gamma=None, n_processes=30, n_tasks=75, n_val_tasks=25, exp_dir=EXP_FOLDER, inner_solver_str=('ssubgd', 'subgd'),
                 use_hyper_bounds=False, inner_solver_test_str='ssubgd', show_plot=True, save_res=True, verbose=1):

    return exp_meta_val(exp_str='school', seed=seed, lambdas=lambdas, alphas=alphas,
                 gamma=gamma, n_processes=n_processes, w_bar=0, y_snr=0, task_std=0, n_tasks=n_tasks, n_train=0, n_dims=0,
                 n_tasks_test=n_val_tasks, n_test=0, val_perc=0.5, exp_dir=exp_dir, inner_solver_str=inner_solver_str,
                 use_hyper_bounds=use_hyper_bounds, inner_solver_test_str=inner_solver_test_str, show_plot=show_plot,
                 save_res=save_res, verbose=verbose)


def school_multi_seed(seeds=list(range(10)), lambdas=np.logspace(-3, 3, num=100), alphas=np.logspace(-1, 6, num=10),
                 gamma=None, n_processes=30, n_tasks=75, n_val_tasks=25, exp_dir=EXP_FOLDER, inner_solver_str=('ssubgd', 'subgd'),
                 use_hyper_bounds=False, inner_solver_test_str='ssubgd', show_plot=True, save_res=True, verbose=1):

    return exp_multi_seed(exp_str='school', seeds=seeds, lambdas=lambdas, alphas=alphas,
                 gamma=gamma, n_processes=n_processes, w_bar=0, y_snr=0, task_std=0, n_tasks=n_tasks, n_train=0, n_dims=0,
                 n_tasks_test=n_val_tasks, n_test=0, val_perc=0.5, exp_dir=exp_dir, inner_solver_str=inner_solver_str,
                 use_hyper_bounds=use_hyper_bounds, inner_solver_test_str=inner_solver_test_str, show_plot=show_plot,
                 save_res=save_res, verbose=verbose)


def lenk_meta_val(seed=0, lambdas=np.logspace(-3, 3, num=100), alphas=np.logspace(-1, 6, num=10),
                 gamma=None, n_processes=30, n_tasks=100, exp_dir=EXP_FOLDER, inner_solver_str=('ssubgd', 'subgd'),
                 use_hyper_bounds=False, inner_solver_test_str='ssubgd', show_plot=True, save_res=True, verbose=1):

    return exp_meta_val(exp_str='school', seed=seed, lambdas=lambdas, alphas=alphas,
                 gamma=gamma, n_processes=n_processes, w_bar=0, y_snr=0, task_std=0, n_tasks=n_tasks, n_train=0, n_dims=0,
                 n_tasks_test=0, n_test=0, val_perc=0.5, exp_dir=exp_dir, inner_solver_str=inner_solver_str,
                 use_hyper_bounds=use_hyper_bounds, inner_solver_test_str=inner_solver_test_str, show_plot=show_plot,
                 save_res=save_res, verbose=verbose)


def lenk_multi_seed(seeds=list(range(10)), lambdas=np.logspace(-3, 3, num=100), alphas=np.logspace(-1, 6, num=10),
                 gamma=None, n_processes=30, n_tasks=100, exp_dir=EXP_FOLDER, inner_solver_str=('ssubgd', 'subgd'),
                 use_hyper_bounds=False, inner_solver_test_str='ssubgd', show_plot=True, save_res=True, verbose=1):

    return exp_multi_seed(exp_str='school', seeds=seeds, lambdas=lambdas, alphas=alphas,
                 gamma=gamma, n_processes=n_processes, w_bar=0, y_snr=0, task_std=0, n_tasks=n_tasks, n_train=0, n_dims=0,
                 n_tasks_test=0, n_test=0, val_perc=0.5, exp_dir=exp_dir, inner_solver_str=inner_solver_str,
                 use_hyper_bounds=use_hyper_bounds, inner_solver_test_str=inner_solver_test_str, show_plot=show_plot,
                 save_res=save_res, verbose=verbose)

################### UTILS ###############################


def plot_results(exp_dir_path, m_ltl_dict, m_itl, m_oracle, m_inner_oracle, m_inner_initial, m_wbar, name='',
                 y_label_add=' (mean and std over test tasks)',
                 show_plot=False):
    for mn in m_itl:
        plot_2fig(m_ltl_dict[mn], m_itl[mn], dg(m_oracle, mn), dg(m_inner_initial, mn), dg(m_inner_oracle, mn),
                  dg(m_wbar, mn), '', y_label='test ' + mn + y_label_add,
                  title='', name=mn +name,
                  save_dir_path=exp_dir_path, show_plot=show_plot)


def print_results(m_ltl_dict, m_itl, m_oracle, m_inner_initial):
    for mn in m_itl:
        print_metric_mean_and_std(m_itl[mn], name=mn + " ITL")
        print_metric_mean_and_std(m_ltl_dict[mn], name=mn + " LTL")
        print_metric_mean_and_std(m_inner_initial[mn], name=mn + " w = 0")

        if m_oracle is not None:
            print_metric_mean_and_std(m_oracle[mn], name=mn + " Oracle")


def save_results(exp_dir_path, exp_parameters, m_ltl_dict, m_itl, m_oracle, m_inner_oracle, m_inner_initial,
                 m_wbar, hs_dict, add_str=''):
    if exp_dir_path is None:
        return

    add_str = add_str + '-'
    save_exp_parameters(exp_parameters, exp_dir_path)
    save_nparray(hs_dict, add_str+'hs', exp_dir_path)
    for mn in m_itl:
        if m_wbar is not None:
            np.savetxt(os.path.join(exp_dir_path, add_str+mn+"-wbar-oracle.csv"), m_wbar[mn], delimiter=",")
        if m_inner_oracle is not None:
            np.savetxt(os.path.join(exp_dir_path,add_str+mn+"-inner-oracle.csv"), m_inner_oracle[mn], delimiter=",")

        np.savetxt(os.path.join(exp_dir_path, add_str+mn+"-zero.csv"), m_inner_initial[mn], delimiter=",")
        save_nparray(m_ltl_dict[mn], add_str+mn+'-ltl', exp_dir_path)
        save_nparray(m_itl[mn], add_str+mn+'-itl', exp_dir_path)
        if m_oracle is not None:
            save_nparray(m_oracle[mn], add_str+mn+"-oracle", exp_dir_path)


def print_similarities(h, w_bar, t):
    print('l2 dist h-%d, w_bar:  %f' % (t, norm(h - w_bar)))
    print('cos sim h-%d, w_bar:  %f' % (t, (h @ w_bar) / (norm(h) * norm(w_bar) + 1e-10)))


def dg(d, k): return None if d is None else d[k]


def accuracy(y_true, y_pred):
    return accuracy_score(y_true.astype(int), y_pred.astype(int))


def ne_exp_var(y_true, y_pred):
    return - explained_variance_score(y_true, y_pred)



if __name__ == '__main__':
    main()

