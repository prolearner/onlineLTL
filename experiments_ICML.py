import argparse

import numpy as np
from numpy.linalg import  norm
import data_generator as gen
from algorithms import meta_ssgd, LTL_evaluation, no_train_evaluation, \
    lmbd_theory, lmbd_theory_meta, alpha_theory, inner_solver_selector
from plots import plot_2fig
from utils import print_metric_mean_and_std, save_nparray, make_exp_dir, save_exp_parameters
from losses import HingeLoss, AbsoluteLoss, Loss
import os


parser = argparse.ArgumentParser(description='LTL online numpy experiments')
parser.add_argument('--seed', type=int, default=0, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--n-processes', type=int, default=30, metavar='N',
                    help='n processes for grid search (default: 30)')


EXP_FOLDER = 'exps'


def print_similarities(h, w_bar, t):
    print('l2 dist h-%d, w_bar:  %f' % (t, norm(h - w_bar)))
    print('cos sim h-%d, w_bar:  %f' % (t, (h @ w_bar) / (norm(h) * norm(w_bar) + 1e-10)))


def exp_selector(exp_str, seed=0, task_std=1, y_snr=10, val_perc=0.5, w_bar=4, n_dims=30,
                 n_train_tasks=0, n_val_tasks=0):
    if exp_str == 'exp1':
        tasks_gen = gen.TasksGenerator(seed=seed, task_std=task_std, y_snr=y_snr, val_perc=val_perc, n_dims=n_dims,
                                       tasks_generation='exp1', w_bar=w_bar)
        exp_name = exp_str + 'w_bar' + str(w_bar) + 'taskstd' + str(task_std) + 'y_snr' + str(y_snr) + \
                   'dim' + str(n_dims)
        loss = AbsoluteLoss
    elif exp_str == 'exp2':
        tasks_gen = gen.TasksGenerator(seed=seed, task_std=task_std, y_snr=y_snr, val_perc=val_perc, n_dims=n_dims,
                                       tasks_generation='expclass', w_bar=w_bar)
        exp_name = exp_str + 'w_bar' + str(w_bar) + 'taskstd' + str(task_std) + 'y_snr' + str(y_snr) + \
                   'dim' + str(n_dims)
        loss = HingeLoss
    elif exp_str == 'school':
        tasks_gen = gen.RealDatasetGenerator(gen_f=gen.schools_data_gen, seed=seed, n_train_tasks=n_train_tasks,
                                             n_val_tasks=n_val_tasks,
                                             val_perc=val_perc)
        exp_name = 'expSchool' + 'n_tasks_val' + str(n_val_tasks) + 'n_tasks' + str(tasks_gen.n_tasks) \
                   + 'dim' + str(tasks_gen.n_dims)
        loss = AbsoluteLoss
    else:
        raise NotImplementedError('exp: {} not implemented'.format(exp_str))

    return loss, tasks_gen, exp_name


def exp(exp_name, tasks_gen, loss_class: Loss, alpha=0.1, lmbd=(0.01, 0.1), gamma=None,
        n_tasks=100, n_train=5, n_tasks_test=200, n_test=100, val_perc=0.0, show_plot=False,
        inner_solver_str=('ssubgd', 'subgd'), inner_solver_test_str='ssubgd', exp_dir='',
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
    loss_inner_initial = no_train_evaluation(data_valid['X_test'], data_valid['Y_test'], inner_solvers, verbose=verbose)

    if oracle_valid is not None:
        inner_solvers = get_solvers([oracle_valid['W_true'][:, i] for i in range(len(data_valid['Y_test']))])
        loss_inner_oracle = no_train_evaluation(data_valid['X_test'], data_valid['Y_test'], inner_solvers,
                                                verbose=verbose)

        inner_solvers = get_solvers([oracle_valid['w_bar'] for _ in range(len(data_valid['Y_test']))])
        loss_wbar = no_train_evaluation(data_valid['X_test'], data_valid['Y_test'], inner_solvers, verbose=verbose)

        # Evaluate losses for the oracle meta model h = \bar{w}
        inner_solver = inner_solver_test_class(lmbd_oracle, oracle_valid['w_bar'], loss_class, gamma=gamma)
        losses_oracle = LTL_evaluation(data_valid['X_train'], data_valid['Y_train'],
                                       data_valid['X_test'], data_valid['Y_test'], inner_solver, verbose=verbose)
    else:
        losses_oracle, loss_inner_oracle, loss_wbar = None, None, None

    # Meta train and evaluation
    hs_dict = {}
    losses_ltl_dict = {}
    for is_str in inner_solver_str:
        inner_solver = inner_solver_selector(is_str)(lmbd, np.zeros(tasks_gen.n_dims), loss_class, gamma=gamma)
        inner_solver_test = inner_solver_test_class(lmbd, np.zeros(tasks_gen.n_dims), loss_class, gamma=gamma)
        hs, losses_ltl = meta_ssgd(alpha, data_train['X_train'], data_train['Y_train'], data_valid,
                                   inner_solver, inner_solver_test, eval_online=eval_online)
        hs_dict[is_str] = hs
        losses_ltl_dict[is_str] = losses_ltl

    # Evaluate losses for the itl case: starting from h = 0
    inner_solver = inner_solver_test_class(lmbd_itl, np.zeros(tasks_gen.n_dims), loss_class, gamma=gamma)
    losses_itl = LTL_evaluation(data_valid['X_train'], data_valid['Y_train'],
                                data_valid['X_test'], data_valid['Y_test'], inner_solver, verbose=verbose)
    print('hs :', hs)
    if oracle_valid is not None:
        for i, h in enumerate(hs):
            print_similarities(h, oracle_valid['w_bar'], i)

    metric_name = loss_class.name
    print_metric_mean_and_std(losses_itl, name=metric_name + " ITL")
    print_metric_mean_and_std(losses_ltl_dict, name=metric_name + " LTL")
    print_metric_mean_and_std(loss_inner_initial, name=metric_name + " w = 0")

    if losses_oracle is not None:
        print_metric_mean_and_std(losses_oracle, name=metric_name + " Oracle")

    if save_res:
        exp_dir_path = make_exp_dir(os.path.join(exp_dir, exp_str))
        save_exp_parameters(exp_parameters, exp_dir_path)
        if loss_wbar is not None:
            np.savetxt(os.path.join(exp_dir_path, "wbar-oracle.csv"), loss_wbar, delimiter=",")
        if loss_inner_oracle is not None:
            np.savetxt(os.path.join(exp_dir_path, "inner-oracle.csv"), loss_inner_oracle, delimiter=",")

        np.savetxt(os.path.join(exp_dir_path, "zero-losses.csv"), loss_inner_initial, delimiter=",")
        save_nparray(losses_ltl_dict, 'ltl', exp_dir_path)
        save_nparray(hs_dict, 'hs', exp_dir_path)
        np.savetxt(os.path.join(exp_dir_path, "itl.csv"), losses_itl, delimiter=",")
        np.savetxt(os.path.join(exp_dir_path, "oracle.csv"), losses_itl, delimiter=",")
    else:
        exp_dir_path = None

    plot_2fig(losses_ltl_dict, losses_itl, losses_oracle, loss_inner_initial, loss_inner_oracle,
              loss_wbar, '', y_label='test' + metric_name + ' (mean and std over test tasks)',
              title='',
              save_dir_path=exp_dir_path, show_plot=show_plot)

    return {'losses_itl': losses_itl, 'losses_oracle': losses_oracle, 'losses_ltl_dict': losses_ltl_dict,
            'hs_dict': hs_dict, 'wbar-oracle': loss_wbar, 'inner-oracle': loss_inner_oracle,
            'zero-losses': loss_inner_initial, 'alpha': alpha, 'lmbd': lmbd, 'lmbd_itl': lmbd_itl,
            'lmbd_oracle': lmbd_oracle}


def exp_meta_val(exp_str='exp1', seed=0, lambdas=np.logspace(-6, 3, num=10), alphas=(0.01, 0.1, 1, 10, 100, 1000),
                 gamma=None, n_processes=30, w_bar=4, y_snr=100, task_std=1, n_tasks=100, n_train=100, n_dims=30,
                 n_tasks_test=200, n_test=100, val_perc=0.0, exp_dir=EXP_FOLDER, inner_solver_str=('ssubgd', 'subgd'),
                 use_hyper_bounds=False, inner_solver_test_str='ssubgd', show_plot=True, save_res=True, verbose=1):

    loss_class, tasks_gen, inner_exp_name = exp_selector(exp_str, seed=seed, task_std=task_std, y_snr=y_snr,
                                                   n_train_tasks=n_tasks, n_val_tasks=20, n_dims=n_dims,
                                                   val_perc=val_perc, w_bar=w_bar)

    params = {'exp_name': inner_exp_name, 'tasks_gen': tasks_gen, 'loss_class': loss_class, 'n_tasks': n_tasks,
              'n_train': n_train, 'alpha': alphas, 'lmbd': lambdas, 'gamma': gamma, 'n_tasks_test': n_tasks_test,
              'n_test': n_test, 'val_perc': val_perc, 'inner_solver_str': inner_solver_str,
              'inner_solver_test_str': inner_solver_test_str, 'show_plot': show_plot, 'exp_dir': exp_dir,
              'verbose': verbose, 'save_res': save_res}

    # execute single experiment if lambdas and alphas are not tuple:
    if (not hasattr(alphas, '__iter__')) and (not hasattr(lambdas, '__iter__')):
        return exp(**params)

    from grid_search import HyperList, par_grid_search, find_best_config

    exp_name = 'grid_search' + inner_exp_name + 'is' \
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
        return np.mean(res['losses_itl'])

    def oracle_metric(res):
        return np.mean(res['losses_oracle'])

    # Evaluate losses for the itl case: starting from h = 0
    best_itl, _ = find_best_config(itl_metric, results)
    inner_solver = inner_solver_test_class(best_itl['params']['lmbd'], np.zeros(tasks_gen.n_dims), loss_class,
                                           gamma=best_itl['params']['gamma'])
    losses_itl = LTL_evaluation(data_test['X_train'], data_test['Y_train'],
                                data_test['X_test'], data_test['Y_test'], inner_solver, verbose=verbose)

    # Evaluate losses for the oracle meta model h = \bar{w}
    if oracle_test is not None:
        best_oracle, _ = find_best_config(oracle_metric, results)
        inner_solver = inner_solver_test_class(best_oracle['params']['lmbd'], oracle_test['w_bar'], loss_class,
                                               gamma=best_oracle['params']['gamma'])
        losses_oracle = LTL_evaluation(data_test['X_train'], data_test['Y_train'],
                                       data_test['X_test'], data_test['Y_test'], inner_solver, verbose=verbose)
    else:
        losses_oracle, best_oracle = None, None

    hs_dict = {}
    losses_ltl_dict = {}
    for is_name in inner_solver_str:
        losses_ltl = np.zeros((n_tasks + 1, len(data_test['X_train'])))
        hs = np.zeros((n_tasks + 1, tasks_gen.n_dims))
        for t in range(n_tasks + 1):
            def ltl_metric(res):
                return np.mean(res['losses_ltl_dict'][is_name][t])

            best_ltl, _ = find_best_config(ltl_metric, results)
            hs[t] = best_ltl['out']['hs_dict'][is_name][:t + 1].mean(axis=0)
            inner_solver = inner_solver_test_class(best_ltl['params']['lmbd'], hs[t], loss_class,
                                                   gamma=best_ltl['params']['gamma'])

            losses_ltl[t] = LTL_evaluation(X=data_test['X_train'], y=data_test['Y_train'],
                                           X_test=data_test['X_test'], y_test=data_test['Y_test'],
                                           inner_solver=inner_solver, verbose=verbose)

            print(str(t) + '-' + 'loss-test  : ', np.mean(losses_ltl[t]), np.std(losses_ltl[t]))

            if oracle_test is not None:
                print_similarities(hs[t], oracle_test['w_bar'], t)

        losses_ltl_dict[is_name] = losses_ltl
        hs_dict[is_name] = hs

    # Get eval loss for w = 0, w = w_\mu, w = \bar{w}
    def get_solvers(h_list, lmbd=0.0, gamma=0.0):
        return [inner_solver_test_class(lmbd, h, loss_class, gamma=gamma) for h in h_list]

    inner_solvers = get_solvers([np.zeros(tasks_gen.n_dims) for _ in range(len(data_test['Y_test']))])
    loss_inner_initial = no_train_evaluation(data_test['X_test'], data_test['Y_test'], inner_solvers, verbose=verbose)

    if oracle_test is not None:
        inner_solvers = get_solvers([oracle_test['W_true'][:, i] for i in range(len(data_test['Y_test']))])
        loss_inner_oracle = no_train_evaluation(data_test['X_test'], data_test['Y_test'], inner_solvers,
                                                verbose=verbose)

        inner_solvers = get_solvers([oracle_test['w_bar'] for _ in range(len(data_test['Y_test']))])
        loss_wbar = no_train_evaluation(data_test['X_test'], data_test['Y_test'], inner_solvers, verbose=verbose)
    else:
        loss_wbar, loss_inner_oracle = None, None

    ltl_hyper_str = '_'.join([h + str(best_ltl['params'][h]) for h in ['lmbd', 'alpha']])
    itl_hyper_str = '_'.join([h + str(best_itl['params'][h]) for h in ['lmbd']])
    exp_parameters['best_ltl'] = best_ltl['params']
    exp_parameters['best_itl'] = best_itl['params']

    if best_oracle is not None:
        oracle_hyper_str = '_'.join([h + str(best_oracle['params'][h]) for h in ['lmbd']])
        losses_oracle_dict = {oracle_hyper_str: losses_oracle}
        exp_parameters['best_oracle'] = best_oracle['params']
    else:
        losses_oracle_dict = None

    losses_itl_dict = {itl_hyper_str: losses_itl}

    loss_name = loss_class.name

    plot_2fig(losses_ltl_dict, losses_itl_dict, losses_oracle_dict, loss_inner_initial, loss_inner_oracle, loss_wbar,
              '', y_label='test ' + loss_name + ' (mean and std over test tasks)',
              title='', save_dir_path=exp_dir_path, show_plot=show_plot, name='loss')

    # get theory hyperparams results
    if use_hyper_bounds and oracle_test is not None:
        params['use_hyper_bounds'] = True
        theory_result = exp(**params)

        ltl_hyper_str_theory = '_'.join([h + str(theory_result[h]) for h in ['lmbd', 'alpha']])
        itl_hyper_str_theory = '_'.join([h + str(theory_result[h]) for h in ['lmbd_itl']])
        oracle_hyper_str_theory = '_'.join([h + str(theory_result[h]) for h in ['lmbd_oracle']])

        print(ltl_hyper_str_theory, itl_hyper_str_theory, oracle_hyper_str_theory)

        theory_str = 'th'

        losses_oracle_dict[theory_str] = theory_result['losses_oracle']
        losses_itl_dict[theory_str] = theory_result['losses_itl']
        for is_name in inner_solver_str:
            losses_ltl_dict[is_name + '-' + theory_str] = theory_result['losses_ltl_dict'][is_name]
            losses_ltl_dict[is_name + '-' + ltl_hyper_str] = losses_ltl_dict[is_name]
            losses_ltl_dict.pop(is_name, None)

        plot_2fig(losses_ltl_dict, losses_itl_dict, losses_oracle_dict, loss_inner_initial, loss_inner_oracle,
                  loss_wbar, '', y_label='test ' + loss_name + ' (mean and std over test tasks)',
                  title='', save_dir_path=exp_dir_path, show_plot=show_plot, name='lossth')

    print_metric_mean_and_std(losses_itl_dict, name=loss_name + " ITL")
    print_metric_mean_and_std(losses_ltl_dict, name=loss_name + " LTL")
    print_metric_mean_and_std(loss_inner_initial, name=loss_name + " w = 0")
    if losses_oracle_dict is not None:
        print_metric_mean_and_std(losses_oracle_dict, name=loss_name + " Oracle")

    save_exp_parameters(exp_parameters, exp_dir_path)
    save_nparray(losses_ltl_dict, 'ltl', exp_dir_path)
    save_nparray(hs_dict, 'hs', exp_dir_path)
    save_nparray(losses_itl_dict, 'itl', exp_dir_path)
    if oracle_test is not None:
        save_nparray(losses_oracle_dict, 'oracle', exp_dir_path)
        np.savetxt(os.path.join(exp_dir_path, "wbar-oracle.csv"), loss_wbar, delimiter=",")
        np.savetxt(os.path.join(exp_dir_path, "inner-oracle.csv"), loss_inner_oracle, delimiter=",")
    np.savetxt(os.path.join(exp_dir_path, "zero-losses.csv"), loss_inner_initial, delimiter=",")

    return {'losses_itl_dict': losses_itl_dict, 'losses_oracle_dict': losses_oracle_dict,
            'losses_ltl_dict': losses_ltl_dict, 'hs_dict': hs_dict,
            'wbar-oracle': loss_wbar, 'inner-oracle': loss_inner_oracle, 'zero-losses': loss_inner_initial}


def grid_search_variance(exp_str='exp2', seed=0, n_processes=10):
    for w_bar in [16, 32]:
        for n_train in [10, 50, 200]:
            for tasks_std in [1, 2, 4, 8]:
                exp_meta_val(exp_str=exp_str, seed=seed, n_train=n_train, task_std=tasks_std, y_snr=10,
                             n_processes=n_processes, w_bar=w_bar, inner_solver_str=['ssubgd'],
                             use_hyper_bounds=True, n_tasks=1000, show_plot=False)


def school_meta_val(seed=0, lambdas=np.logspace(-3, 3, num=100), alphas=np.logspace(-4, 3, num=10),
                 gamma=None, n_processes=30, n_tasks=80, exp_dir=EXP_FOLDER, inner_solver_str=('ssubgd', 'subgd'),
                 use_hyper_bounds=False, inner_solver_test_str='ssubgd', show_plot=True, save_res=True, verbose=1):

    return exp_meta_val(exp_str='school', seed=seed, lambdas=lambdas, alphas=alphas,
                 gamma=gamma, n_processes=n_processes, w_bar=0, y_snr=0, task_std=0, n_tasks=n_tasks, n_train=0, n_dims=0,
                 n_tasks_test=0, n_test=0, val_perc=0.5, exp_dir=exp_dir, inner_solver_str=inner_solver_str,
                 use_hyper_bounds=use_hyper_bounds, inner_solver_test_str=inner_solver_test_str, show_plot=show_plot,
                 save_res=save_res, verbose=verbose)


if __name__ == '__main__':
    args = parser.parse_args()

    # exp_meta_val('school', y_snr=10, task_std=2, n_train=100, n_tasks=80, val_perc=0.5, n_processes=40,

    #             lambdas=np.logspace(-3, 3, num=10), alphas=np.logspace(-3, 3, num=10),
    #             inner_solver_str=['ssubgd'], w_bar=16, verbose=2, use_hyper_bounds=True)
    # grid_search_variance(exp_str='exp1', n_processes=30)
    # grid_search_several_trials(exp_str='exp2', n_processes=30)
    # exp2(seed=0, y_snr=10, task_std=2, n_tasks=100, n_train=50, n_dims=30, alpha=100, w_bar=16,
    #     lmbd=0.05, gamma=None, n_tasks_test=200, n_test=100, val_perc=0.0, inner_solver_str=['ssubgd'],
    #     use_hyper_bounds=False,
    #    inner_solver_test_str='ssubgd', show_plot=True)

    exp_meta_val('exp1', seed=args.seed, n_processes=args.n_processes, alphas=100, lambdas=0.01,
                 n_tasks=10, n_train=10)
    #school_meta_val(seed=args.seed, n_processes=args.n_processes, inner_solver_test_str='subgd')
