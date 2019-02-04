import numpy as np
import os

from algorithms import no_train_evaluation, LTL_evaluation, meta_ssgd, eval_biases
from grid_search import HyperList, par_grid_search, find_best_config
from plots import plot
from utils import save_nparray, make_exp_dir, print_metric_mean_and_std, save_exp_parameters

EXP_FOLDER = 'exps'


class Results:
    @staticmethod
    def get_from_results(results):
        return Results(results.save_dir, results.do_plot, results.show_plot, results.name)

    def __init__(self, save_dir=None, do_plot=True, show_plot=False, name='res'):
        self.save_dir = save_dir
        #if save_dir is not None:
        #    self.save_dir = make_exp_dir(os.path.join(save_dir, name), time=True)

        self.do_plot = do_plot
        self.show_plot = show_plot
        self.name = name
        self.metrics = None
        self.hs = None
        self.parameters = None
        self.color = None

    def add_parameters(self, params):
        self.parameters = params
        save_exp_parameters(params, self.save_dir, name=self.name+'-params')

    def add_metrics(self, metric_dict, y_label='Test Error'):
        self.metrics = metric_dict
        for m_name, metric in metric_dict.items():
            self._save_metric(metric, m_name, y_label)

    def add_hs(self, hs):
        self.hs = hs
        c_name = 'hs-'+self.name
        if self.save_dir is not None:
            save_nparray(hs, c_name, self.save_dir)

    def _save_metric(self, metric, metric_name='loss', y_label='Test Error'):
        c_name = metric_name+'-'+self.name
        print_metric_mean_and_std(metric, c_name)
        if self.do_plot and (self.save_dir is not None or self.show_plot):
            plot(metric, None, None, None, None, None,
                 '', y_label, c_name, self.save_dir, self.show_plot, self.name)

        if self.save_dir is not None:
            save_nparray(metric, c_name, self.save_dir)


def get_name_meta(loss_class, data_valid, lmbd, inner_solver_test_class, data_train, alpha, inner_solver_class):
    dim = data_train['X_train'][0].shape[1]
    n = data_train['X_train'][0].shape[0]
    n_eval = data_train['X_val'][0].shape[0]
    n_tasks = len(data_train['Y_train'])
    n_tasks_eval = len(data_valid['Y_train'])
    return 'loss' + loss_class.name + 'is' + inner_solver_class.name + 'ist' + inner_solver_test_class.name + \
              'alpha' + str(alpha) + 'lmbd' + str(lmbd) + 'T' + str(n_tasks) + 'n' + str(n) + \
              'dim' + str(dim) + 'taskseval' + str(n_tasks_eval) + 'neval' + str(n_eval)


def get_name(loss_class, data_valid, lmbd, inner_solver_test_class):
    dim = data_valid['X_train'][0].shape[1]
    n = data_valid['X_train'][0].shape[0]
    n_eval = data_valid['X_val'][0].shape[0]
    n_tasks_eval = len(data_valid['Y_train'])
    return 'loss' + loss_class.name + 'ist' + inner_solver_test_class.name + \
            'lmbd' + str(lmbd) + 'n' + str(n) + 'dim' + str(dim) + 'taskseval' + str(n_tasks_eval) + 'neval'\
            + str(n_eval)


def eval_models(w_list, inner_solver_test_class, loss_class, data_valid, metric_dict, results: Results, verbose=1):
    inner_solvers = [inner_solver_test_class(0.0, w, loss_class, gamma=0.0) for w in w_list]
    metrics = no_train_evaluation(data_valid['X_test'], data_valid['Y_test'], inner_solvers,
                               metric_dict=metric_dict, verbose=verbose)
    results.add_metrics(metrics)
    return results


def eval_bias(h, lmbd, gamma, inner_solver_test_class, loss_class, data_valid, metric_dict, results: Results,
              verbose=1, is_parallel=False):
    inner_solver = inner_solver_test_class(lmbd, h, loss_class, gamma=gamma)
    metrics = LTL_evaluation(data_valid['X_train'], data_valid['Y_train'], data_valid['X_test'], data_valid['Y_test'],
                             inner_solver, metric_dict=metric_dict, verbose=verbose)

    if is_parallel and results.save_dir is not None:
        results = Results.get_from_results(results)
        exp_str = get_name(loss_class, data_valid, lmbd, inner_solver_test_class)
        results.save_dir = make_exp_dir(os.path.join(results.save_dir, exp_str), time=False)

    results.add_metrics(metrics)
    return results


def meta_train_eval(h0, alpha, lmbd, gamma, inner_solver_class, inner_solver_test_class, loss_class, data_train,
                    data_valid, metric_dict, results: Results, verbose=1, is_parallel=False):
    inner_solver = inner_solver_class(lmbd, h0, loss_class, gamma=gamma)
    inner_solver_test = inner_solver_test_class(lmbd, h0, loss_class, gamma=gamma)
    hs, metrics = meta_ssgd(alpha, data_train['X_train'], data_train['Y_train'], data_valid,
                            inner_solver, inner_solver_test, metric_dict=metric_dict, eval_online=True, verbose=verbose)

    if is_parallel and results.save_dir is not None:
        results = Results.get_from_results(results)
        exp_str = get_name_meta(loss_class, data_valid, lmbd, inner_solver_test_class, data_train, alpha,
                                inner_solver_class)
        results.save_dir = make_exp_dir(os.path.join(results.save_dir, exp_str), time=False)

    results.add_hs(hs)
    results.add_metrics(metrics)
    return results


def val(h, val_metric, lambdas, gamma, inner_solver_test_class, loss_class,
        data_valid, data_test, metric_dict, results: Results, n_processes=30, verbose=1):

    lambdas = HyperList(lambdas)
    inner_results = Results(None, False, False, name=results.name)

    params = {'h': h, 'lmbd':lambdas, 'gamma': gamma,
              'inner_solver_test_class': inner_solver_test_class, 'loss_class': loss_class,
              'data_valid': data_valid, 'metric_dict': metric_dict, 'results': inner_results, 'verbose': verbose,
              'is_parallel': True}

    results_grid = par_grid_search(params, eval_bias, n_processes=n_processes)

    def metric(res: Results):
        return np.mean(res.metrics[val_metric])

    best, _ = find_best_config(metric, results_grid)

    results = eval_bias(h, best['params']['lmbd'], best['params']['gamma'],inner_solver_test_class,loss_class,
                     data_test, metric_dict, results, verbose=verbose)

    results.add_parameters({'lmbd': best['params']['lmbd'], 'gamma': best['params']['gamma']})
    return results


def meta_val(val_metric, h0, alphas, lambdas, gamma, inner_solver_class, inner_solver_test_class,loss_class,
             data_train, data_valid, data_test, metric_dict, results: Results, n_processes=30, verbose=1):

    alphas = HyperList(alphas)
    lambdas = HyperList(lambdas)
    T = len(data_train['Y_train'])
    dim = data_train['X_train'][0].shape[1]

    inner_results = Results(None, False, False, name=results.name)

    params = {'h0': h0, 'alpha': alphas, 'lmbd':lambdas, 'gamma':gamma, 'inner_solver_class':inner_solver_class,
              'inner_solver_test_class': inner_solver_test_class, 'loss_class': loss_class, 'data_train':data_train,
              'data_valid':data_valid, 'metric_dict': metric_dict, 'results': inner_results, 'verbose': verbose,
              'is_parallel':True}

    results_grid = par_grid_search(params, meta_train_eval, n_processes=n_processes)

    hs = np.zeros((T + 1, dim))
    inner_solver_test_list = []
    best_hp = {'alpha': [], 'lmbd': [], 'gamma':[]}

    for t in range(T + 1):
        def metric(res: Results):
            return np.mean(res.metrics[val_metric][t])

        best, _ = find_best_config(metric, results_grid)
        best_hp['alpha'].append(best['params']['alpha'])
        best_hp['lmbd'].append(best['params']['lmbd'])
        best_hp['gamma'].append(best['params']['gamma'])
        hs[t] = best['out'].hs[:t + 1].mean(axis=0)
        inner_solver_test_list.append(inner_solver_test_class(best['params']['lmbd'], hs[t], loss_class,
                                      gamma=best['params']['gamma']))

    metrics = eval_biases(data_test, inner_solver_test_list, metric_dict, verbose=verbose)
    results.add_hs(hs)
    results.add_metrics(metrics)
    results.add_parameters(best_hp)
    return results


