import numpy as np
import os

from algorithms import no_train_evaluation, LTL_evaluation, meta_ssgd, eval_biases, inner_solver_selector
from experiments_ICML import print_results, plot_results, save_results
from grid_search import HyperList, par_grid_search, find_best_config
from plots import plot, plot_resultsList
from utils import save_nparray, make_exp_dir, print_metric_mean_and_std, save_exp_parameters
import data
from losses import AbsoluteLoss, HingeLoss
from sklearn.metrics import explained_variance_score, accuracy_score


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
        self.parameters= None

    def add_parameters(self, params):
        self.parameters = params
        save_exp_parameters(params, self.save_dir, name=self.name+'-params')

    def add_metrics(self, metric_dict, y_label='Test Error'):
        self.metrics = metric_dict
        for m_name, metric in metric_dict.items():
            self._save_metric(metric_dict, m_name, y_label)

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

    params = {'h': h, 'lmbd':lambdas, 'gamma':gamma,
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


def exp(exp_str = 'exp1', seed = 0, lambdas = np.logspace(-6, 3, num=10), alphas = np.logspace(-6, 3, num=10),
        gamma = None, n_processes = 30, w_bar = 4, y_snr = 100, task_std = 1, n_tasks = 100, n_train = 100, n_dims = 30,
        n_tasks_test = 200, n_test = 100, val_perc = 0.0,
        exp_dir = EXP_FOLDER, inner_solver_str = ('ssubgd', 'fista'),
        inner_solver_test_str = ('ssubgd', 'fista'), show_plot=False, verbose =0):

    loss_class, tasks_gen, inner_exp_name,\
        metric_dict, val_metric = select_exp(exp_str, seed=seed, task_std=task_std, y_snr=y_snr,
                                             n_train_tasks=n_tasks, n_val_tasks=n_tasks_test, n_dims=n_dims,
                                             val_perc=val_perc, w_bar=w_bar)

    exp_name = 'grid_search' + inner_exp_name + 'seed' + str(seed) + 'vm' + val_metric + 'is' \
               + str(inner_solver_str) + 'ist' + str(inner_solver_test_str) + 'n' + str(n_train) + 'val_perc' + str(val_perc)

    exp_parameters = locals()
    print('parameters ' + exp_name, exp_parameters)

    data_train, oracle_train = tasks_gen(n_tasks=n_tasks, n_train=n_train, n_test=n_test, sel='train')
    data_valid, oracle_valid = tasks_gen(n_tasks=n_tasks_test, n_train=n_train, n_test=n_test, sel='val')
    data_test, oracle_test = tasks_gen(n_tasks=n_tasks_test, n_train=n_train, n_test=n_test, sel='test')

    exp_dir_path = make_exp_dir(os.path.join(exp_dir, exp_name))
    save_exp_parameters(exp_parameters, exp_dir_path)

    res_dict = {}
    for ts in inner_solver_test_str:
        inner_solver_test_class = inner_solver_selector(ts)

        # metaval for ITL
        results = Results(save_dir=exp_dir_path, do_plot=False, show_plot=show_plot, name='ITL-ts' + ts)
        h = np.zeros(tasks_gen.n_dims)
        itl_res = val(h, val_metric, lambdas, gamma, inner_solver_test_class, loss_class,
                      data_valid, data_test, metric_dict, results, n_processes=n_processes, verbose=verbose)

        res_dict[itl_res.name] = itl_res

        # metaval for MEAN
        if oracle_valid is not None:
            results = Results(save_dir=exp_dir_path, do_plot=False, show_plot=show_plot, name='MEAN-ts' + ts)
            h = oracle_valid['w_bar']
            oracle_res = val(h, val_metric, lambdas, gamma, inner_solver_test_class, loss_class,
                          data_valid, data_test, metric_dict, results, n_processes=n_processes, verbose=verbose)

            res_dict[oracle_res.name] = oracle_res

        for s in inner_solver_str:
            inner_solver_class = inner_solver_selector(s)

            results = Results(save_dir=exp_dir_path, do_plot=False, show_plot=show_plot, name='LTL-tr'+s +'ts'+ts)

            h0 = np.zeros(tasks_gen.n_dims)
            ltl_res = meta_val(val_metric=val_metric, h0=h0, alphas=alphas, lambdas=lambdas, gamma=gamma,
                               inner_solver_class=inner_solver_class,inner_solver_test_class=inner_solver_test_class,
                               loss_class=loss_class, data_train=data_train, data_valid=data_valid, data_test=data_test,
                               metric_dict=metric_dict, results=results, n_processes=n_processes, verbose=verbose)
            res_dict[ltl_res.name] = ltl_res

    plot_resultsList(n_tasks+1, res_dict, save_dir_path=exp_dir_path, show_plot=show_plot, filename='ltl_plots')
    return res_dict


def multi_seed(exp_str='exp1', seeds=list(range(10)), lambdas=np.logspace(-6, 3, num=10), alphas=np.logspace(-6, 3, num=10),
                 gamma=None, n_processes=30, w_bar=4, y_snr=100, task_std=1, n_tasks=100, n_train=100, n_dims=30,
                 n_tasks_test=200, n_test=100, val_perc=0.0, search_oracle=False,
                 exp_dir=EXP_FOLDER, inner_solver_str=('ssubgd', 'subgd'),
                 use_hyper_bounds=False, inner_solver_test_str=('ssubgd', 'subgd'), show_plot=True, save_res=True, verbose=1):

    loss_class, tasks_gen, inner_exp_name,\
        metric_dict, val_metric = select_exp(exp_str, seed=seeds[0], task_std=task_std, y_snr=y_snr,
                                             n_train_tasks=n_tasks, n_val_tasks=n_tasks_test, n_dims=n_dims,
                                             val_perc=val_perc, w_bar=w_bar)

    exp_name = 'grid_search' + inner_exp_name + 'over' + str(len(seeds)) + 'seeds' + 'is' \
               + str(inner_solver_str) + 'ist' + str(inner_solver_test_str) + 'n' + str(n_train) + 'val_perc' + str(val_perc)

    exp_parameters = locals()
    print('parameters ' + exp_str, exp_parameters)
    exp_dir_path = make_exp_dir(os.path.join(exp_dir, exp_name))

    res_list = []
    avg_res_dict = {}
    for s in seeds:
        r = exp(exp_str=exp_str, seed=s, lambdas=lambdas, alphas=alphas, gamma=gamma, n_processes=n_processes,
                w_bar=w_bar, y_snr=y_snr, task_std=task_std, n_tasks=n_tasks, n_train=n_train, n_dims=n_dims,
                n_tasks_test=n_tasks_test, n_test=n_test, exp_dir=exp_dir_path, val_perc=val_perc,
                inner_solver_str=inner_solver_str, inner_solver_test_str=inner_solver_test_str,
                show_plot=show_plot, verbose=verbose)

        res_list.append(r)
        for name in list(res_list[0].keys()):
            avg_res_dict[name] = Results(save_dir=exp_dir_path, do_plot=False, show_plot=False, name=name)
            metric_dict = {}
            for m_name in list(res_list[0][name].metrics.keys()):
                metric_dict[m_name] = [res[name].metrics[m_name] for res in res_list]
                last_axes = len(metric_dict[m_name][0].shape)-1
                metric_dict[m_name] = np.concatenate([np.expand_dims(np.mean(o, axis=last_axes), last_axes)
                                                      for o in metric_dict[m_name]], axis=last_axes)
            avg_res_dict[name].add_metrics(metric_dict)

        plot_resultsList(n_tasks + 1, avg_res_dict, save_dir_path=exp_dir_path, show_plot=show_plot, filename='plots',
                         title=str(len(res_list))+' runs')


def lenk_multi_seed(seeds=list(range(10)), lambdas=np.logspace(-1, 6, num=100), alphas=np.logspace(-1, 6, num=10), reg=False,
                 n_train=None,gamma=None, n_processes=30, n_tasks=100, n_val_tasks=40, exp_dir=EXP_FOLDER, inner_solver_str=('ssubgd', 'subgd'),
                 use_hyper_bounds=False, inner_solver_test_str=('ssubgd', 'fista'), show_plot=True, save_res=True, verbose=1):

    reg_str = '' if not reg else 'Reg'
    return multi_seed(exp_str='lenk'+reg_str, seeds=seeds, lambdas=lambdas, alphas=alphas,
                 gamma=gamma, n_processes=n_processes, w_bar=0, y_snr=0, task_std=0, n_tasks=n_tasks, n_train=n_train, n_dims=0,
                 n_tasks_test=n_val_tasks, n_test=0, exp_dir=exp_dir, inner_solver_str=inner_solver_str,
                 use_hyper_bounds=use_hyper_bounds, inner_solver_test_str=inner_solver_test_str, show_plot=show_plot,
                 save_res=save_res, verbose=verbose)


def school_multi_seed(seeds=list(range(10)), lambdas=np.logspace(-3, 3, num=10), alphas=np.logspace(-1, 6, num=10),
                 gamma=None, n_processes=30, n_tasks=75, n_val_tasks=25, exp_dir=EXP_FOLDER, inner_solver_str=('ssubgd', 'subgd'),
                 use_hyper_bounds=False, inner_solver_test_str='ssubgd', show_plot=True, save_res=True, verbose=1):

    return multi_seed(exp_str='school', seeds=seeds, lambdas=lambdas, alphas=alphas,
                 gamma=gamma, n_processes=n_processes, w_bar=0, y_snr=0, task_std=0, n_tasks=n_tasks, n_train=None, n_dims=0,
                 n_tasks_test=n_val_tasks, n_test=0, val_perc=0.5, exp_dir=exp_dir, inner_solver_str=inner_solver_str,
                 use_hyper_bounds=use_hyper_bounds, inner_solver_test_str=inner_solver_test_str, show_plot=show_plot,
                 save_res=save_res, verbose=verbose)


def select_exp(exp_str, seed=0, task_std=1, y_snr=10, val_perc=0.5, w_bar=4, n_dims=30,
               n_train_tasks=0, n_val_tasks=0):
    if exp_str == 'exp1':
        tasks_gen = data.data_generator.TasksGenerator(seed=seed, task_std=task_std, y_snr=y_snr, val_perc=val_perc, n_dims=n_dims,
                                       tasks_generation='exp1', w_bar=w_bar)
        exp_name = exp_str + 'w_bar' + str(w_bar) + 'taskstd' + str(task_std) + 'y_snr' + str(y_snr) + \
                   'dim' + str(n_dims)
        loss = AbsoluteLoss
        metric_dict = {}
        val_metric = 'loss'
    elif exp_str == 'exp2':
        tasks_gen = data.data_generator.TasksGenerator(seed=seed, task_std=task_std, y_snr=y_snr, val_perc=val_perc, n_dims=n_dims,
                                       tasks_generation='expclass', w_bar=w_bar)
        exp_name = exp_str + 'w_bar' + str(w_bar) + 'taskstd' + str(task_std) + 'y_snr' + str(y_snr) + \
                   'dim' + str(n_dims) + 'y_dist' + str(tasks_gen.y_dist)
        loss = HingeLoss
        val_metric = 'loss'
        metric_dict = {}
    elif exp_str == 'school':
        tasks_gen = data.data_load.RealDatasetGenerator(gen_f=data.data_load.schools_data_gen, seed=seed,
                                                        n_train_tasks=n_train_tasks,
                                                        n_val_tasks=n_val_tasks,
                                                        val_perc=val_perc)
        exp_name = 'expSchool' + 'n_tasks_val' + str(n_val_tasks) + 'n_tasks' + str(tasks_gen.n_tasks) \
                   + 'dim' + str(tasks_gen.n_dims)
        loss = AbsoluteLoss
        val_metric = 'loss'
        metric_dict = {'negexpvar': ne_exp_var}
    elif exp_str == 'lenk':
        tasks_gen = data.data_load.RealDatasetGenerator(gen_f=data.data_load.computer_data_gen,
                                                        seed=seed, n_train_tasks=n_train_tasks,
                                                        n_val_tasks=n_val_tasks,
                                                        val_perc=val_perc)
        exp_name = 'expLenk' + 'n_tasks_train' + str(n_train_tasks) + 'n_tasks_val' + str(n_val_tasks) \
                   + 'n_tasks' + str(tasks_gen.n_tasks) + 'dim' + str(tasks_gen.n_dims)
        loss = HingeLoss
        val_metric = 'loss'
        metric_dict = {}
    elif exp_str == 'lenkReg':
        tasks_gen = data.data_load.RealDatasetGenerator(gen_f=data.data_load.computer_data_ge_reg,
                                                        seed=seed, n_train_tasks=n_train_tasks,
                                                        n_val_tasks=n_val_tasks,
                                                        val_perc=val_perc)
        exp_name = 'expLenkReg' + 'n_tasks_train' + str(n_train_tasks) + 'n_tasks_val' + str(n_val_tasks) \
                   + 'n_tasks' + str(tasks_gen.n_tasks) + 'dim' + str(tasks_gen.n_dims)
        loss = AbsoluteLoss
        val_metric = 'loss'
        metric_dict = {}
    else:
        raise NotImplementedError('exp: {} not implemented'.format(exp_str))

    return loss, tasks_gen, exp_name, metric_dict, val_metric


def dg(d, k): return None if d is None else d[k]


def accuracy(y_true, y_pred):
    return accuracy_score(y_true.astype(int), y_pred.astype(int))


def ne_exp_var(y_true, y_pred):
    return - explained_variance_score(y_true, y_pred)


if __name__ == '__main__':
    lenk_multi_seed(lambdas=[0.1, 10], alphas=[0.1, 10], reg=True,
                    inner_solver_test_str=['ssubgd', 'fista'])