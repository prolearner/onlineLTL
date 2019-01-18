import argparse
import copy
import subprocess
import time
from multiprocessing import Pool


class HyperList:
    def __init__(self, list):
        self.list = list


def grid_search(args_dict, exp_f, result_dict):
    has_hypers = False
    for k, v in args_dict.items():
        if type(v) == HyperList:
            has_hypers = True
            for h in v.list:
                new_args = copy.deepcopy(args_dict)
                new_args[k] = h
                grid_search(new_args, exp_f, result_dict)
            break

    if not has_hypers:
        exp_f(**args_dict)


def par_grid_search(args_dict, exp_f, n_processes=10):
    pool = Pool(processes=n_processes)              # start n_processes worker processes

    results = []
    _inner_par_grid_search(args_dict, exp_f, results, pool)
    pool.close()
    pool.join()

    for r in results:
        r['out'] = r['out'].get()
    return results


def find_best_config(metric_f, results):
    best_index = 0
    best_result = results[0]
    for i, r in enumerate(results):
        if metric_f(r['out']) < metric_f(best_result['out']):
            best_index = i
            best_result = r
    return best_result, best_index


def _inner_par_grid_search(args_dict, exp_f, results, pool):
    has_hypers = False
    for k, v in args_dict.items():
        if type(v) == HyperList:
            has_hypers = True
            for h in v.list:
                new_args = copy.deepcopy(args_dict)
                new_args[k] = h
                _inner_par_grid_search(new_args, exp_f, results, pool)
            break

    if not has_hypers:
        results.append({'params': args_dict, 'out': pool.apply_async(exp_f, kwds=args_dict)})


