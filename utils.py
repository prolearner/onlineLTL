import datetime

import numpy as np
import json
import os


def _apply_to_obj(func, key, value, *args, **kwargs):
    return func(key, value, *args, **kwargs)


def _apply_to_dict(func, dict, *args, **kwargs):
    out = None
    for k, v in dict.items():
        out = func(k, v, *args, **kwargs)
    return out


def apply_to_dict_or_obj(func, dict_or_obj, *args, **kwargs):
    if isinstance(dict_or_obj, dict):
        return _apply_to_dict(func, dict_or_obj, *args, **kwargs)
    else:
        return _apply_to_obj(func, '', dict_or_obj, *args, **kwargs)


def print_metric_mean_and_std(metric, name=""):

    def print_metric(key, metric, name=""):
        if len(metric.shape) > 1:
            metric = metric[-1]
        print('-' * 10 + name + '-' + key + ' mean std', np.mean(metric), np.std(metric))

    apply_to_dict_or_obj(print_metric, metric, name=name)


def save_nparray(array, name, exp_dir_path):

    def save(key, array, name):
        np.savetxt(os.path.join(exp_dir_path, name + "-" + key + ".csv"), array, delimiter=",")

    apply_to_dict_or_obj(save, array, name=name)


def is_jsonable(x):
    try:
        json.dumps(x)
        return True
    except:
        return False


class Objdict(dict):
    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError("No such attribute: " + name)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        if name in self:
            del self[name]
        else:
            raise AttributeError("No such attribute: " + name)


class PrintLevels:
    inner_train = 3
    inner_eval = 2
    outer_train = 1
    outer_eval = 0


def make_exp_dir(experiment_name, time=True):
    now = '-' + str(datetime.datetime.now()) if time else ''
    experiment_path = experiment_name + now
    os.makedirs(experiment_path)
    return experiment_path


def save_exp_parameters(exp_parameters, exp_dir_path, name='parameters'):
    param_serializable = {}
    for k, v in exp_parameters.items():
        if is_jsonable(v):
            param_serializable[k] = v
        else:
            param_serializable[k] = str(v)

    import json
    with open(os.path.join(exp_dir_path, name+'.txt'), 'w') as file:
        file.write(json.dumps(param_serializable))  # use `json.loads` to do the reverse