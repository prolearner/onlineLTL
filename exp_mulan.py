import argparse
from experiments import mulan_multi_seed, exp_itl_only
import numpy as np

parser = argparse.ArgumentParser(description='LTL online numpy experiments')
parser.add_argument('--seed', type=int, default=0, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--n-processes', type=int, default=4, metavar='N',
                    help='n processes for grid search (default: 30)')


args = parser.parse_args()
n_processes = args.n_processes

dataset_name = 'bibtex'

mulan_dict = {'Corel5k': {'n_tasks':100, 'n_val_tasks': 100, 'n_train': 20},
              'CAL500': {'n_tasks':100, 'n_val_tasks': 40, 'n_train': 100},
              'delicious': {'n_tasks':500, 'n_val_tasks': 183, 'n_train': 20},
              'bookmarks': {'n_tasks':100, 'n_val_tasks': 40, 'n_train': 20},
              'bibtex': {'n_tasks': 80, 'n_val_tasks': 40}
              }


def exp():
    for n in [5, 10, 20, 100]:
        mulan_multi_seed(data_name=dataset_name, seeds=list(range(10)),lambdas=np.logspace(-3, 3, num=5),
                             alphas=np.logspace(-3, 3, num=5), inner_solver_str=['ssubgd'],
                             inner_solver_test_str=['ssubgd'], n_train=n,
                             n_processes=n_processes, **mulan_dict[dataset_name])


def exp_itl():
    mulan_dict[dataset_name]['n_tasks_test'] = mulan_dict[dataset_name]['n_val_tasks']
    del mulan_dict[dataset_name]['n_val_tasks']
    exp_itl_only(seed=0, exp_str=dataset_name, inner_solver_test_str=['ssubgd'],
                 lambdas=np.logspace(-3, 3, num=5),
                 verbose=5, n_processes=n_processes, **mulan_dict[dataset_name])

exp()