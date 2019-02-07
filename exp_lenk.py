import argparse
from experiments import lenk_multi_seed, exp_itl_only
import numpy as np

parser = argparse.ArgumentParser(description='LTL online numpy experiments')
parser.add_argument('--seed', type=int, default=0, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--n-processes', type=int, default=30, metavar='N',
                    help='n processes for grid search (default: 30)')


args = parser.parse_args()
n_processes = args.n_processes


def exp_len():  # 8 train example were used in Argiryu et al. 2007
    for i in [8, 16]:
        #lenk_multi_seed(n_train=i, n_processes=n_processes)
        lenk_multi_seed(reg=False, n_train=i, inner_solver_str=['ssubgd', 'fista'],
                        inner_solver_test_str=['ssubgd', 'fista'], n_processes=n_processes)


def exp_len_itl():
    exp_itl_only(seed=0, exp_str='lenkReg', inner_solver_test_str=['subgd'],
                 lambdas=[0.01],
                 verbose=5, n_tasks=120, n_tasks_test=40, n_train=None, gamma=None, n_processes=n_processes)

#lenk_meta_val(reg=False, lambdas=1.9, alphas=0.6, inner_solver_test_str='ssubgd', inner_solver_str=['ssubgd'])
#exp_len()


exp_len()
