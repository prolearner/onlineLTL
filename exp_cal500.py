import argparse
from experiments import cal500_multi_seed, exp_itl_only
import numpy as np

parser = argparse.ArgumentParser(description='LTL online numpy experiments')
parser.add_argument('--seed', type=int, default=0, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--n-processes', type=int, default=20, metavar='N',
                    help='n processes for grid search (default: 30)')


args = parser.parse_args()
n_processes = args.n_processes


def exp_cal():
    #lenk_multi_seed(n_train=i, n_processes=n_processes)
    for nt in [None, 50, 100, None]:
        cal500_multi_seed(seeds=list(range(10)), n_train=nt, inner_solver_str=['ssubgd'],
                            inner_solver_test_str=['ssubgd'], n_processes=n_processes)


def exp_cal_itl():
    exp_itl_only(seed=0, exp_str='cal500', inner_solver_test_str=['ssubgd'],
                 lambdas=np.logspace(-6, 6, 20),
                 verbose=3, n_tasks=90, n_tasks_test=40, n_train=50, gamma=None, n_processes=n_processes)

#lenk_meta_val(reg=False, lambdas=1.9, alphas=0.6, inner_solver_test_str='ssubgd', inner_solver_str=['ssubgd'])
#exp_len()


exp_cal()