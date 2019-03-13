import argparse
from experiments import delicious_multi_seed, exp_itl_only
import numpy as np

parser = argparse.ArgumentParser(description='LTL online numpy experiments')
parser.add_argument('--seed', type=int, default=0, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--n-processes', type=int, default=20, metavar='N',
                    help='n processes for grid search (default: 30)')


args = parser.parse_args()
n_processes = args.n_processes


def exp_del():  # 8 train example were used in Argiryu et al. 2007
    for n in [20, 100]:
        #lenk_multi_seed(n_train=i, n_processes=n_processes)
        delicious_multi_seed(seeds=list(range(10)),lambdas=np.logspace(-3, 3, num=5),
                             alphas=np.logspace(-3, 3, num=5),
                             n_train=n, inner_solver_str=['ssubgd'],
                             inner_solver_test_str=['ssubgd'], n_processes=n_processes)


def exp_del_itl():
    exp_itl_only(seed=0, exp_str='delicious', inner_solver_test_str=['ssubgd'],
                 lambdas=[3],
                 verbose=5, n_tasks=500, n_tasks_test=200, n_train=20, gamma=None, n_processes=n_processes)

#lenk_meta_val(reg=False, lambdas=1.9, alphas=0.6, inner_solver_test_str='ssubgd', inner_solver_str=['ssubgd'])
#exp_len()


exp_del_itl()