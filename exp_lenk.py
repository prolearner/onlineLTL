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

# if reg = False it conducts the classification experiments by thresholding all rates grater than 5 to one and
# the others to 0 and using the Hinge loss. Put reg = True for the regression experiments.
reg = False
n_train = [8]  # 8 train example were used in Argiryu et al. 2007
# candidate values for the hyperparameters grid search
lambdas = np.logspace(-3, 3, num=30)
alphas = np.logspace(-3, 3, num=30)


def exp_len():
    for n in n_train:
        lenk_multi_seed(seeds=list(range(10)), reg=False, n_train=n, inner_solver_str=['fista', 'ssubgd'],
                        lambdas=lambdas, alphas=alphas, inner_solver_test_str=['fista', 'ssubgd'],
                        n_processes=n_processes)


def exp_len_itl():
    exp_itl_only(seed=0, exp_str='lenkReg', inner_solver_test_str=['subgd'],
                 lambdas=[0.01],
                 verbose=5, n_tasks=120, n_tasks_test=40, n_train=None, gamma=None, n_processes=n_processes)


exp_len()
