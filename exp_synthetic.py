import argparse

from experiments import multi_seed

parser = argparse.ArgumentParser(description='LTL online numpy experiments')
parser.add_argument('--seed', type=int, default=0, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--n-processes', type=int, default=30, metavar='N',
                    help='n processes for grid search (default: 30)')


args = parser.parse_args()
n_processes = args.n_processes


def exp_class():
    multi_seed('exp2', n_train=10, n_dims=30, n_tasks=1000, w_bar=4, y_snr=10, task_std=1,
               inner_solver_str=['fista', 'ssubgd'], inner_solver_test_str=['fista', 'ssubgd'], n_processes=n_processes)


def exp_reg():
    multi_seed('exp1', n_train=10, n_dims=30, n_tasks=1000, w_bar=4, y_snr=10, task_std=1,
               inner_solver_str=['fista', 'ssubgd'], inner_solver_test_str=['fista', 'ssubgd'], n_processes=n_processes)


exp_reg()
exp_class()