import argparse

from experiments import multi_seed

parser = argparse.ArgumentParser(description='LTL online numpy experiments')
parser.add_argument('--seed', type=int, default=0, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--n-processes', type=int, default=30, metavar='N',
                    help='n processes for grid search (default: 30)')


args = parser.parse_args()
n_processes = args.n_processes

for n_train in [10, 50]:
    for task_std in [1, 2, 4]:
        for exp in ['exp1', 'exp2']:
            multi_seed(exp, n_train=n_train, n_tasks=500, w_bar=4, y_snr=10, task_std=task_std, use_hyper_bounds=False,
                       inner_solver_str=['fista', 'ssubgd'], inner_solver_test_str=['fista', 'ssubgd'],
                       n_processes=n_processes)


def exp_class():
    for n_train in [10, 50, 100]:
        for tasks_std in [0.5, 1, 2, 3, 4]:
            multi_seed('exp2', n_train=n_train, n_tasks=1000, w_bar=4, y_snr=1, task_std=tasks_std,
                            use_hyper_bounds=True, inner_solver_str=['ssubgd'], search_oracle=False)


def exp_reg():
    for n_train in [10, 50, 100]:
        for y_snr in [0.5, 1, 2]:
            for tasks_std in [0.5, 1, 2, 4]:
                    multi_seed('exp1', n_train=n_train, n_tasks=1000, w_bar=4, y_snr=y_snr,
                                   task_std=tasks_std, use_hyper_bounds=True, inner_solver_str=['ssubgd'])


#exp_reg()
#exp_class()
#exp_meta_val('exp2', n_train=10, n_tasks=600, w_bar=4, y_snr=1, task_std=1, lambdas=1, alphas=1, use_hyper_bounds=True, inner_solver_str=['ssubgd'])
