import argparse
from experiments_ICML import lenk_multi_seed, lenk_meta_val


parser = argparse.ArgumentParser(description='LTL online numpy experiments')
parser.add_argument('--seed', type=int, default=0, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--n-processes', type=int, default=30, metavar='N',
                    help='n processes for grid search (default: 30)')


args = parser.parse_args()
n_processes = args.n_processes


def exp_len():
    for i in [16]:
        #lenk_multi_seed(n_train=i, n_processes=n_processes)
        lenk_multi_seed(reg=True, n_train=i, inner_solver_str=['ssubgd', 'subgd', 'ista', 'fista'], inner_solver_test_str='ssubgd')


#lenk_meta_val(reg=False, lambdas=1.9, alphas=0.6, inner_solver_test_str='ssubgd', inner_solver_str=['ssubgd'])
exp_len()
